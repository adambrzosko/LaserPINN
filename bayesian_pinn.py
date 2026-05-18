"""
Bayesian PINN for Uncertainty Quantification (#11)

Uses Monte Carlo Dropout to obtain posterior distributions over laser parameters
instead of point estimates. After training in inverse mode, multiple forward
passes with dropout enabled produce samples from an approximate posterior.

Outputs:
  images/bayesian_pinn/posterior_distributions.png
  images/bayesian_pinn/predictive_uncertainty.png
  images/bayesian_pinn/training_convergence.png
"""

import math
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from dfb_laser import DFBLaserParams, q, h, c
from laser_pinn import (
    Scales, LearnableParams, generate_reference, make_collocation,
    physics_residuals, compute_loss, subsample, DTYPE, device,
)

# ── Output directory ─────────────────────────────────────────────────────────
os.makedirs('images/bayesian_pinn', exist_ok=True)


# ── Bayesian PINN with MC Dropout ────────────────────────────────────────────

class BayesianLaserPINN(nn.Module):
    """
    PINN for laser rate equations with dropout for Bayesian inference.

    Same architecture as LaserPINN but with nn.Dropout(p) after each hidden
    layer. Keeping dropout active during inference (model.train()) allows
    Monte Carlo Dropout sampling from an approximate posterior.
    """

    def __init__(self, hidden_dim=128, n_layers=6, n_out=3, dropout_p=0.1):
        super().__init__()
        self.n_out = n_out
        self.dropout_p = dropout_p
        self.input_layer = nn.Linear(2, hidden_dim)
        self.hidden = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.dropouts = nn.ModuleList(
            [nn.Dropout(p=dropout_p) for _ in range(n_layers)]
        )
        self.output_layer = nn.Linear(hidden_dim, n_out)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, tau, i_norm):
        x = torch.cat([tau, i_norm], dim=-1)
        h = torch.tanh(self.input_layer(x))
        for k, (layer, drop) in enumerate(zip(self.hidden, self.dropouts)):
            h_new = torch.tanh(layer(h))
            h_new = drop(h_new)
            if k % 2 == 1:
                h_new = h_new + h  # skip connection every 2 layers
            h = h_new
        out = self.output_layer(h)
        return tuple(out[:, i:i+1] for i in range(self.n_out))


# ── Physics residuals for Bayesian model ─────────────────────────────────────

def bayesian_physics_residuals(model, tau, i_norm, params, scales, learnable=None):
    """
    Compute rate-equation residuals via autograd (same as physics_residuals
    from laser_pinn.py but accepts BayesianLaserPINN).
    """
    tau = tau.requires_grad_(True)
    n, sigma, omega_norm = model(tau, i_norm)

    ones = torch.ones_like(n)
    dn_dtau = torch.autograd.grad(n, tau, ones, create_graph=True, retain_graph=True)[0]
    dsigma_dtau = torch.autograd.grad(sigma, tau, ones, create_graph=True, retain_graph=True)[0]

    def P(name):
        if learnable is not None:
            return learnable.get(name)
        return torch.tensor(getattr(params, name), dtype=DTYPE, device=device)

    N = scales.N_tr + n * scales.N_scale
    sigma_clamped = torch.clamp(sigma, -15.0, 5.0)
    S = scales.S_ref * (10.0 ** sigma_clamped)
    I = i_norm * scales.I_th

    a = P('a')
    N_tr = P('N_tr')
    epsilon = P('epsilon')
    g = a * (N - N_tr) / (1.0 + epsilon * S)

    A_coeff = P('A')
    B_coeff = P('B')
    C_coeff = P('C')
    R_sp = A_coeff * N + B_coeff * N**2 + C_coeff * N**3

    Gamma = P('Gamma')
    v_g = torch.tensor(params.v_g, dtype=DTYPE, device=device)
    V = torch.tensor(params.V, dtype=DTYPE, device=device)
    tau_p = torch.tensor(params.tau_p, dtype=DTYPE, device=device)
    beta_sp = P('beta_sp')
    alpha_H = P('alpha_H')

    q_val = torch.tensor(q, dtype=DTYPE, device=device)
    t_ref = torch.tensor(scales.t_ref, dtype=DTYPE, device=device)
    ln10 = torch.tensor(math.log(10.0), dtype=DTYPE, device=device)
    N_scale_t = torch.tensor(scales.N_scale, dtype=DTYPE, device=device)
    omega_scale_t = torch.tensor(scales.omega_scale, dtype=DTYPE, device=device)

    rhs_n = (t_ref / N_scale_t) * (I / (q_val * V) - R_sp - Gamma * v_g * g * S)
    rhs_sigma = (t_ref / ln10) * (
        Gamma * v_g * g - 1.0 / tau_p + beta_sp * B_coeff * N**2 / S
    )
    rhs_omega = (t_ref / omega_scale_t) * 0.5 * alpha_H * (
        Gamma * v_g * a * (N - N_tr) - 1.0 / tau_p
    )

    return dn_dtau - rhs_n, dsigma_dtau - rhs_sigma, omega_norm - rhs_omega


def bayesian_compute_loss(model, tau_colloc, i_norm_colloc, params, scales,
                          tau_data=None, i_norm_data=None,
                          n_data=None, sigma_data=None, omega_data=None,
                          learnable=None,
                          lam_phys=1.0, lam_data=1.0, causal_eps=1.0):
    """Total PINN loss = physics residual + data fitting (for Bayesian model)."""
    res_n, res_sigma, res_omega = bayesian_physics_residuals(
        model, tau_colloc, i_norm_colloc, params, scales, learnable
    )

    r2 = (res_n.detach()**2 + res_sigma.detach()**2 + res_omega.detach()**2)
    r2_clamped = torch.clamp(r2, max=50.0)
    w_causal = torch.exp(-causal_eps * torch.cumsum(r2_clamped, dim=0))
    w_causal = w_causal / (w_causal.mean() + 1e-30)

    L_phys = torch.mean(w_causal * (res_n**2 + res_sigma**2 + res_omega**2))

    L_data = torch.tensor(0.0, dtype=DTYPE, device=device)
    if n_data is not None:
        tau_d = tau_data.requires_grad_(False)
        n_pred, sig_pred, om_pred = model(tau_d, i_norm_data)
        L_data = torch.mean((n_pred - n_data)**2
                            + (sig_pred - sigma_data)**2
                            + (om_pred - omega_data)**2)

    loss = lam_phys * L_phys + lam_data * L_data
    return loss, {
        'total': loss.item(),
        'physics': L_phys.item(),
        'data': L_data.item(),
    }


# ── Training ─────────────────────────────────────────────────────────────────

def train_bayesian_inverse(params_true, I_bias, learn_which, scales,
                           ref_data, noisy_sigma,
                           perturbation=0.3, epochs=2000, lr_net=5e-4,
                           lr_params=1e-2, verbose=True):
    """
    Train Bayesian PINN in inverse mode with dropout enabled.

    The network and learnable parameters are trained simultaneously:
    - Network fits the noisy data
    - Physics residuals regularize
    - Learnable parameters are driven toward true values
    """
    from dataclasses import replace

    model = BayesianLaserPINN(hidden_dim=128, n_layers=6, dropout_p=0.1)
    model = model.to(device).to(DTYPE)

    # Perturbed initial guesses for learnable parameters
    perturbed_vals = {}
    for name in learn_which:
        true_val = getattr(params_true, name)
        perturbed_vals[name] = true_val * (1.0 + perturbation)
    params_guess = replace(params_true, **perturbed_vals)
    learnable = LearnableParams(params_guess, learn_which).to(device)

    # Prepare data tensors
    tau_data = ref_data['tau']
    i_norm_data = ref_data['i_norm']
    n_data = ref_data['n']
    omega_data = ref_data['omega']

    tau_max = float(tau_data.max())

    # Optimizer for both network and parameters
    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': lr_net},
        {'params': learnable.parameters(), 'lr': lr_params},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    history = []
    param_history = {name: [] for name in learn_which}
    t_start = time.time()

    for epoch in range(epochs):
        # Physics weight ramp
        frac = min(epoch / (epochs * 0.5), 1.0)
        lam_phys = 0.01 + 0.99 * frac

        tau_colloc = make_collocation(tau_max, 1500)
        i_norm_colloc = torch.full_like(tau_colloc, I_bias / scales.I_th)

        optimizer.zero_grad()
        loss, info = bayesian_compute_loss(
            model, tau_colloc, i_norm_colloc, params_guess, scales,
            tau_data=tau_data, i_norm_data=i_norm_data,
            n_data=n_data, sigma_data=noisy_sigma, omega_data=omega_data,
            learnable=learnable,
            lam_phys=lam_phys, lam_data=10.0, causal_eps=1.0,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(learnable.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        history.append(info)
        for name in learn_which:
            param_history[name].append(learnable.get(name).item())

        if verbose and epoch % 500 == 0:
            elapsed = time.time() - t_start
            param_str = ', '.join(
                f'{n}={learnable.get(n).item():.3e}' for n in learn_which
            )
            print(f"    epoch {epoch:5d}/{epochs} | loss {info['total']:.3e} | "
                  f"{param_str} | {elapsed:.1f}s")

    elapsed = time.time() - t_start
    if verbose:
        print(f"  Training complete in {elapsed:.1f}s")

    return model, learnable, history, param_history


# ── MC Dropout inference ─────────────────────────────────────────────────────

def estimate_param_uncertainty(model, learnable, tau_data, i_norm_data,
                               noisy_sigma, n_data, omega_data,
                               params_guess, scales, n_mc_steps=50):
    """
    Estimate parameter posterior uncertainty using a local Laplace approximation.

    For each MC dropout sample of the network, run a few optimization steps on
    the parameters from a slightly perturbed starting point. This gives samples
    from the approximate posterior over parameters conditioned on network
    variability from dropout.
    """
    import copy
    from dataclasses import replace as dc_replace

    base_values = {name: learnable.get(name).item() for name in learnable.learn_which}
    param_samples = {name: [] for name in learnable.learn_which}

    model.train()  # dropout active

    for i in range(n_mc_steps):
        # Create a fresh learnable with small perturbation around converged values
        perturbed_vals = {}
        for name in learnable.learn_which:
            # Perturb by ~2% to explore local posterior
            perturb = 1.0 + 0.02 * np.random.randn()
            perturbed_vals[name] = base_values[name] * perturb
        params_pert = dc_replace(params_guess, **perturbed_vals)
        learn_temp = LearnableParams(params_pert, learnable.learn_which).to(device)

        # Short optimization (10 steps) with current dropout mask
        opt_temp = torch.optim.Adam(learn_temp.parameters(), lr=5e-3)
        tau_max = float(tau_data.max())

        for step in range(10):
            opt_temp.zero_grad()
            tau_colloc = make_collocation(tau_max, 500)
            i_norm_colloc = torch.full_like(tau_colloc, i_norm_data[0].item())

            loss, _ = bayesian_compute_loss(
                model, tau_colloc, i_norm_colloc, params_guess, scales,
                tau_data=tau_data, i_norm_data=i_norm_data,
                n_data=n_data, sigma_data=noisy_sigma, omega_data=omega_data,
                learnable=learn_temp,
                lam_phys=1.0, lam_data=10.0,
            )
            loss.backward()
            opt_temp.step()

        for name in learnable.learn_which:
            param_samples[name].append(learn_temp.get(name).item())

    # Convert to arrays
    for name in learnable.learn_which:
        param_samples[name] = np.array(param_samples[name])

    return param_samples


def mc_dropout_inference(model, learnable, tau, i_norm, scales, n_samples=200):
    """
    Perform MC Dropout inference: run n_samples forward passes with dropout
    enabled to sample from the approximate posterior over predictions.

    The dropout creates stochastic variation in network outputs, providing
    uncertainty estimates on N(t) and S(t) waveforms.

    Returns:
        predictions: dict with 'N', 'S' arrays of shape (n_samples, n_points)
    """
    model.train()  # keep dropout active

    n_points = tau.shape[0]
    N_samples = np.zeros((n_samples, n_points))
    S_samples = np.zeros((n_samples, n_points))

    with torch.no_grad():
        for i in range(n_samples):
            n_pred, sigma_pred, omega_pred = model(tau, i_norm)

            N_pred = (scales.N_tr + n_pred.numpy().flatten() * scales.N_scale)
            sigma_clamped = np.clip(sigma_pred.numpy().flatten(), -15.0, 5.0)
            S_pred = scales.S_ref * 10.0**sigma_clamped

            N_samples[i] = N_pred
            S_samples[i] = S_pred

    return {'N': N_samples, 'S': S_samples}


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_posterior_distributions(param_samples, params_true, learn_which, savepath):
    """Histogram of each inferred parameter from MC Dropout with true value marked."""
    n_params = len(learn_which)
    fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 4.5))
    if n_params == 1:
        axes = [axes]

    param_labels = {
        'alpha_H': r'$\alpha_H$ (Henry factor)',
        'epsilon': r'$\varepsilon$ (gain compression, m$^3$)',
        'a': r'$a$ (differential gain, m$^2$)',
        'beta_sp': r'$\beta_{sp}$ (spontaneous coupling)',
    }

    for ax, name in zip(axes, learn_which):
        samples = param_samples[name]
        true_val = getattr(params_true, name)

        # Normalize for display
        mean_val = np.mean(samples)
        std_val = np.std(samples)
        ci_low = np.percentile(samples, 2.5)
        ci_high = np.percentile(samples, 97.5)

        ax.hist(samples, bins=30, density=True, alpha=0.7, color='steelblue',
                edgecolor='white', linewidth=0.5)
        ax.axvline(true_val, color='red', linestyle='--', linewidth=2,
                   label=f'True = {true_val:.3e}')
        ax.axvline(mean_val, color='navy', linestyle='-', linewidth=2,
                   label=f'Mean = {mean_val:.3e}')
        ax.axvspan(ci_low, ci_high, alpha=0.15, color='orange',
                   label=f'95% CI')

        label = param_labels.get(name, name)
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel('Density')
        ax.set_title(f'{name} posterior')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('MC Dropout Posterior Distributions', fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {savepath}")


def plot_predictive_uncertainty(predictions, ref_data, scales, params_true, savepath):
    """Mean prediction +/- 2 sigma for N(t) and S(t) with ground truth overlay."""
    t_ns = ref_data['t'] * 1e9

    N_samples = predictions['N']
    S_samples = predictions['S']

    N_mean = np.mean(N_samples, axis=0)
    N_std = np.std(N_samples, axis=0)
    S_mean = np.mean(S_samples, axis=0)
    S_std = np.std(S_samples, axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle('Predictive Uncertainty (MC Dropout, 200 samples)', fontsize=13)

    # Carrier density
    ax = axes[0]
    ax.plot(t_ns, ref_data['N'] * 1e-24, 'k-', lw=2, label='Ground truth', zorder=5)
    ax.plot(t_ns, N_mean * 1e-24, 'r--', lw=1.5, label='MC mean', zorder=4)
    ax.fill_between(t_ns,
                    (N_mean - 2*N_std) * 1e-24,
                    (N_mean + 2*N_std) * 1e-24,
                    alpha=0.3, color='salmon', label=r'Mean $\pm$ 2$\sigma$')
    ax.set_ylabel('Carrier density ($\\times 10^{24}$ m$^{-3}$)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Photon density
    ax = axes[1]
    ax.semilogy(t_ns, ref_data['S'], 'k-', lw=2, label='Ground truth', zorder=5)
    ax.semilogy(t_ns, np.maximum(S_mean, 1e-10), 'r--', lw=1.5,
                label='MC mean', zorder=4)
    ax.fill_between(t_ns,
                    np.maximum(S_mean - 2*S_std, 1e-10),
                    S_mean + 2*S_std,
                    alpha=0.3, color='salmon', label=r'Mean $\pm$ 2$\sigma$')
    ax.set_ylabel('Photon density (m$^{-3}$)')
    ax.set_xlabel('Time (ns)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {savepath}")


def plot_training_convergence(history, savepath):
    """Loss vs epoch."""
    epochs = np.arange(len(history))
    total = np.array([h['total'] for h in history])
    phys = np.array([h['physics'] for h in history])
    data = np.array([h['data'] for h in history])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(epochs, total, 'k-', lw=1.5, label='Total', alpha=0.8)
    ax.semilogy(epochs, phys, 'b-', lw=1, label='Physics', alpha=0.7)
    ax.semilogy(epochs, np.maximum(data, 1e-30), 'g-', lw=1, label='Data', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Bayesian PINN Training Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {savepath}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 70)
    print("  Bayesian PINN for Uncertainty Quantification")
    print("  (MC Dropout posterior over laser parameters)")
    print("=" * 70)

    # ── Setup ────────────────────────────────────────────────────────────────
    params = DFBLaserParams()
    scales = Scales(params)
    I_th = scales.I_th
    I_bias = 3.0 * I_th  # Step response at 3x threshold

    learn_which = ['alpha_H', 'epsilon', 'a', 'beta_sp']

    print(f"\n  Laser: 1550 nm InGaAsP/InP DFB")
    print(f"  I_th = {I_th*1e3:.2f} mA")
    print(f"  I_bias = {I_bias*1e3:.2f} mA ({I_bias/I_th:.1f}x threshold)")
    print(f"  Parameters to learn: {learn_which}")
    print(f"  MC Dropout rate: p=0.1")
    print()

    # ── Step 1: Generate synthetic experimental data ─────────────────────────
    print("  Step 1: Generating synthetic experimental data...")
    t_end = 10e-9
    n_points = 2000

    ref = generate_reference(params, scales, I_bias, t_end, n_points)

    # Add noise to intensity only (SNR = 25 dB)
    # SNR = 25 dB => noise_power = signal_power / 10^(25/10)
    torch.manual_seed(42)
    S_signal = ref['sigma']  # log10(S/S_ref), work in log domain for noise
    snr_linear = 10.0**(25.0 / 10.0)
    signal_power = torch.mean(S_signal**2)
    noise_std = torch.sqrt(signal_power / snr_linear)
    noisy_sigma = S_signal + noise_std * torch.randn_like(S_signal)

    print(f"    Generated {n_points} points, SNR = 25 dB on intensity")
    print(f"    Noise std (sigma domain): {noise_std.item():.4f}")

    # ── Step 2: Train Bayesian PINN in inverse mode ──────────────────────────
    print("\n  Step 2: Training Bayesian PINN (inverse mode, 2000 epochs)...")
    print(f"    Learnable params: {learn_which}")
    print(f"    Initial perturbation: +30%")
    print()

    model, learnable, history, param_history = train_bayesian_inverse(
        params, I_bias, learn_which, scales,
        ref_data=ref, noisy_sigma=noisy_sigma,
        perturbation=0.3, epochs=2000,
        lr_net=5e-4, lr_params=1e-2,
    )

    # ── Step 3: MC Dropout inference for predictive uncertainty ────────────────
    print("\n  Step 3: MC Dropout inference (200 forward passes for waveforms)...")
    n_mc_samples = 200

    predictions = mc_dropout_inference(
        model, learnable, ref['tau'], ref['i_norm'], scales,
        n_samples=n_mc_samples,
    )

    # ── Step 4: Parameter posterior via local re-optimization ────────────────
    print("\n  Step 4: Estimating parameter posterior (50 MC re-optimization steps)...")
    from dataclasses import replace as dc_replace
    perturbed_vals = {name: getattr(params, name) * 1.3 for name in learn_which}
    params_guess = dc_replace(params, **perturbed_vals)

    param_samples_mc = estimate_param_uncertainty(
        model, learnable, ref['tau'], ref['i_norm'],
        noisy_sigma, ref['n'], ref['omega'],
        params_guess, scales, n_mc_steps=50,
    )

    # ── Step 5: Compute posterior statistics ─────────────────────────────────
    print("\n  Step 5: Posterior statistics")
    print()
    print(f"  {'Parameter':<12} {'True':>12} {'Mean':>12} {'Std':>12} "
          f"{'95% CI low':>12} {'95% CI high':>12} {'Rel.Err%':>8}")
    print(f"  {'='*80}")

    for name in learn_which:
        true_val = getattr(params, name)
        samples = param_samples_mc[name]
        mean_val = np.mean(samples)
        std_val = np.std(samples)
        ci_low = np.percentile(samples, 2.5)
        ci_high = np.percentile(samples, 97.5)
        rel_err = abs(mean_val - true_val) / true_val * 100

        print(f"  {name:<12} {true_val:>12.4e} {mean_val:>12.4e} {std_val:>12.4e} "
              f"{ci_low:>12.4e} {ci_high:>12.4e} {rel_err:>7.2f}%")

    # Check if true values are within 95% CI
    print()
    print("  Coverage check (true value within 95% CI?):")
    for name in learn_which:
        true_val = getattr(params, name)
        samples = param_samples_mc[name]
        ci_low = np.percentile(samples, 2.5)
        ci_high = np.percentile(samples, 97.5)
        covered = ci_low <= true_val <= ci_high
        status = "YES" if covered else "NO"
        print(f"    {name:<12}: {status}")

    # ── Step 6: Predictive uncertainty on waveforms ──────────────────────────
    print("\n  Step 6: Computing predictive uncertainty bands...")
    N_mean = np.mean(predictions['N'], axis=0)
    S_mean = np.mean(predictions['S'], axis=0)
    N_std = np.std(predictions['N'], axis=0)
    S_std = np.std(predictions['S'], axis=0)

    # Relative uncertainty
    mask_S = ref['S'] > ref['S'].max() * 1e-3
    rel_unc_N = np.mean(N_std / np.abs(N_mean)) * 100
    rel_unc_S = np.mean(S_std[mask_S] / np.abs(S_mean[mask_S])) * 100
    print(f"    Mean relative uncertainty: N = {rel_unc_N:.2f}%, S = {rel_unc_S:.2f}%")

    # ── Step 7: Generate figures ─────────────────────────────────────────────
    print("\n  Step 7: Generating figures...")

    plot_posterior_distributions(
        param_samples_mc, params, learn_which,
        'images/bayesian_pinn/posterior_distributions.png'
    )

    plot_predictive_uncertainty(
        predictions, ref, scales, params,
        'images/bayesian_pinn/predictive_uncertainty.png'
    )

    plot_training_convergence(
        history,
        'images/bayesian_pinn/training_convergence.png'
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  Bayesian PINN complete.")
    print(f"  MC Dropout waveform samples: {n_mc_samples}")
    print(f"  MC parameter re-optimization steps: 50")
    print(f"  Training epochs: 2000")
    print(f"  Dropout rate: p=0.1")
    print("  Figures saved to images/bayesian_pinn/")
    print("=" * 70)
