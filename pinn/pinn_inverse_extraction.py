"""
PINN Inverse Parameter Extraction (#7)

Train a physics-informed neural network on laser rate equations with key
parameters (alpha_H, epsilon, beta_sp, Gamma*a) as learnable unknowns.
Generates synthetic "experimental" data from the ODE solver with known
ground-truth parameters, adds realistic measurement noise, then lets the
PINN infer the parameters from intensity-only data (no phase information).

Tests robustness across several SNR levels (20, 25, 30 dB).

Figures saved to images/pinn_inverse/:
  1. parameter_convergence.png — learned params vs epoch
  2. fit_quality.png — PINN prediction vs noisy data vs ground truth
  3. noise_robustness.png — parameter accuracy vs SNR
"""

import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
from dataclasses import replace

from core.dfb_laser import DFBLaserParams, solve_transient, q, h, c  # noqa: F401
from gsdfb.pinn_utils import (  # noqa: E402
    Scales, add_noise_snr, generate_gain_switched_data,
    LaserPINN, LearnableParams,  # noqa: F811
    physics_residuals, compute_loss, make_collocation, subsample,
    DEVICE as device, DTYPE,
)
from gsdfb.plotting import setup_plotting, save_fig, img_dir  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# Output directory
OUT_DIR = img_dir('pinn_inverse')


# ── PINN for inverse extraction (intensity only) ────────────────────────────

class InversePINN(nn.Module):
    """
    PINN that takes (tau, i_norm) and outputs (n, sigma).
    For intensity-only inversion we only need carrier and photon density.
    """
    def __init__(self, hidden_dim=96, n_layers=5):
        super().__init__()
        self.input_layer = nn.Linear(2, hidden_dim)
        self.hidden = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.output_layer = nn.Linear(hidden_dim, 2)  # n, sigma
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, tau, i_norm):
        x = torch.cat([tau, i_norm], dim=-1)
        h = torch.tanh(self.input_layer(x))
        for k, layer in enumerate(self.hidden):
            h_new = torch.tanh(layer(h))
            if k % 2 == 1:
                h_new = h_new + h
            h = h_new
        out = self.output_layer(h)
        return out[:, 0:1], out[:, 1:2]  # n, sigma


def physics_residuals_intensity(model, tau, i_norm, params, scales, learnable):
    """
    Compute carrier and photon rate equation residuals (no chirp).
    """
    tau = tau.requires_grad_(True)
    n, sigma = model(tau, i_norm)

    ones = torch.ones_like(n)
    dn_dtau = torch.autograd.grad(n, tau, ones, create_graph=True, retain_graph=True)[0]
    dsigma_dtau = torch.autograd.grad(sigma, tau, ones, create_graph=True, retain_graph=True)[0]

    def P(name):
        return learnable.get(name)

    N = scales.N_tr + n * scales.N_scale
    sigma_clamped = torch.clamp(sigma, -15.0, 5.0)
    S = scales.S_ref * (10.0 ** sigma_clamped)
    I = i_norm * scales.I_th

    a = P('a')
    N_tr = torch.tensor(params.N_tr, dtype=DTYPE, device=device)
    epsilon = P('epsilon')
    g = a * (N - N_tr) / (1.0 + epsilon * S)

    A_coeff = torch.tensor(params.A, dtype=DTYPE, device=device)
    B_coeff = torch.tensor(params.B, dtype=DTYPE, device=device)
    C_coeff = torch.tensor(params.C, dtype=DTYPE, device=device)
    R_sp = A_coeff * N + B_coeff * N**2 + C_coeff * N**3

    Gamma = torch.tensor(params.Gamma, dtype=DTYPE, device=device)
    v_g = torch.tensor(params.v_g, dtype=DTYPE, device=device)
    V = torch.tensor(params.V, dtype=DTYPE, device=device)
    tau_p = torch.tensor(params.tau_p, dtype=DTYPE, device=device)
    beta_sp = P('beta_sp')
    q_val = torch.tensor(q, dtype=DTYPE, device=device)
    t_ref = torch.tensor(scales.t_ref, dtype=DTYPE, device=device)
    ln10 = torch.tensor(math.log(10.0), dtype=DTYPE, device=device)
    N_scale_t = torch.tensor(scales.N_scale, dtype=DTYPE, device=device)

    # Carrier equation
    rhs_n = (t_ref / N_scale_t) * (I / (q_val * V) - R_sp - Gamma * v_g * g * S)

    # Photon equation (log-transformed)
    rhs_sigma = (t_ref / ln10) * (
        Gamma * v_g * g - 1.0 / tau_p + beta_sp * B_coeff * N**2 / S
    )

    return dn_dtau - rhs_n, dsigma_dtau - rhs_sigma


def train_inverse_intensity(params_true, gs_data, snr_db=25,
                            learn_which=None, perturbation=0.5,
                            epochs=2500, lr_net=5e-4, lr_params=5e-3,
                            seed=42, verbose=True):
    """
    Train PINN in inverse mode using intensity-only data.

    Returns trained model, learnable params, history, and parameter history.
    """
    if learn_which is None:
        learn_which = ['alpha_H', 'epsilon', 'beta_sp', 'a']

    scales = Scales(params_true)
    t = gs_data['t']
    S_clean = gs_data['S']
    N_clean = gs_data['N']

    # Add noise to intensity
    S_noisy = add_noise_snr(S_clean, snr_db, seed=seed)

    # Prepare normalized data tensors
    tau_np = t / scales.t_ref
    sigma_noisy_np = np.log10(np.maximum(S_noisy, 1e-30) / scales.S_ref)
    n_np = (N_clean - scales.N_tr) / scales.N_scale  # carrier not observed, but for reference

    # Current waveform at each time point
    I_arr = np.array([gs_data['I_func'](ti) for ti in t])
    i_norm_np = I_arr / scales.I_th

    def to_t(arr):
        return torch.tensor(arr, dtype=DTYPE, device=device).reshape(-1, 1)

    tau_data = to_t(tau_np)
    sigma_data = to_t(sigma_noisy_np)
    i_norm_data = to_t(i_norm_np)

    # Subsample data for training efficiency
    n_data_pts = min(800, len(tau_np))
    idx_data = np.linspace(0, len(tau_np) - 1, n_data_pts).astype(int)
    tau_d = tau_data[idx_data]
    sigma_d = sigma_data[idx_data]
    i_norm_d = i_norm_data[idx_data]

    # Model and learnable params
    model = InversePINN(hidden_dim=96, n_layers=5).to(device).to(DTYPE)

    # Perturb initial guesses
    rng = np.random.default_rng(seed + 100)
    perturbed_vals = {}
    for name in learn_which:
        true_val = getattr(params_true, name)
        # Random perturbation between -perturbation and +perturbation (relative)
        factor = 1.0 + perturbation * (2.0 * rng.random() - 1.0)
        factor = max(factor, 0.3)  # don't go below 30% of true value
        perturbed_vals[name] = true_val * factor
    params_init = replace(params_true, **perturbed_vals)
    learnable = LearnableParams(params_init, learn_which).to(device)

    # Optimizers
    opt_net = torch.optim.Adam(model.parameters(), lr=lr_net)
    opt_params = torch.optim.Adam(learnable.parameters(), lr=lr_params)
    sched_net = torch.optim.lr_scheduler.CosineAnnealingLR(opt_net, T_max=epochs, eta_min=lr_net * 0.01)
    sched_params = torch.optim.lr_scheduler.CosineAnnealingLR(opt_params, T_max=epochs, eta_min=lr_params * 0.01)

    tau_max = tau_np[-1]
    n_colloc = 1500

    history = []
    param_history = {name: [] for name in learn_which}
    t_start = time.time()

    for epoch in range(epochs):
        # Physics weight ramps up
        frac = min(epoch / (epochs * 0.5), 1.0)
        lam_phys = 0.1 + 0.9 * frac

        # Collocation points
        tau_c = torch.rand(n_colloc, 1, dtype=DTYPE, device=device) * tau_max
        tau_c = torch.sort(tau_c, dim=0)[0]
        # Interpolate current at collocation points
        tau_c_np = tau_c.detach().numpy().flatten()
        i_c_np = np.interp(tau_c_np * scales.t_ref, t, I_arr) / scales.I_th
        i_c = torch.tensor(i_c_np, dtype=DTYPE, device=device).reshape(-1, 1)

        opt_net.zero_grad()
        opt_params.zero_grad()

        # Physics loss
        res_n, res_s = physics_residuals_intensity(model, tau_c, i_c, params_true, scales, learnable)
        L_phys = torch.mean(res_n**2 + res_s**2)

        # Data loss (intensity only)
        _, sigma_pred = model(tau_d, i_norm_d)
        L_data = torch.mean((sigma_pred - sigma_d)**2)

        loss = lam_phys * L_phys + 10.0 * L_data
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(learnable.parameters(), max_norm=2.0)

        opt_net.step()
        opt_params.step()
        sched_net.step()
        sched_params.step()

        info = {'total': loss.item(), 'physics': L_phys.item(), 'data': L_data.item()}
        history.append(info)
        for name in learn_which:
            param_history[name].append(learnable.get(name).item())

        if verbose and epoch % 500 == 0:
            elapsed = time.time() - t_start
            param_str = ', '.join(f'{n}={learnable.get(n).item():.3e}' for n in learn_which[:2])
            print(f"    epoch {epoch:5d}/{epochs} | loss {loss.item():.3e} | "
                  f"phys {L_phys.item():.3e} | data {L_data.item():.3e} | "
                  f"{param_str} | {elapsed:.1f}s")

    elapsed = time.time() - t_start
    if verbose:
        print(f"  Training complete in {elapsed:.1f}s")

    return model, learnable, scales, history, param_history, {
        'tau_data': tau_data, 'sigma_data': sigma_data, 'i_norm_data': i_norm_data,
        'S_noisy': S_noisy, 'S_clean': S_clean, 'N_clean': N_clean, 't': t,
    }


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_parameter_convergence(param_history, params_true, learn_which, title=None):
    """Plot learned parameters vs epoch with true values as dashed lines."""
    n_params = len(learn_which)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for i, name in enumerate(learn_which):
        ax = axes[i]
        true_val = getattr(params_true, name)
        vals = np.array(param_history[name])
        epochs_arr = np.arange(len(vals))

        ax.plot(epochs_arr, vals / true_val, 'b-', lw=1.2)
        ax.axhline(1.0, color='r', ls='--', lw=1.5, label='True value')
        ax.axhline(vals[0] / true_val, color='gray', ls=':', lw=1,
                   label=f'Initial ({vals[0]/true_val:.2f}x)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(f'{name} / true')
        ax.set_title(f'{name}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(2.0, vals[0] / true_val * 1.2))

    # Hide unused axes
    for i in range(n_params, 4):
        axes[i].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=14)
    else:
        fig.suptitle('Inverse PINN: Parameter Convergence (SNR=25 dB)', fontsize=14)
    plt.tight_layout()
    return fig


def plot_fit_quality(model, data_dict, scales, params_true, gs_data):
    """Plot PINN fit vs noisy data vs ground truth."""
    model.eval()
    t = data_dict['t']
    t_ns = t * 1e9

    # Subsample for plotting
    n_plot = min(2000, len(t))
    idx_plot = np.linspace(0, len(t) - 1, n_plot).astype(int)

    tau_plot = data_dict['tau_data'][idx_plot]
    i_norm_plot = data_dict['i_norm_data'][idx_plot]

    with torch.no_grad():
        _, sigma_pred = model(tau_plot, i_norm_plot)

    S_pred = (scales.S_ref * 10.0**(sigma_pred.numpy().flatten()))
    P_pred = params_true.output_power(np.maximum(S_pred, 0))
    P_clean = params_true.output_power(data_dict['S_clean'][idx_plot])
    P_noisy = params_true.output_power(np.maximum(data_dict['S_noisy'][idx_plot], 0))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Inverse PINN: Fit Quality (Intensity-Only Data)', fontsize=14)

    # Full trace
    axes[0].plot(t_ns[idx_plot], P_clean * 1e3, 'k-', lw=1, alpha=0.7, label='Ground truth')
    axes[0].plot(t_ns[idx_plot], P_noisy * 1e3, '.', color='gray', ms=1, alpha=0.3, label='Noisy data')
    axes[0].plot(t_ns[idx_plot], P_pred * 1e3, 'r-', lw=1.2, alpha=0.8, label='PINN prediction')
    axes[0].set_ylabel('Output power (mW)')
    axes[0].set_xlabel('Time (ns)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Full pulse train')

    # Zoomed view (2 periods)
    T_rep = gs_data['T_rep']
    t_zoom_start = 3 * T_rep
    t_zoom_end = 5 * T_rep
    mask_zoom = (t[idx_plot] >= t_zoom_start) & (t[idx_plot] <= t_zoom_end)

    axes[1].plot(t_ns[idx_plot][mask_zoom], P_clean[mask_zoom] * 1e3,
                 'k-', lw=1.5, label='Ground truth')
    axes[1].plot(t_ns[idx_plot][mask_zoom], P_noisy[mask_zoom] * 1e3,
                 'o', color='gray', ms=3, alpha=0.4, label='Noisy data')
    axes[1].plot(t_ns[idx_plot][mask_zoom], P_pred[mask_zoom] * 1e3,
                 'r-', lw=2, alpha=0.9, label='PINN prediction')
    axes[1].set_ylabel('Output power (mW)')
    axes[1].set_xlabel('Time (ns)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Zoomed: 2 pulse periods')

    plt.tight_layout()
    return fig


def plot_noise_robustness(results_by_snr, params_true, learn_which):
    """Plot parameter accuracy vs SNR."""
    snrs = sorted(results_by_snr.keys())
    n_params = len(learn_which)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: relative error vs SNR for each param
    for name in learn_which:
        true_val = getattr(params_true, name)
        errors = []
        for snr in snrs:
            final_val = results_by_snr[snr]['final_params'][name]
            err = abs(final_val - true_val) / true_val * 100
            errors.append(err)
        axes[0].plot(snrs, errors, 'o-', lw=1.5, ms=8, label=name)

    axes[0].set_xlabel('SNR (dB)')
    axes[0].set_ylabel('Relative error (%)')
    axes[0].set_title('Parameter Recovery Accuracy vs SNR')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    # Right: bar chart at each SNR
    x = np.arange(n_params)
    width = 0.25
    for i, snr in enumerate(snrs):
        errors = []
        for name in learn_which:
            true_val = getattr(params_true, name)
            final_val = results_by_snr[snr]['final_params'][name]
            errors.append(abs(final_val - true_val) / true_val * 100)
        offset = (i - 1) * width
        axes[1].bar(x + offset, errors, width, label=f'{snr} dB')

    axes[1].set_xlabel('Parameter')
    axes[1].set_ylabel('Relative error (%)')
    axes[1].set_title('Extraction Error by Parameter and SNR')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(learn_which, fontsize=9)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    fig.suptitle('Noise Robustness: Inverse PINN Parameter Extraction', fontsize=13)
    plt.tight_layout()
    return fig


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    setup_plotting()

    print("=" * 70)
    print("  PINN Inverse Parameter Extraction")
    print("  Intensity-only data, gain-switched 5 GHz pulse train")
    print("=" * 70)

    params_true = DFBLaserParams()
    I_th = params_true.threshold_current()
    learn_which = ['alpha_H', 'epsilon', 'beta_sp', 'a']

    print(f"\n  Laser: 1550 nm InGaAsP/InP DFB, I_th = {I_th*1e3:.1f} mA")
    print(f"  Parameters to learn: {learn_which}")
    print(f"  True values:")
    for name in learn_which:
        print(f"    {name:>10} = {getattr(params_true, name):.3e}")

    # ── Step 1: Generate ground-truth gain-switched pulse train ──────────
    print("\n  Generating 5 GHz gain-switched pulse train...")
    gs_data = generate_gain_switched_data(params_true, f_rep=5e9, n_periods=10,
                                           pts_per_period=400)
    print(f"    {len(gs_data['t'])} points, t_end = {gs_data['t'][-1]*1e9:.1f} ns")

    # ── Step 2: Train at SNR = 25 dB (primary result) ───────────────────
    print("\n" + "-" * 70)
    print("  Training inverse PINN (SNR = 25 dB)...")
    print("-" * 70)

    model_25, learnable_25, scales_25, hist_25, phist_25, data_25 = \
        train_inverse_intensity(
            params_true, gs_data, snr_db=25,
            learn_which=learn_which, perturbation=0.5,
            epochs=2500, lr_net=5e-4, lr_params=5e-3, seed=42,
        )

    # Plot 1: Parameter convergence
    fig1 = plot_parameter_convergence(phist_25, params_true, learn_which)
    save_fig(fig1, os.path.join(OUT_DIR, 'parameter_convergence.png'))

    # Plot 2: Fit quality
    fig2 = plot_fit_quality(model_25, data_25, scales_25, params_true, gs_data)
    save_fig(fig2, os.path.join(OUT_DIR, 'fit_quality.png'))

    # ── Step 3: Noise robustness study ──────────────────────────────────
    print("\n" + "-" * 70)
    print("  Noise robustness study (SNR = 20, 25, 30 dB)...")
    print("-" * 70)

    snr_levels = [20, 25, 30]
    results_by_snr = {}

    for snr in snr_levels:
        print(f"\n  SNR = {snr} dB:")
        model_snr, learnable_snr, _, _, phist_snr, _ = \
            train_inverse_intensity(
                params_true, gs_data, snr_db=snr,
                learn_which=learn_which, perturbation=0.5,
                epochs=2500, lr_net=5e-4, lr_params=5e-3,
                seed=42 + snr,
            )
        final_params = {name: learnable_snr.get(name).item() for name in learn_which}
        results_by_snr[snr] = {
            'final_params': final_params,
            'param_history': phist_snr,
        }

    # Plot 3: Noise robustness
    fig3 = plot_noise_robustness(results_by_snr, params_true, learn_which)
    save_fig(fig3, os.path.join(OUT_DIR, 'noise_robustness.png'))

    # ── Summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY: Parameter Extraction Results")
    print("=" * 70)
    print(f"\n  {'Param':<10} {'True':>12} ", end="")
    for snr in snr_levels:
        print(f"{'SNR='+str(snr)+' dB':>14}", end="")
    print()
    print(f"  {'─'*10} {'─'*12} ", end="")
    for _ in snr_levels:
        print(f"{'─'*14}", end="")
    print()

    for name in learn_which:
        true_val = getattr(params_true, name)
        print(f"  {name:<10} {true_val:>12.3e} ", end="")
        for snr in snr_levels:
            final_val = results_by_snr[snr]['final_params'][name]
            err = abs(final_val - true_val) / true_val * 100
            print(f"  {final_val:.3e}({err:4.1f}%)", end="")
        print()

    print(f"\n  All figures saved to: images/pinn_inverse/")
    print("=" * 70)
