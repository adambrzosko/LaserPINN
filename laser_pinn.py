"""
Physics-Informed Neural Network (PINN) for Semiconductor Laser Rate Equations.

Uses the laser rate equations (carrier density, photon density, chirp) as the
loss function to train a neural network that can:
  1. Forward mode  — predict laser dynamics as a fast surrogate, trained with
                     sparse ODE data + physics (rate equation) regularization
  2. Inverse mode  — extract unknown laser parameters from measured data

Requires: torch, numpy, scipy, matplotlib
"""

import math
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from dfb_laser import DFBLaserParams, solve_transient, q, h, c


# ── Device selection ──────────────────────────────────────────────────────────

device = torch.device('cpu')  # float64 autograd is most reliable on CPU
DTYPE = torch.float64


# ── Normalization scales ──────────────────────────────────────────────────────

class Scales:
    """Normalization constants derived from laser parameters."""

    def __init__(self, params: DFBLaserParams):
        self.t_ref = 1e-9                                       # 1 ns
        self.N_tr = params.N_tr
        g_th = (params.alpha_i + params.alpha_m) / params.Gamma
        N_th = params.N_tr + g_th / params.a
        self.N_scale = N_th - params.N_tr                       # ~5.3e23
        self.N_th = N_th

        self.S_ref = 1e21                                        # characteristic S
        self.omega_scale = 2 * math.pi * 10e9                   # ~10 GHz
        self.I_th = q * params.V * N_th / params.carrier_lifetime(N_th)


# ── Neural network ────────────────────────────────────────────────────────────

class LaserPINN(nn.Module):
    """
    PINN for laser rate equations.

    Inputs:  tau (normalized time), i_norm (normalized current)
    Outputs: n (norm. carrier density), sigma (log10 photon density),
             omega_norm (normalized chirp rate)
    """

    def __init__(self, hidden_dim=128, n_layers=6):
        super().__init__()
        self.input_layer = nn.Linear(2, hidden_dim)
        self.hidden = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.output_layer = nn.Linear(hidden_dim, 3)
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
                h_new = h_new + h          # skip connection every 2 layers
            h = h_new
        out = self.output_layer(h)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]


# ── Learnable physical parameters (inverse mode) ─────────────────────────────

class LearnableParams(nn.Module):
    """Wraps selected DFBLaserParams fields as optimizable log-parameters."""

    def __init__(self, base_params: DFBLaserParams, learn_which: list[str]):
        super().__init__()
        self.base = base_params
        self.learn_which = learn_which
        for name in learn_which:
            val = getattr(base_params, name)
            self.register_parameter(
                f'log_{name}',
                nn.Parameter(torch.tensor(math.log(val), dtype=DTYPE))
            )

    def get(self, name):
        if name in self.learn_which:
            return torch.exp(getattr(self, f'log_{name}'))
        return torch.tensor(getattr(self.base, name), dtype=DTYPE, device=device)

    def current_values(self):
        return {n: self.get(n).item() for n in self.learn_which}


# ── Physics loss ──────────────────────────────────────────────────────────────

def physics_residuals(model, tau, i_norm, params, scales, learnable=None):
    """
    Compute rate-equation residuals via autograd.

    Returns (res_n, res_sigma, res_omega) each shape (B, 1).
    """
    tau = tau.requires_grad_(True)
    n, sigma, omega_norm = model(tau, i_norm)

    # Temporal derivatives via autograd
    ones = torch.ones_like(n)
    dn_dtau = torch.autograd.grad(n, tau, ones, create_graph=True, retain_graph=True)[0]
    dsigma_dtau = torch.autograd.grad(sigma, tau, ones, create_graph=True, retain_graph=True)[0]

    # Helper to fetch param (learnable or fixed)
    def P(name):
        if learnable is not None:
            return learnable.get(name)
        return torch.tensor(getattr(params, name), dtype=DTYPE, device=device)

    # Recover physical quantities
    N = scales.N_tr + n * scales.N_scale
    sigma_clamped = torch.clamp(sigma, -15.0, 5.0)
    S = scales.S_ref * (10.0 ** sigma_clamped)
    I = i_norm * scales.I_th

    # Material gain with compression
    a = P('a')
    N_tr = P('N_tr')
    epsilon = P('epsilon')
    g = a * (N - N_tr) / (1.0 + epsilon * S)

    # Recombination
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

    # ── Carrier equation (normalized)
    rhs_n = (t_ref / N_scale_t) * (I / (q_val * V) - R_sp - Gamma * v_g * g * S)

    # ── Photon equation (log-transformed)
    rhs_sigma = (t_ref / ln10) * (
        Gamma * v_g * g - 1.0 / tau_p + beta_sp * B_coeff * N**2 / S
    )

    # ── Chirp equation (algebraic — no derivative needed)
    rhs_omega = (t_ref / omega_scale_t) * 0.5 * alpha_H * (
        Gamma * v_g * a * (N - N_tr) - 1.0 / tau_p
    )

    return dn_dtau - rhs_n, dsigma_dtau - rhs_sigma, omega_norm - rhs_omega


def compute_loss(model, tau_colloc, i_norm_colloc, params, scales,
                 tau_data=None, i_norm_data=None,
                 n_data=None, sigma_data=None, omega_data=None,
                 learnable=None,
                 lam_phys=1.0, lam_data=1.0, causal_eps=1.0):
    """
    Total PINN loss = physics residual + data fitting.
    """
    # ── Physics residuals on collocation points ──
    res_n, res_sigma, res_omega = physics_residuals(
        model, tau_colloc, i_norm_colloc, params, scales, learnable
    )

    # Causal weighting
    r2 = (res_n.detach()**2 + res_sigma.detach()**2 + res_omega.detach()**2)
    r2_clamped = torch.clamp(r2, max=50.0)
    w_causal = torch.exp(-causal_eps * torch.cumsum(r2_clamped, dim=0))
    w_causal = w_causal / (w_causal.mean() + 1e-30)

    L_phys = torch.mean(w_causal * (res_n**2 + res_sigma**2 + res_omega**2))

    # ── Data loss ──
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


# ── Reference data generation ────────────────────────────────────────────────

def generate_reference(params, scales, I_bias, t_end=10e-9, n_points=2000):
    """Run ODE solver and return normalized arrays + torch tensors."""
    sol = solve_transient(params, lambda t: I_bias, [0, t_end],
                          t_eval=np.linspace(0, t_end, n_points))
    N, S, phi = sol.y

    tau_np = sol.t / scales.t_ref
    n_np = (N - scales.N_tr) / scales.N_scale
    sigma_np = np.log10(np.maximum(S, 1e-30) / scales.S_ref)
    chirp_np = np.gradient(phi, sol.t)
    omega_np = chirp_np * scales.t_ref / scales.omega_scale
    i_norm_np = np.full_like(tau_np, I_bias / scales.I_th)

    def to_t(arr):
        return torch.tensor(arr, dtype=DTYPE, device=device).reshape(-1, 1)

    return {
        'tau': to_t(tau_np), 'n': to_t(n_np), 'sigma': to_t(sigma_np),
        'omega': to_t(omega_np), 'i_norm': to_t(i_norm_np),
        'N': N, 'S': S, 'phi': phi, 't': sol.t,
    }


# ── Training helpers ──────────────────────────────────────────────────────────

def make_collocation(tau_max, n_points, dense_frac=0.5):
    """Collocation points with denser sampling near tau=0."""
    n_dense = int(n_points * dense_frac)
    n_sparse = n_points - n_dense
    tau_dense = torch.rand(n_dense, 1, dtype=DTYPE, device=device) * (tau_max * 0.2)
    tau_sparse = torch.rand(n_sparse, 1, dtype=DTYPE, device=device) * tau_max
    return torch.sort(torch.cat([tau_dense, tau_sparse], dim=0), dim=0)[0]


def subsample(ref, n_pts):
    """Take n_pts evenly spaced points from reference data."""
    N_total = ref['tau'].shape[0]
    idx = torch.linspace(0, N_total - 1, n_pts).long()
    return {k: ref[k][idx] for k in ('tau', 'n', 'sigma', 'omega', 'i_norm')}


# ── Training: forward mode ────────────────────────────────────────────────────

def train_forward(params, I_bias, t_end=10e-9,
                  epochs_adam=5000, epochs_lbfgs=300,
                  n_colloc=2000, n_data_pts=200,
                  lr_adam=5e-4, lr_lbfgs=0.5,
                  verbose=True):
    """
    Train PINN in forward mode.

    Uses sparse ODE reference data to anchor the solution, with rate-equation
    physics loss as a regularizer that enforces physical consistency between
    data points.  The physics loss weight increases over training so the
    network learns to extrapolate beyond the data.
    """
    scales = Scales(params)
    model = LaserPINN().to(device).to(DTYPE)

    # Full ODE reference (for validation) and sparse subset (for training)
    ref = generate_reference(params, scales, I_bias, t_end, n_points=2000)
    data = subsample(ref, n_data_pts)

    tau_max = t_end / scales.t_ref
    history = []
    t_start = time.time()

    # ── Phase 1: Adam with increasing physics weight ──
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_adam)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs_adam, eta_min=lr_adam * 0.01
    )

    for epoch in range(epochs_adam):
        # Physics weight ramps from 0.01 to 1.0 over training
        frac = min(epoch / (epochs_adam * 0.7), 1.0)
        lam_phys = 0.01 + 0.99 * frac

        tau_colloc = make_collocation(tau_max, n_colloc)
        i_norm_colloc = torch.full_like(tau_colloc, I_bias / scales.I_th)

        optimizer.zero_grad()
        loss, info = compute_loss(
            model, tau_colloc, i_norm_colloc, params, scales,
            tau_data=data['tau'], i_norm_data=data['i_norm'],
            n_data=data['n'], sigma_data=data['sigma'], omega_data=data['omega'],
            lam_phys=lam_phys, lam_data=10.0, causal_eps=1.0,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        history.append(info)
        if verbose and epoch % 1000 == 0:
            elapsed = time.time() - t_start
            print(f"  Adam  {epoch:5d}/{epochs_adam} | loss {info['total']:.3e} | "
                  f"phys {info['physics']:.3e} | data {info['data']:.3e} | "
                  f"lam_p={lam_phys:.2f} | {elapsed:.1f}s")

    # ── Phase 2: L-BFGS refinement ──
    if verbose:
        print("  Switching to L-BFGS...")

    lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=lr_lbfgs,
        max_iter=20, history_size=50, line_search_fn='strong_wolfe'
    )
    # Fixed collocation for L-BFGS (it needs deterministic loss)
    tau_colloc_fixed = make_collocation(tau_max, n_colloc)
    i_colloc_fixed = torch.full_like(tau_colloc_fixed, I_bias / scales.I_th)

    for step in range(epochs_lbfgs):
        def closure():
            lbfgs.zero_grad()
            loss, _ = compute_loss(
                model, tau_colloc_fixed, i_colloc_fixed, params, scales,
                tau_data=data['tau'], i_norm_data=data['i_norm'],
                n_data=data['n'], sigma_data=data['sigma'], omega_data=data['omega'],
                lam_phys=1.0, lam_data=10.0, causal_eps=1.0,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            return loss

        lbfgs.step(closure)
        # Evaluate loss for logging (needs grad for autograd in physics_residuals)
        _, info = compute_loss(
            model, tau_colloc_fixed, i_colloc_fixed, params, scales,
            tau_data=data['tau'], i_norm_data=data['i_norm'],
            n_data=data['n'], sigma_data=data['sigma'], omega_data=data['omega'],
            lam_phys=1.0, lam_data=10.0, causal_eps=1.0,
        )
        history.append(info)
        if verbose and step % 100 == 0:
            print(f"  L-BFGS {step:3d}/{epochs_lbfgs} | loss {info['total']:.3e} | "
                  f"phys {info['physics']:.3e} | data {info['data']:.3e}")

    elapsed = time.time() - t_start
    if verbose:
        print(f"  Training complete in {elapsed:.1f}s")

    return model, scales, history, ref


# ── Training: inverse mode ────────────────────────────────────────────────────

def data_residuals(N, S, phi, t, I_val, params, scales, learnable):
    """
    Compute rate-equation residuals directly on data arrays using finite
    differences for time derivatives.  This bypasses the neural network
    entirely — gradients flow only through the physical parameters.
    """
    dt = t[1:] - t[:-1]  # (M-1,)

    # Midpoint values
    N_mid = 0.5 * (N[1:] + N[:-1])
    S_mid = 0.5 * (S[1:] + S[:-1])

    def P(name):
        return learnable.get(name)

    a = P('a')
    N_tr = P('N_tr')
    eps = P('epsilon')
    A_c = P('A')
    B_c = P('B')
    C_c = P('C')
    Gamma = P('Gamma')
    v_g = torch.tensor(params.v_g, dtype=DTYPE, device=device)
    V = torch.tensor(params.V, dtype=DTYPE, device=device)
    tau_p = torch.tensor(params.tau_p, dtype=DTYPE, device=device)
    beta_sp = P('beta_sp')
    alpha_H = P('alpha_H')
    q_val = torch.tensor(q, dtype=DTYPE, device=device)

    g = a * (N_mid - N_tr) / (1.0 + eps * S_mid)
    R_sp = A_c * N_mid + B_c * N_mid**2 + C_c * N_mid**3

    # dN/dt from data (finite difference)
    dNdt_data = (N[1:] - N[:-1]) / dt
    dNdt_model = I_val / (q_val * V) - R_sp - Gamma * v_g * g * S_mid

    # dS/dt from data
    dSdt_data = (S[1:] - S[:-1]) / dt
    dSdt_model = (Gamma * v_g * g - 1.0 / tau_p) * S_mid + beta_sp * B_c * N_mid**2

    # Normalize residuals by characteristic scales
    N_scale = torch.tensor(scales.N_scale, dtype=DTYPE, device=device)
    t_ref = torch.tensor(scales.t_ref, dtype=DTYPE, device=device)

    res_N = (dNdt_data - dNdt_model) * t_ref / N_scale
    res_S = (dSdt_data - dSdt_model) / (S_mid.abs() + 1e-10) * t_ref

    return res_N, res_S


def train_inverse(params_true, I_bias, learn_which, t_end=10e-9,
                  perturbation=0.3, noise_frac=0.01,
                  epochs=5000, lr_params=1e-2, verbose=True):
    """
    Recover unknown laser parameters from (noisy) measurement data.

    Computes rate-equation residuals directly on the data using finite
    differences — no neural network needed for the inverse problem.  This
    ensures gradient signals drive the parameters toward values that make the
    data consistent with the rate equations.

    Uses multiple bias currents for parameter identifiability.
    """
    scales = Scales(params_true)

    # Multiple bias currents for identifiability
    I_th = scales.I_th
    I_biases = [2.0 * I_th, 3.0 * I_th, 5.0 * I_th]
    if verbose:
        print(f"  Using {len(I_biases)} bias currents: "
              + ", ".join(f"{I*1e3:.1f} mA" for I in I_biases))

    # Generate reference data (simulating experimental measurements)
    refs = [generate_reference(params_true, scales, Ib, t_end, n_points=2000)
            for Ib in I_biases]

    # Convert to torch tensors with noise
    torch.manual_seed(42)
    data_per_current = []
    for r, Ib in zip(refs, I_biases):
        N_t = torch.tensor(r['N'], dtype=DTYPE, device=device)
        S_t = torch.tensor(r['S'], dtype=DTYPE, device=device)
        phi_t = torch.tensor(r['phi'], dtype=DTYPE, device=device)
        t_t = torch.tensor(r['t'], dtype=DTYPE, device=device)
        # Add noise
        N_t = N_t + noise_frac * N_t.abs().mean() * torch.randn_like(N_t)
        S_t = S_t * (1.0 + noise_frac * torch.randn_like(S_t))
        S_t = torch.clamp(S_t, min=1.0)
        data_per_current.append((N_t, S_t, phi_t, t_t, Ib))

    ref = refs[1]  # 3x I_th for display

    # Perturbed initial guesses
    from dataclasses import replace
    perturbed_vals = {}
    for name in learn_which:
        true_val = getattr(params_true, name)
        perturbed_vals[name] = true_val * (1.0 + perturbation)
    params_guess = replace(params_true, **perturbed_vals)
    learnable = LearnableParams(params_guess, learn_which).to(device)

    opt = torch.optim.Adam(learnable.parameters(), lr=lr_params)
    def lr_lambda(epoch):
        warmup = int(epochs * 0.1)
        if epoch < warmup:
            return epoch / max(warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup) / (epochs - warmup)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    history = []
    param_history = {name: [] for name in learn_which}
    t_start = time.time()

    for epoch in range(epochs):
        opt.zero_grad()

        total_loss = torch.tensor(0.0, dtype=DTYPE, device=device)
        for N_t, S_t, phi_t, t_t, Ib in data_per_current:
            res_N, res_S = data_residuals(N_t, S_t, phi_t, t_t, Ib,
                                          params_guess, scales, learnable)
            total_loss = total_loss + torch.mean(res_N**2) + torch.mean(res_S**2)

        total_loss = total_loss / len(data_per_current)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(learnable.parameters(), max_norm=5.0)
        opt.step()
        sched.step()

        info = {'total': total_loss.item(), 'physics': total_loss.item(), 'data': 0.0}
        history.append(info)
        for name in learn_which:
            param_history[name].append(learnable.get(name).item())

        if verbose and epoch % 500 == 0:
            elapsed = time.time() - t_start
            param_str = ', '.join(
                f'{n}={learnable.get(n).item():.3e}' for n in learn_which
            )
            print(f"    epoch {epoch:5d}/{epochs} | "
                  f"residual {total_loss.item():.3e} | {param_str} | {elapsed:.1f}s")

    elapsed = time.time() - t_start
    if verbose:
        print(f"  Inverse training complete in {elapsed:.1f}s")
        print("\n  Parameter recovery:")
        print(f"  {'Param':<12} {'True':>12} {'Initial':>12} {'Recovered':>12} {'Error':>8}")
        print(f"  {'─'*60}")
        for name in learn_which:
            true_val = getattr(params_true, name)
            init_val = getattr(params_guess, name)
            rec_val = learnable.get(name).item()
            err = abs(rec_val - true_val) / true_val * 100
            print(f"  {name:<12} {true_val:>12.3e} {init_val:>12.3e} {rec_val:>12.3e} {err:>7.1f}%")

    # Build a dummy model for the plotting function (train quickly on ref data)
    model = LaserPINN().to(device).to(DTYPE)
    opt_m = torch.optim.Adam(model.parameters(), lr=5e-4)
    ref_t = generate_reference(params_true, scales, I_bias, t_end, n_points=500)
    for _ in range(1000):
        opt_m.zero_grad()
        n_p, s_p, o_p = model(ref_t['tau'], ref_t['i_norm'])
        l = torch.mean((n_p - ref_t['n'])**2 + (s_p - ref_t['sigma'])**2 + (o_p - ref_t['omega'])**2)
        l.backward()
        opt_m.step()

    return model, learnable, scales, history, param_history, ref


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_forward_results(model, ref, scales, params, title='PINN vs ODE Solver'):
    """Compare PINN predictions vs ODE solver."""
    model.eval()
    with torch.no_grad():
        n_pred, sigma_pred, omega_pred = model(ref['tau'], ref['i_norm'])

    N_pred = (scales.N_tr + n_pred.numpy() * scales.N_scale).flatten()
    S_pred = (scales.S_ref * 10.0**(sigma_pred.numpy())).flatten()
    P_pred = params.output_power(np.maximum(S_pred, 0))
    P_ref = params.output_power(ref['S'])

    t_ns = ref['t'] * 1e9

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(title, fontsize=14)

    axes[0, 0].plot(t_ns, ref['N'] * 1e-24, 'k-', lw=2, label='ODE')
    axes[0, 0].plot(t_ns, N_pred * 1e-24, 'r--', lw=1.5, label='PINN')
    axes[0, 0].set_ylabel('Carrier density (10$^{24}$ m$^{-3}$)')
    axes[0, 0].set_xlabel('Time (ns)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].semilogy(t_ns, ref['S'], 'k-', lw=2, label='ODE')
    axes[0, 1].semilogy(t_ns, np.maximum(S_pred, 1e-10), 'r--', lw=1.5, label='PINN')
    axes[0, 1].set_ylabel('Photon density (m$^{-3}$)')
    axes[0, 1].set_xlabel('Time (ns)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(t_ns, P_ref * 1e3, 'k-', lw=2, label='ODE')
    axes[1, 0].plot(t_ns, P_pred * 1e3, 'r--', lw=1.5, label='PINN')
    axes[1, 0].set_ylabel('Output power (mW)')
    axes[1, 0].set_xlabel('Time (ns)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Relative error
    mask = ref['S'] > ref['S'].max() * 1e-6  # meaningful photon densities
    rel_err_N = np.abs(N_pred - ref['N']) / np.abs(ref['N']) * 100
    rel_err_S = np.full_like(ref['S'], np.nan)
    rel_err_S[mask] = np.abs(S_pred[mask] - ref['S'][mask]) / ref['S'][mask] * 100

    axes[1, 1].semilogy(t_ns, rel_err_N, 'b-', lw=1, label='N error %')
    axes[1, 1].semilogy(t_ns[mask], rel_err_S[mask], 'r-', lw=1, label='S error %')
    axes[1, 1].set_ylabel('Relative error (%)')
    axes[1, 1].set_xlabel('Time (ns)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(1e-4, 200)

    plt.tight_layout()
    return fig


def plot_loss_history(history, title='Training Convergence'):
    """Training loss convergence."""
    epochs = np.arange(len(history))
    total = np.array([h['total'] for h in history])
    phys = np.array([h['physics'] for h in history])
    data = np.array([h['data'] for h in history])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(epochs, total, 'k-', lw=1.5, label='Total', alpha=0.8)
    ax.semilogy(epochs, phys, 'b-', lw=1, label='Physics', alpha=0.7)
    if data.max() > 0:
        ax.semilogy(epochs, np.maximum(data, 1e-30), 'g-', lw=1, label='Data', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_param_convergence(param_history, params_true, learn_which):
    """Parameter recovery convergence for inverse mode."""
    n_params = len(learn_which)
    fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 4))
    if n_params == 1:
        axes = [axes]

    for ax, name in zip(axes, learn_which):
        true_val = getattr(params_true, name)
        vals = np.array(param_history[name])
        ax.plot(vals / true_val, 'b-', lw=1)
        ax.axhline(1.0, color='k', ls='--', lw=1, label='True value')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(f'{name} / true')
        ax.set_title(f'{name} convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('Inverse Mode — Parameter Recovery', fontsize=13)
    plt.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    params = DFBLaserParams()
    I_th = params.threshold_current()
    I_bias = 3.0 * I_th

    print("=" * 65)
    print("  Physics-Informed Neural Network for Laser Rate Equations")
    print("=" * 65)
    print(f"  Laser:     1550 nm InGaAsP/InP DFB")
    print(f"  I_th:      {I_th*1e3:.1f} mA")
    print(f"  I_bias:    {I_bias*1e3:.1f} mA  ({I_bias/I_th:.0f}x threshold)")
    print(f"  Device:    {device}")
    print()

    # ── Demo 1: Forward mode ──────────────────────────────────────────────
    print("━" * 65)
    print("  FORWARD MODE — physics-regularized surrogate model")
    print("━" * 65)
    print("  Training with sparse ODE data + rate equation physics loss...")
    print()

    model_fwd, scales, hist_fwd, ref_fwd = train_forward(
        params, I_bias, t_end=10e-9,
        epochs_adam=5000, epochs_lbfgs=300,
        n_colloc=2000, n_data_pts=200,
    )

    fig1 = plot_forward_results(model_fwd, ref_fwd, scales, params,
                                 title='Forward Mode — PINN vs ODE Solver')
    fig1.savefig('pinn_forward_results.png', dpi=150)
    print("  Saved: pinn_forward_results.png")

    fig2 = plot_loss_history(hist_fwd, 'Forward Mode — Training Convergence')
    fig2.savefig('pinn_forward_loss.png', dpi=150)
    print("  Saved: pinn_forward_loss.png")

    # Timing comparison
    model_fwd.eval()
    with torch.no_grad():
        t0 = time.time()
        for _ in range(100):
            model_fwd(ref_fwd['tau'], ref_fwd['i_norm'])
        t_pinn = (time.time() - t0) / 100

    t0 = time.time()
    for _ in range(100):
        solve_transient(params, lambda t: I_bias, [0, 10e-9],
                        t_eval=np.linspace(0, 10e-9, 2000))
    t_ode = (time.time() - t0) / 100
    print(f"\n  Inference speed:  PINN {t_pinn*1e3:.2f} ms  vs  ODE {t_ode*1e3:.2f} ms  "
          f"({t_ode/t_pinn:.1f}x speedup)")

    # Accuracy
    model_fwd.eval()
    with torch.no_grad():
        n_p, sig_p, _ = model_fwd(ref_fwd['tau'], ref_fwd['i_norm'])
    N_p = (scales.N_tr + n_p.numpy().flatten() * scales.N_scale)
    S_p = (scales.S_ref * 10.0**(sig_p.numpy().flatten()))
    mask = ref_fwd['S'] > ref_fwd['S'].max() * 1e-3
    err_N = np.mean(np.abs(N_p - ref_fwd['N']) / ref_fwd['N']) * 100
    err_S = np.mean(np.abs(S_p[mask] - ref_fwd['S'][mask]) / ref_fwd['S'][mask]) * 100
    print(f"  Accuracy:  N mean error = {err_N:.2f}%,  S mean error = {err_S:.2f}%")

    # ── Demo 2: Inverse mode ─────────────────────────────────────────────
    print()
    print("━" * 65)
    print("  INVERSE MODE — recovering laser parameters from noisy data")
    print("━" * 65)

    learn_which = ['a', 'epsilon']
    print(f"  Learning: {learn_which}")
    print(f"  Initial perturbation: +30%,  Noise: 1%")
    print()

    model_inv, learnable, scales_inv, hist_inv, phist, ref_inv = train_inverse(
        params, I_bias, learn_which,
        t_end=10e-9, perturbation=0.3, noise_frac=0.01,
        epochs=5000, lr_params=1e-2,
    )

    fig3 = plot_forward_results(model_inv, ref_inv, scales_inv, params,
                                 title='Inverse Mode — PINN Fit to Noisy Data')
    fig3.savefig('pinn_inverse_fit.png', dpi=150)
    print("  Saved: pinn_inverse_fit.png")

    fig4 = plot_loss_history(hist_inv, 'Inverse Mode — Training Convergence')
    fig4.savefig('pinn_inverse_loss.png', dpi=150)
    print("  Saved: pinn_inverse_loss.png")

    fig5 = plot_param_convergence(phist, params, learn_which)
    fig5.savefig('pinn_inverse_params.png', dpi=150)
    print("  Saved: pinn_inverse_params.png")

    plt.close('all')
    print()
    print("=" * 65)
    print("  Done. All plots saved to working directory.")
    print("=" * 65)
