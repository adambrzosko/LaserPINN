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
from scipy.integrate import solve_ivp

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

    def __init__(self, hidden_dim=128, n_layers=6, n_out=3, normalize_angular=False):
        super().__init__()
        self.n_out = n_out
        # When True: outputs 2 and 3 (cos φ, sin φ) are projected onto the unit
        # circle inside forward(), so cos²φ + sin²φ = 1 holds exactly.
        self.normalize_angular = normalize_angular
        self.input_layer = nn.Linear(2, hidden_dim)
        self.hidden = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)]
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
        for k, layer in enumerate(self.hidden):
            h_new = torch.tanh(layer(h))
            if k % 2 == 1:
                h_new = h_new + h          # skip connection every 2 layers
            h = h_new
        out = self.output_layer(h)
        outputs = [out[:, i:i+1] for i in range(self.n_out)]
        if self.normalize_angular and self.n_out >= 4:
            # Project cos/sin to unit circle: (cos_r, sin_r) → (cos_r, sin_r)/‖·‖
            # This makes the unit-circle constraint exact (structural, not penalised).
            cos_r, sin_r = outputs[2], outputs[3]
            r = torch.sqrt(cos_r ** 2 + sin_r ** 2 + 1e-8)
            outputs[2] = cos_r / r
            outputs[3] = sin_r / r
        return tuple(outputs)


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


# ── Injection reference data ──────────────────────────────────────────────────

def _ode_injection(t, y, params, I_val, kappa, S_inj, delta_omega):
    """Lang-Kobayashi rate equations (inline, no external import needed)."""
    N, S, phi = y
    S = max(S, 1e-10)
    g = params.gain(N, S)
    R_sp = params.A * N + params.B * N**2 + params.C * N**3
    dNdt = I_val / (q * params.V) - R_sp - params.Gamma * params.v_g * g * S
    dSdt = ((params.Gamma * params.v_g * g - 1.0 / params.tau_p) * S
            + params.beta_sp * params.B * N**2
            + 2.0 * kappa * np.sqrt(S * S_inj) * np.cos(phi))
    dphidt = (0.5 * params.alpha_H * (params.Gamma * params.v_g * params.a
              * (N - params.N_tr) - 1.0 / params.tau_p)
              - delta_omega + kappa * np.sqrt(S_inj / S) * np.sin(phi))
    return [dNdt, dSdt, dphidt]


def generate_reference_injection(params, scales, I_bias, kappa, S_inj, delta_omega,
                                  t_end=10e-9, n_points=2000):
    """
    Run the injected DFB ODE (Lang-Kobayashi) and return normalized tensors.

    Uses angular phase encoding: outputs include cos_phi and sin_phi,
    which stay bounded in [-1, 1] regardless of total phase accumulation.
    """
    t_eval = np.linspace(0, t_end, n_points)
    sol = solve_ivp(
        _ode_injection, [0, t_end], [params.N_tr, 1e10, 0.0],
        args=(params, I_bias, kappa, S_inj, delta_omega),
        t_eval=t_eval, method='RK45', rtol=1e-9, atol=1e-12,
        max_step=t_end / 2000,
    )
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
        'cos_phi': to_t(np.cos(phi)), 'sin_phi': to_t(np.sin(phi)),
        'N': N, 'S': S, 'phi': phi, 't': sol.t,
    }


# ── Stochastic (Langevin) reference data ──────────────────────────────────────

def generate_reference_stochastic(params, scales, I_bias, t_end=10e-9,
                                   n_points=2000, n_realizations=5,
                                   dt_langevin=5e-13, seed=42):
    """
    Generate stochastic laser trajectories using Euler-Maruyama integration.

    Langevin noise sources added (Gardiner formulation for photon statistics):
      F_N  ~ sqrt(2 * R_sp)   per sqrt(s)  — carrier shot noise
      F_S  ~ sqrt(2 * β*B*N²) per sqrt(s)  — spontaneous photon noise
      F_φ  ~ sqrt(β*B*N² / (2*S)) per sqrt(s) — phase diffusion

    Returns a list of n_realizations dicts in the same format as
    generate_reference, so they can be used as drop-in replacements.
    """
    rng = np.random.default_rng(seed)
    n_steps = int(t_end / dt_langevin)
    dt = dt_langevin

    # Initial conditions from deterministic steady state
    ref0 = generate_reference(params, scales, I_bias, t_end=min(3e-9, t_end * 0.3),
                               n_points=300)
    N0 = ref0['N'][-1]
    S0 = max(ref0['S'][-1], 1.0)
    phi0 = ref0['phi'][-1]

    # Pre-generate noise for all realizations at once (shape: n_real × n_steps)
    dW_N = rng.standard_normal((n_realizations, n_steps)) * math.sqrt(dt)
    dW_S = rng.standard_normal((n_realizations, n_steps)) * math.sqrt(dt)
    dW_phi = rng.standard_normal((n_realizations, n_steps)) * math.sqrt(dt)

    # State arrays (vectorized across realizations)
    N_arr = np.zeros((n_realizations, n_steps + 1))
    S_arr = np.zeros((n_realizations, n_steps + 1))
    phi_arr = np.zeros((n_realizations, n_steps + 1))
    N_arr[:, 0] = N0
    S_arr[:, 0] = S0
    phi_arr[:, 0] = phi0

    I_val = I_bias  # constant CW current

    for k in range(n_steps):
        Nk = N_arr[:, k]
        Sk = np.maximum(S_arr[:, k], 1.0)

        gk = params.a * (Nk - params.N_tr) / (1.0 + params.epsilon * Sk)
        R_sp_k = params.A * Nk + params.B * Nk**2 + params.C * Nk**3
        R_st_k = params.Gamma * params.v_g * gk * Sk
        Bsp_N2 = params.beta_sp * params.B * np.maximum(Nk, 0.0)**2

        # Diffusion coefficients
        sig_N = np.sqrt(2.0 * np.maximum(R_sp_k, 0.0))
        sig_S = np.sqrt(2.0 * Bsp_N2)
        sig_phi = np.sqrt(Bsp_N2 / (2.0 * Sk))

        N_arr[:, k + 1] = np.maximum(
            Nk + (I_val / (q * params.V) - R_sp_k - R_st_k) * dt
            + sig_N * dW_N[:, k], 0.0)
        S_arr[:, k + 1] = np.maximum(
            Sk + ((R_st_k - Sk / params.tau_p) + Bsp_N2) * dt
            + sig_S * dW_S[:, k], 1.0)
        phi_arr[:, k + 1] = (
            phi_arr[:, k]
            + 0.5 * params.alpha_H * (params.Gamma * params.v_g * params.a
              * (Nk - params.N_tr) - 1.0 / params.tau_p) * dt
            + sig_phi * dW_phi[:, k])

    # Downsample to n_points
    idx_ds = np.linspace(0, n_steps, n_points, dtype=int)
    t_ds = idx_ds * dt

    def to_t(arr):
        return torch.tensor(arr, dtype=DTYPE, device=device).reshape(-1, 1)

    realizations = []
    for r in range(n_realizations):
        N_ds = N_arr[r, idx_ds]
        S_ds = S_arr[r, idx_ds]
        phi_ds = phi_arr[r, idx_ds]

        n_np = (N_ds - scales.N_tr) / scales.N_scale
        sigma_np = np.log10(np.maximum(S_ds, 1e-30) / scales.S_ref)
        chirp_np = np.gradient(phi_ds, t_ds) if len(t_ds) > 1 else np.zeros_like(t_ds)
        omega_np = chirp_np * scales.t_ref / scales.omega_scale
        i_norm_np = np.full_like(t_ds, I_bias / scales.I_th)

        realizations.append({
            'tau': to_t(t_ds / scales.t_ref), 'n': to_t(n_np),
            'sigma': to_t(sigma_np), 'omega': to_t(omega_np),
            'i_norm': to_t(i_norm_np),
            'N': N_ds, 'S': S_ds, 'phi': phi_ds, 't': t_ds,
        })

    return realizations


# ── Injection physics loss ────────────────────────────────────────────────────

def physics_residuals_injection(model, tau, i_norm, params, scales,
                                  kappa_val, S_inj_val, delta_omega_val,
                                  learnable=None):
    """
    Rate-equation residuals for Lang-Kobayashi injected laser.

    Model outputs 4 values: (n, sigma, cos_phi, sin_phi).
    Phase is encoded as (cos φ, sin φ) to keep network outputs bounded
    regardless of total phase accumulation.

    Residuals:
      res_n     : dn/dτ   − RHS_N
      res_sigma : dσ/dτ   − RHS_S (with injection term)
      res_cos   : d(cosφ)/dτ + sinφ · RHS_φ
      res_sin   : d(sinφ)/dτ − cosφ · RHS_φ
      circle    : cos²φ + sin²φ − 1  (unit circle soft constraint)
    """
    tau = tau.requires_grad_(True)
    n, sigma, cos_phi, sin_phi = model(tau, i_norm)

    ones = torch.ones_like(n)
    dn_dtau = torch.autograd.grad(n, tau, ones, create_graph=True, retain_graph=True)[0]
    dsigma_dtau = torch.autograd.grad(sigma, tau, ones, create_graph=True, retain_graph=True)[0]
    dcos_dtau = torch.autograd.grad(cos_phi, tau, ones, create_graph=True, retain_graph=True)[0]
    dsin_dtau = torch.autograd.grad(sin_phi, tau, ones, create_graph=True, retain_graph=True)[0]

    def P(name):
        if learnable is not None:
            return learnable.get(name)
        return torch.tensor(getattr(params, name), dtype=DTYPE, device=device)

    N = scales.N_tr + n * scales.N_scale
    sigma_clamped = torch.clamp(sigma, -15.0, 5.0)
    S = scales.S_ref * (10.0 ** sigma_clamped)
    I = i_norm * scales.I_th

    a = P('a');  N_tr = P('N_tr');  epsilon = P('epsilon')
    A_c = P('A');  B_c = P('B');  C_c = P('C')
    Gamma = P('Gamma');  beta_sp = P('beta_sp');  alpha_H = P('alpha_H')
    v_g = torch.tensor(params.v_g, dtype=DTYPE, device=device)
    V = torch.tensor(params.V, dtype=DTYPE, device=device)
    tau_p = torch.tensor(params.tau_p, dtype=DTYPE, device=device)
    q_val = torch.tensor(q, dtype=DTYPE, device=device)
    t_ref = torch.tensor(scales.t_ref, dtype=DTYPE, device=device)
    ln10 = torch.tensor(math.log(10.0), dtype=DTYPE, device=device)
    N_sc = torch.tensor(scales.N_scale, dtype=DTYPE, device=device)

    kappa = torch.tensor(kappa_val, dtype=DTYPE, device=device)
    S_inj = torch.tensor(S_inj_val, dtype=DTYPE, device=device)
    d_omega = torch.tensor(delta_omega_val, dtype=DTYPE, device=device)

    g = a * (N - N_tr) / (1.0 + epsilon * S)
    R_sp = A_c * N + B_c * N**2 + C_c * N**3
    sqrt_ratio = torch.sqrt(torch.clamp(S_inj / (S + 1e-30), min=0.0))

    # ── Carrier (unchanged)
    rhs_n = (t_ref / N_sc) * (I / (q_val * V) - R_sp - Gamma * v_g * g * S)

    # ── Photon (with injection: +2κ√(S_inj/S)·cosφ after log-transform)
    rhs_sigma = (t_ref / ln10) * (
        Gamma * v_g * g - 1.0 / tau_p
        + beta_sp * B_c * N**2 / S
        + 2.0 * kappa * sqrt_ratio * cos_phi
    )

    # ── Phase RHS: dphi/dt × t_ref
    RHS_phi = t_ref * (
        0.5 * alpha_H * (Gamma * v_g * a * (N - N_tr) - 1.0 / tau_p)
        - d_omega + kappa * sqrt_ratio * sin_phi
    )

    # Angular encoding: d(cosφ)/dτ = −sinφ · RHS_φ,  d(sinφ)/dτ = cosφ · RHS_φ
    rhs_cos = -sin_phi * RHS_phi
    rhs_sin = cos_phi * RHS_phi

    return (dn_dtau - rhs_n, dsigma_dtau - rhs_sigma,
            dcos_dtau - rhs_cos, dsin_dtau - rhs_sin)


def compute_loss_injection(model, tau_colloc, i_norm_colloc, params, scales,
                            kappa, S_inj, delta_omega,
                            tau_data=None, i_norm_data=None,
                            n_data=None, sigma_data=None,
                            cos_phi_data=None, sin_phi_data=None,
                            lam_phys=1.0, lam_data=1.0,
                            causal_eps=1.0):
    """Total loss for injected PINN (physics + data).

    The unit-circle constraint (cos²φ + sin²φ = 1) is satisfied exactly by the
    normalize_angular projection in LaserPINN.forward — no penalty term needed.
    """
    res_n, res_sig, res_cos, res_sin = physics_residuals_injection(
        model, tau_colloc, i_norm_colloc, params, scales, kappa, S_inj, delta_omega
    )

    r2 = (res_n.detach()**2 + res_sig.detach()**2
          + res_cos.detach()**2 + res_sin.detach()**2)
    r2c = torch.clamp(r2, max=50.0)
    w = torch.exp(-causal_eps * torch.cumsum(r2c, dim=0))
    w = w / (w.mean() + 1e-30)

    L_phys = torch.mean(w * (res_n**2 + res_sig**2 + res_cos**2 + res_sin**2))

    L_data = torch.tensor(0.0, dtype=DTYPE, device=device)
    if n_data is not None:
        n_p, sig_p, cos_p, sin_p = model(tau_data, i_norm_data)
        L_data = torch.mean(
            (n_p - n_data)**2 + (sig_p - sigma_data)**2
            + (cos_p - cos_phi_data)**2 + (sin_p - sin_phi_data)**2
        )

    loss = lam_phys * L_phys + lam_data * L_data
    return loss, {
        'total': loss.item(), 'physics': L_phys.item(), 'data': L_data.item(),
    }


# ── Training: injection forward mode ─────────────────────────────────────────

def train_forward_injection(params, I_bias, kappa, S_inj, delta_omega,
                              t_end=10e-9,
                              epochs_adam=5000, epochs_lbfgs=300,
                              n_colloc=2000, n_data_pts=200,
                              lr_adam=5e-4, lr_lbfgs=0.5,
                              verbose=True):
    """
    Train a 4-output PINN (n, σ, cosφ, sinφ) on injected DFB laser dynamics.

    Angular phase encoding keeps network outputs in [-1, 1] so training
    is stable even when total phase accumulates many radians.
    """
    scales = Scales(params)
    model = LaserPINN(n_out=4, normalize_angular=True).to(device).to(DTYPE)

    ref = generate_reference_injection(params, scales, I_bias, kappa, S_inj,
                                        delta_omega, t_end, n_points=2000)
    data = subsample(ref, n_data_pts)

    tau_max = t_end / scales.t_ref
    history = []
    t_start = time.time()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_adam)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs_adam, eta_min=lr_adam * 0.01
    )

    for epoch in range(epochs_adam):
        frac = min(epoch / (epochs_adam * 0.7), 1.0)
        lam_phys = 0.01 + 0.99 * frac

        tau_colloc = make_collocation(tau_max, n_colloc)
        i_norm_colloc = torch.full_like(tau_colloc, I_bias / scales.I_th)

        optimizer.zero_grad()
        loss, info = compute_loss_injection(
            model, tau_colloc, i_norm_colloc, params, scales,
            kappa, S_inj, delta_omega,
            tau_data=data['tau'], i_norm_data=data['i_norm'],
            n_data=data['n'], sigma_data=data['sigma'],
            cos_phi_data=data.get('cos_phi'), sin_phi_data=data.get('sin_phi'),
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
                  f"{elapsed:.1f}s")

    if verbose:
        print("  Switching to L-BFGS...")

    lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=lr_lbfgs,
        max_iter=20, history_size=50, line_search_fn='strong_wolfe'
    )
    tau_cf = make_collocation(tau_max, n_colloc)
    i_cf = torch.full_like(tau_cf, I_bias / scales.I_th)

    for step in range(epochs_lbfgs):
        def closure():
            lbfgs.zero_grad()
            loss, _ = compute_loss_injection(
                model, tau_cf, i_cf, params, scales,
                kappa, S_inj, delta_omega,
                tau_data=data['tau'], i_norm_data=data['i_norm'],
                n_data=data['n'], sigma_data=data['sigma'],
                cos_phi_data=data.get('cos_phi'), sin_phi_data=data.get('sin_phi'),
                lam_phys=1.0, lam_data=10.0, causal_eps=1.0,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            return loss
        lbfgs.step(closure)
        _, info = compute_loss_injection(
            model, tau_cf, i_cf, params, scales,
            kappa, S_inj, delta_omega,
            tau_data=data['tau'], i_norm_data=data['i_norm'],
            n_data=data['n'], sigma_data=data['sigma'],
            cos_phi_data=data.get('cos_phi'), sin_phi_data=data.get('sin_phi'),
            lam_phys=1.0, lam_data=10.0, causal_eps=1.0,
        )
        history.append(info)
        if verbose and step % 100 == 0:
            print(f"  L-BFGS {step:3d}/{epochs_lbfgs} | loss {info['total']:.3e}")

    elapsed = time.time() - t_start
    if verbose:
        print(f"  Injection PINN training complete in {elapsed:.1f}s")

    return model, scales, history, ref


# ── Training helpers ──────────────────────────────────────────────────────────

def make_collocation(tau_max, n_points, dense_frac=0.5):
    """Collocation points with denser sampling near tau=0."""
    n_dense = int(n_points * dense_frac)
    n_sparse = n_points - n_dense
    tau_dense = torch.rand(n_dense, 1, dtype=DTYPE, device=device) * (tau_max * 0.2)
    tau_sparse = torch.rand(n_sparse, 1, dtype=DTYPE, device=device) * tau_max
    return torch.sort(torch.cat([tau_dense, tau_sparse], dim=0), dim=0)[0]


def subsample(ref, n_pts):
    """Take n_pts evenly spaced points from reference data (including optional cos_phi, sin_phi)."""
    N_total = ref['tau'].shape[0]
    idx = torch.linspace(0, N_total - 1, n_pts).long()
    keys = [k for k in ('tau', 'n', 'sigma', 'omega', 'i_norm', 'cos_phi', 'sin_phi')
            if k in ref]
    return {k: ref[k][idx] for k in keys}


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
                  epochs=5000, lr_params=1e-2,
                  use_langevin=False, verbose=True):
    """
    Recover unknown laser parameters from (noisy) measurement data.

    Computes rate-equation residuals directly on the data using finite
    differences — no neural network needed for the inverse problem.  This
    ensures gradient signals drive the parameters toward values that make the
    data consistent with the rate equations.

    Uses multiple bias currents for parameter identifiability.

    Parameters
    ----------
    use_langevin : bool
        If True, generate measurement data using Euler-Maruyama integration
        with physically-motivated Langevin noise (shot noise + spontaneous
        emission fluctuations), rather than simple additive Gaussian noise.
        This produces realistic RIN-like noise with the correct spectral
        shape (peak at relaxation oscillation frequency).
    """
    scales = Scales(params_true)

    # Multiple bias currents for identifiability
    I_th = scales.I_th
    I_biases = [2.0 * I_th, 3.0 * I_th, 5.0 * I_th]
    if verbose:
        noise_label = 'Langevin SDE' if use_langevin else f'Gaussian {noise_frac*100:.0f}%'
        print(f"  Using {len(I_biases)} bias currents: "
              + ", ".join(f"{I*1e3:.1f} mA" for I in I_biases))
        print(f"  Noise model: {noise_label}")

    # Generate reference data (simulating experimental measurements)
    data_per_current = []
    if use_langevin:
        # Physically-motivated noise from Langevin (Euler-Maruyama) integration
        for idx_i, Ib in enumerate(I_biases):
            lrefs = generate_reference_stochastic(
                params_true, scales, Ib, t_end, n_points=2000,
                n_realizations=1, seed=42 + idx_i,
            )
            r = lrefs[0]
            N_t = torch.tensor(r['N'], dtype=DTYPE, device=device)
            S_t = torch.tensor(r['S'], dtype=DTYPE, device=device)
            phi_t = torch.tensor(r['phi'], dtype=DTYPE, device=device)
            t_t = torch.tensor(r['t'], dtype=DTYPE, device=device)
            S_t = torch.clamp(S_t, min=1.0)
            data_per_current.append((N_t, S_t, phi_t, t_t, Ib))
    else:
        # Simple additive Gaussian noise
        refs = [generate_reference(params_true, scales, Ib, t_end, n_points=2000)
                for Ib in I_biases]
        torch.manual_seed(42)
        for r, Ib in zip(refs, I_biases):
            N_t = torch.tensor(r['N'], dtype=DTYPE, device=device)
            S_t = torch.tensor(r['S'], dtype=DTYPE, device=device)
            phi_t = torch.tensor(r['phi'], dtype=DTYPE, device=device)
            t_t = torch.tensor(r['t'], dtype=DTYPE, device=device)
            N_t = N_t + noise_frac * N_t.abs().mean() * torch.randn_like(N_t)
            S_t = S_t * (1.0 + noise_frac * torch.randn_like(S_t))
            S_t = torch.clamp(S_t, min=1.0)
            data_per_current.append((N_t, S_t, phi_t, t_t, Ib))

    # Clean deterministic reference at 3× I_th (for display / dummy-model training)
    ref = generate_reference(params_true, scales, I_biases[1], t_end, n_points=500)

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


def plot_forward_injection_results(model, ref, scales, params,
                                    title='Injection PINN vs ODE Solver'):
    """Compare injection PINN predictions vs ODE: N, S, cos/sin φ, chirp."""
    model.eval()
    with torch.no_grad():
        n_p, sig_p, cos_p, sin_p = model(ref['tau'], ref['i_norm'])

    N_pred = (scales.N_tr + n_p.numpy() * scales.N_scale).flatten()
    S_pred = (scales.S_ref * 10.0**(sig_p.numpy())).flatten()
    cos_pred = cos_p.numpy().flatten()
    sin_pred = sin_p.numpy().flatten()

    t_ns = ref['t'] * 1e9
    # Derived chirp: unwrap PINN phase angle and differentiate
    phi_pred = np.unwrap(np.arctan2(sin_pred, cos_pred))
    chirp_pred = np.gradient(phi_pred, ref['t']) / (2 * np.pi) * 1e-9  # GHz
    chirp_ref = np.gradient(ref['phi'], ref['t']) / (2 * np.pi) * 1e-9

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(title, fontsize=14)

    axes[0, 0].plot(t_ns, ref['N'] * 1e-24, 'k-', lw=2, label='ODE')
    axes[0, 0].plot(t_ns, N_pred * 1e-24, 'r--', lw=1.5, label='PINN')
    axes[0, 0].set_ylabel('Carrier density (10$^{24}$ m$^{-3}$)')
    axes[0, 0].set_xlabel('Time (ns)')
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].semilogy(t_ns, ref['S'], 'k-', lw=2, label='ODE')
    axes[0, 1].semilogy(t_ns, np.maximum(S_pred, 1e-10), 'r--', lw=1.5, label='PINN')
    axes[0, 1].set_ylabel('Photon density (m$^{-3}$)')
    axes[0, 1].set_xlabel('Time (ns)')
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    # cos φ and sin φ comparison
    axes[1, 0].plot(t_ns, np.cos(ref['phi']), 'k-', lw=2, label='ODE cos φ')
    axes[1, 0].plot(t_ns, cos_pred, 'r--', lw=1.5, label='PINN cos φ')
    axes[1, 0].plot(t_ns, np.sin(ref['phi']), 'b-', lw=1, alpha=0.7, label='ODE sin φ')
    axes[1, 0].plot(t_ns, sin_pred, 'g--', lw=1, alpha=0.7, label='PINN sin φ')
    axes[1, 0].set_ylabel('cos φ,  sin φ')
    axes[1, 0].set_xlabel('Time (ns)')
    axes[1, 0].legend(fontsize=8); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(t_ns, chirp_ref, 'k-', lw=2, label='ODE chirp')
    axes[1, 1].plot(t_ns, chirp_pred, 'r--', lw=1.5, label='PINN chirp')
    axes[1, 1].set_ylabel('Chirp (GHz)')
    axes[1, 1].set_xlabel('Time (ns)')
    axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

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

    # ── Demo 3: Injection PINN (SLD-seeded DFB laser) ────────────────────
    print()
    print("━" * 65)
    print("  INJECTION PINN — SLD-injected DFB laser dynamics")
    print("━" * 65)

    # Moderate injection at zero detuning (injection-locking regime)
    # kappa derived from eta=0.05: (v_g/2L)*sqrt(1-R1)*eta ≈ 5.5e9 s⁻¹
    kappa_demo = 5.5e9             # injection coupling rate (s⁻¹)
    S_inj_demo = 5e17              # injected photon density (m⁻³)
    delta_omega_demo = 0.0         # zero detuning → clean injection locking

    print(f"  κ = {kappa_demo:.1e} s⁻¹,  S_inj = {S_inj_demo:.1e} m⁻³,  Δν = 0 GHz")
    print(f"  Angular phase encoding: outputs (n, σ, cos φ, sin φ)")
    print()

    model_inj, scales_inj, hist_inj, ref_inj = train_forward_injection(
        params, I_bias, kappa_demo, S_inj_demo, delta_omega_demo,
        t_end=10e-9, epochs_adam=5000, epochs_lbfgs=300,
        n_colloc=2000, n_data_pts=200,
    )

    fig6 = plot_forward_injection_results(
        model_inj, ref_inj, scales_inj, params,
        title='Injection PINN — PINN vs ODE (Lang-Kobayashi)',
    )
    fig6.savefig('pinn_injection_results.png', dpi=150)
    print("  Saved: pinn_injection_results.png")

    fig7 = plot_loss_history(hist_inj, 'Injection PINN — Training Convergence')
    fig7.savefig('pinn_injection_loss.png', dpi=150)
    print("  Saved: pinn_injection_loss.png")

    model_inj.eval()
    with torch.no_grad():
        n_pi, sig_pi, cos_pi, sin_pi = model_inj(ref_inj['tau'], ref_inj['i_norm'])
    N_pi = (scales_inj.N_tr + n_pi.numpy().flatten() * scales_inj.N_scale)
    S_pi = (scales_inj.S_ref * 10.0**(sig_pi.numpy().flatten()))
    mask_inj = ref_inj['S'] > ref_inj['S'].max() * 1e-3
    err_N_inj = np.mean(np.abs(N_pi - ref_inj['N']) / ref_inj['N']) * 100
    err_S_inj = np.mean(np.abs(S_pi[mask_inj] - ref_inj['S'][mask_inj])
                        / ref_inj['S'][mask_inj]) * 100
    circle_err = float(np.mean(cos_pi.numpy()**2 + sin_pi.numpy()**2 - 1))
    # With normalize_angular, unit circle is exact; verify numerically
    circle_dev = float(np.mean(cos_pi.numpy()**2 + sin_pi.numpy()**2)) - 1.0
    print(f"  Accuracy:  N error = {err_N_inj:.2f}%,  S error = {err_S_inj:.2f}%")
    print(f"  Unit circle deviation: {abs(circle_dev)*100:.6f}%  (structural constraint)")

    # ── Demo 4: Inverse mode with Langevin noise ─────────────────────────
    print()
    print("━" * 65)
    print("  INVERSE MODE — Langevin noise (physically-motivated RIN)")
    print("━" * 65)
    print(f"  Learning: {learn_which}")
    print(f"  Noise model: Euler-Maruyama SDE (shot noise + spontaneous emission)")
    print(f"  Initial perturbation: +30%")
    print()

    model_lan, learnable_lan, scales_lan, hist_lan, phist_lan, ref_lan = train_inverse(
        params, I_bias, learn_which,
        t_end=10e-9, perturbation=0.3, noise_frac=0.01,
        epochs=5000, lr_params=1e-2, use_langevin=True,
    )

    fig8 = plot_param_convergence(phist_lan, params, learn_which)
    fig8.suptitle('Inverse Mode — Langevin Noise — Parameter Recovery', fontsize=12)
    fig8.savefig('pinn_inverse_langevin_params.png', dpi=150)
    print("  Saved: pinn_inverse_langevin_params.png")

    # Side-by-side noise comparison plot (Gaussian vs Langevin data, 3× I_th)
    scales_cmp = Scales(params)
    ref_gauss = generate_reference(params, scales_cmp, 3.0 * I_th, 10e-9, n_points=2000)
    lan_refs = generate_reference_stochastic(params, scales_cmp, 3.0 * I_th,
                                              10e-9, n_points=2000,
                                              n_realizations=3, seed=0)
    fig9, axes9 = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig9.suptitle('Noise Comparison: Gaussian vs Langevin (Euler-Maruyama)', fontsize=13)
    t_ns_cmp = ref_gauss['t'] * 1e9
    axes9[0].plot(t_ns_cmp, ref_gauss['N'] * 1e-24, 'k-', lw=1.5, label='Deterministic')
    for ri, lr in enumerate(lan_refs):
        axes9[0].plot(lr['t'] * 1e9, lr['N'] * 1e-24, alpha=0.6, lw=0.8,
                      label=f'Langevin #{ri+1}' if ri == 0 else '_')
    axes9[0].set_ylabel('Carrier density (10$^{24}$ m$^{-3}$)')
    axes9[0].legend(); axes9[0].grid(True, alpha=0.3)
    axes9[1].semilogy(t_ns_cmp, ref_gauss['S'], 'k-', lw=1.5, label='Deterministic')
    for ri, lr in enumerate(lan_refs):
        axes9[1].semilogy(lr['t'] * 1e9, np.maximum(lr['S'], 1.0), alpha=0.6, lw=0.8,
                          label=f'Langevin #{ri+1}' if ri == 0 else '_')
    axes9[1].set_ylabel('Photon density (m$^{-3}$)')
    axes9[1].set_xlabel('Time (ns)')
    axes9[1].legend(); axes9[1].grid(True, alpha=0.3)
    plt.tight_layout()
    fig9.savefig('pinn_langevin_noise.png', dpi=150)
    print("  Saved: pinn_langevin_noise.png")

    plt.close('all')
    print()
    print("=" * 65)
    print("  Done. All plots saved to working directory.")
    print("=" * 65)
