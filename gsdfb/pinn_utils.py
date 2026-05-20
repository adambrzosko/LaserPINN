"""
Shared PINN / ML utilities for gain-switched DFB laser simulations.

Consolidates:
- Scales class (normalisation constants)
- add_noise_snr (synthetic noise injection)
- generate_gain_switched_data (ODE-based training data)
- Common training helpers

Used by: pinn_inverse_extraction, bayesian_pinn, transfer_learning,
         neural_ode_noise, neural_surrogate, pinn_bandwidth
"""
import math
import numpy as np
import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.dfb_laser import DFBLaserParams, solve_transient, q

DEVICE = torch.device('cpu')
DTYPE = torch.float64


# ── Normalisation scales ─────────────────────────────────────────────────────

class Scales:
    """Normalisation constants derived from laser parameters.

    All PINNs use the same scaling to keep activations O(1):
        - Time normalised by t_ref = 1 ns
        - Carrier density as (N - N_tr) / N_scale
        - Photon density in log10 space: sigma = log10(S / S_ref)
    """

    def __init__(self, params: DFBLaserParams):
        self.t_ref = 1e-9                                         # 1 ns
        self.N_tr = params.N_tr
        g_th = (params.alpha_i + params.alpha_m) / params.Gamma
        N_th = params.N_tr + g_th / params.a
        self.N_scale = N_th - params.N_tr                         # ~5.3e23
        self.N_th = N_th
        self.S_ref = 1e21                                          # characteristic S
        self.omega_scale = 2 * math.pi * 10e9                     # ~10 GHz
        self.I_th = q * params.V * N_th / params.carrier_lifetime(N_th)

    def normalise_t(self, t):
        """t (s) -> tau (dimensionless)."""
        return t / self.t_ref

    def normalise_N(self, N):
        """N (m^-3) -> n (dimensionless, ~O(1))."""
        return (N - self.N_tr) / self.N_scale

    def normalise_S(self, S):
        """S (m^-3) -> sigma (log10 scale)."""
        return np.log10(np.maximum(S, 1e-10) / self.S_ref)

    def denormalise_N(self, n):
        """n -> N (m^-3)."""
        return self.N_tr + n * self.N_scale

    def denormalise_S(self, sigma):
        """sigma -> S (m^-3)."""
        return self.S_ref * 10.0 ** sigma


# ── Noise injection ──────────────────────────────────────────────────────────

def add_noise_snr(data, snr_db=25, seed=42):
    """Add Gaussian noise to a signal at a specified SNR (dB).

    Works with both numpy arrays and torch tensors.

    Parameters
    ----------
    data : array or tensor — clean signal
    snr_db : float — signal-to-noise ratio in dB
    seed : int — random seed

    Returns
    -------
    noisy : same type as input — signal + noise, clamped >= 1e-10
    """
    if isinstance(data, torch.Tensor):
        torch.manual_seed(seed)
        power = (data ** 2).mean()
        noise_power = power / (10.0 ** (snr_db / 10.0))
        noise = torch.randn_like(data) * torch.sqrt(noise_power)
        return torch.clamp(data + noise, min=1e-10)
    else:
        rng = np.random.default_rng(seed)
        power = np.mean(data ** 2)
        noise_power = power / (10.0 ** (snr_db / 10.0))
        noise = rng.normal(0, np.sqrt(noise_power), size=data.shape)
        return np.maximum(data + noise, 1e-10)


# ── Data generation ──────────────────────────────────────────────────────────

def generate_gain_switched_data(params, f_rep=5e9, n_periods=10,
                                 pts_per_period=400):
    """Generate a gain-switched pulse train using the ODE solver.

    Returns dict with 't', 'N', 'S', 'phi', 'I_func', 'I_th', 'f_rep', 'T_rep'.
    """
    I_th = params.threshold_current()
    I_off = 0.9 * I_th
    I_on = 5.0 * I_th
    T_rep = 1.0 / f_rep
    t_on = 0.3 * T_rep
    t_rise = min(20e-12, t_on / 4.0)
    t_end = n_periods * T_rep
    n_pts = n_periods * pts_per_period

    def I_func(t):
        t_in = t % T_rep
        if t_in < t_rise:
            f = 0.5 * (1.0 - np.cos(np.pi * t_in / t_rise))
            return I_off + (I_on - I_off) * f
        elif t_in < t_on - t_rise:
            return I_on
        elif t_in < t_on:
            f = 0.5 * (1.0 - np.cos(np.pi * (t_on - t_in) / t_rise))
            return I_off + (I_on - I_off) * f
        else:
            return I_off

    t_eval = np.linspace(0, t_end, n_pts)
    sol = solve_transient(params, I_func, [0, t_end], t_eval=t_eval)
    N, S, phi = sol.y

    return {
        't': sol.t, 'N': N, 'S': S, 'phi': phi,
        'I_func': I_func, 'I_th': I_th,
        'f_rep': f_rep, 'T_rep': T_rep,
    }


# ── Common PINN building blocks ─────────────────────────────────────────────

def make_residual_mlp(n_in, n_out, hidden_dim=96, n_layers=5,
                      activation='tanh', dropout_p=0.0):
    """Build a residual MLP with skip connections every 2 layers.

    Returns nn.Module with forward(x) -> (n_out,) tensor.
    """
    act = {'tanh': nn.Tanh, 'relu': nn.ReLU, 'gelu': nn.GELU}[activation]

    class ResidualMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layer = nn.Linear(n_in, hidden_dim)
            self.hidden = nn.ModuleList(
                [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)]
            )
            self.output_layer = nn.Linear(hidden_dim, n_out)
            self.act = act()
            self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.zeros_(m.bias)

        def forward(self, x):
            h = self.act(self.input_layer(x))
            for k, layer in enumerate(self.hidden):
                h_new = self.act(layer(self.dropout(h)))
                if k % 2 == 1:
                    h_new = h_new + h  # skip connection
                h = h_new
            return self.output_layer(h)

    return ResidualMLP()


def to_tensor(arr, requires_grad=False):
    """Convert numpy array to torch float64 tensor on CPU."""
    t = torch.tensor(arr, dtype=DTYPE, device=DEVICE)
    if requires_grad:
        t = t.requires_grad_(True)
    return t


# ── Learnable parameters ─────────────────────────────────────────────────────

# ── LaserPINN model ──────────────────────────────────────────────────────────

class LaserPINN(nn.Module):
    """PINN for laser rate equations.

    Inputs:  tau (normalised time), i_norm (normalised current)
    Outputs: n (carrier density), sigma (log10 photon density),
             omega_norm (normalised chirp rate)
    """

    def __init__(self, hidden_dim=128, n_layers=6, n_out=3,
                 normalize_angular=False):
        super().__init__()
        self.n_out = n_out
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
                h_new = h_new + h
            h = h_new
        out = self.output_layer(h)
        outputs = [out[:, i:i+1] for i in range(self.n_out)]
        if self.normalize_angular and self.n_out >= 4:
            cos_r, sin_r = outputs[2], outputs[3]
            r = torch.sqrt(cos_r ** 2 + sin_r ** 2 + 1e-8)
            outputs[2] = cos_r / r
            outputs[3] = sin_r / r
        return tuple(outputs)


# ── Physics residuals ────────────────────────────────────────────────────────

def physics_residuals(model, tau, i_norm, params, scales, learnable=None):
    """Compute rate-equation residuals via autograd.

    Returns (res_n, res_sigma, res_omega) each shape (B, 1).
    """
    tau = tau.requires_grad_(True)
    n, sigma, omega_norm = model(tau, i_norm)

    ones = torch.ones_like(n)
    dn_dtau = torch.autograd.grad(n, tau, ones, create_graph=True,
                                   retain_graph=True)[0]
    dsigma_dtau = torch.autograd.grad(sigma, tau, ones, create_graph=True,
                                       retain_graph=True)[0]

    def P(name):
        if learnable is not None:
            return learnable.get(name)
        return torch.tensor(getattr(params, name), dtype=DTYPE, device=DEVICE)

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
    v_g = torch.tensor(params.v_g, dtype=DTYPE, device=DEVICE)
    V = torch.tensor(params.V, dtype=DTYPE, device=DEVICE)
    tau_p = torch.tensor(params.tau_p, dtype=DTYPE, device=DEVICE)
    beta_sp = P('beta_sp')
    alpha_H = P('alpha_H')

    q_val = torch.tensor(q, dtype=DTYPE, device=DEVICE)
    t_ref = torch.tensor(scales.t_ref, dtype=DTYPE, device=DEVICE)
    ln10 = torch.tensor(math.log(10.0), dtype=DTYPE, device=DEVICE)
    N_scale_t = torch.tensor(scales.N_scale, dtype=DTYPE, device=DEVICE)
    omega_scale_t = torch.tensor(scales.omega_scale, dtype=DTYPE, device=DEVICE)

    rhs_n = (t_ref / N_scale_t) * (I / (q_val * V) - R_sp - Gamma * v_g * g * S)
    rhs_sigma = (t_ref / ln10) * (
        Gamma * v_g * g - 1.0 / tau_p + beta_sp * B_coeff * N**2 / S
    )
    rhs_omega = (t_ref / omega_scale_t) * 0.5 * alpha_H * (
        Gamma * v_g * a * (N - N_tr) - 1.0 / tau_p
    )

    return dn_dtau - rhs_n, dsigma_dtau - rhs_sigma, omega_norm - rhs_omega


def compute_loss(model, tau_colloc, i_norm_colloc, params, scales,
                 tau_data=None, i_norm_data=None,
                 n_data=None, sigma_data=None, omega_data=None,
                 learnable=None,
                 lam_phys=1.0, lam_data=1.0, causal_eps=1.0):
    """Total PINN loss = physics residual + data fitting."""
    res_n, res_sigma, res_omega = physics_residuals(
        model, tau_colloc, i_norm_colloc, params, scales, learnable
    )

    r2 = (res_n.detach()**2 + res_sigma.detach()**2 + res_omega.detach()**2)
    r2_clamped = torch.clamp(r2, max=50.0)
    w_causal = torch.exp(-causal_eps * torch.cumsum(r2_clamped, dim=0))
    w_causal = w_causal / (w_causal.mean() + 1e-30)

    L_phys = torch.mean(w_causal * (res_n**2 + res_sigma**2 + res_omega**2))

    L_data = torch.tensor(0.0, dtype=DTYPE, device=DEVICE)
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


# ── Learnable parameters ─────────────────────────────────────────────────────

class LearnableParams(nn.Module):
    """Wraps selected DFBLaserParams fields as optimisable log-parameters.

    Usage:
        learnable = LearnableParams(params, ['alpha_H', 'epsilon'])
        alpha_H = learnable.get('alpha_H')   # differentiable tensor
        fixed   = learnable.get('A')          # fixed tensor from base params
    """

    def __init__(self, base_params, learn_which):
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
        return torch.tensor(getattr(self.base, name), dtype=DTYPE, device=DEVICE)

    def current_values(self):
        return {n: self.get(n).item() for n in self.learn_which}


# ── Collocation helpers ──────────────────────────────────────────────────────

def make_collocation(tau_max, n_points, dense_frac=0.5):
    """Collocation points with denser sampling near tau=0."""
    n_dense = int(n_points * dense_frac)
    n_sparse = n_points - n_dense
    tau_dense = torch.rand(n_dense, 1, dtype=DTYPE, device=DEVICE) * (tau_max * 0.2)
    tau_sparse = torch.rand(n_sparse, 1, dtype=DTYPE, device=DEVICE) * tau_max
    return torch.sort(torch.cat([tau_dense, tau_sparse], dim=0), dim=0)[0]


def subsample(ref, n_pts):
    """Take n_pts evenly spaced points from reference data dict."""
    N_total = ref['tau'].shape[0]
    idx = torch.linspace(0, N_total - 1, n_pts).long()
    keys = [k for k in ('tau', 'n', 'sigma', 'omega', 'i_norm', 'cos_phi', 'sin_phi')
            if k in ref]
    return {k: ref[k][idx] for k in keys}


def generate_reference(params, scales, I_bias, t_end=10e-9, n_points=2000):
    """Run ODE solver at constant bias and return normalised arrays + torch tensors.

    Returns dict with keys: tau, n, sigma, omega, i_norm (tensors),
    plus N, S, phi, t (numpy arrays).
    """
    sol = solve_transient(params, lambda t: I_bias, [0, t_end],
                          t_eval=np.linspace(0, t_end, n_points))
    N_arr, S_arr, phi_arr = sol.y

    tau_np = sol.t / scales.t_ref
    n_np = (N_arr - scales.N_tr) / scales.N_scale
    sigma_np = np.log10(np.maximum(S_arr, 1e-30) / scales.S_ref)
    chirp_np = np.gradient(phi_arr, sol.t)
    omega_np = chirp_np * scales.t_ref / scales.omega_scale
    i_norm_np = np.full_like(tau_np, I_bias / scales.I_th)

    def _to_t(arr):
        return torch.tensor(arr, dtype=DTYPE, device=DEVICE).reshape(-1, 1)

    return {
        'tau': _to_t(tau_np), 'n': _to_t(n_np), 'sigma': _to_t(sigma_np),
        'omega': _to_t(omega_np), 'i_norm': _to_t(i_norm_np),
        'N': N_arr, 'S': S_arr, 'phi': phi_arr, 't': sol.t,
    }
