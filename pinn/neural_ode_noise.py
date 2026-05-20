"""
#10 -- Neural ODE for Noise Model

Replaces the Langevin noise terms in the semiconductor laser rate equations
with a learned stochastic forcing function. Trains via moment-matching to
reproduce the statistics of the standard Langevin model.

Compares learned noise amplitudes to analytical values:
  sigma_N ~ sqrt(2 * R_sp)
  sigma_S ~ sqrt(2 * beta_sp * B * N^2)

Produces figures in images/neural_ode/:
  1. noise_amplitudes.png       - learned vs analytical Langevin noise
  2. statistics_comparison.png  - histograms of peak power and phase differences
  3. training_loss.png          - loss components vs epoch
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from core.dfb_laser import DFBLaserParams, q
from core.million_pulse_comparison import simulate_pulses
from gsdfb.analysis import compute_r1
from gsdfb.plotting import setup_plotting, save_fig, img_dir

# ── Output directory ─────────────────────────────────────────────────────────

OUT_DIR = img_dir('neural_ode')


# ── Laser setup ──────────────────────────────────────────────────────────────

laser = DFBLaserParams()
I_th = laser.threshold_current()
setup_plotting()

print("=" * 70)
print("  Neural ODE for Noise Model")
print("=" * 70)
print(f"  Laser: I_th = {I_th*1e3:.2f} mA, tau_p = {laser.tau_p*1e12:.1f} ps")
print()


# ── Step 1: Generate ground truth statistics ─────────────────────────────────

F_REP = 5e9           # 5 GHz repetition rate
N_PULSES_GT = 100000  # ground truth: 100k pulses
N_DISCARD = 200
DT = 1.0e-12
DUTY = 0.30
I_BIAS_FACTOR = 0.9
I_PEAK_FACTOR = 5.0

T_rep = 1.0 / F_REP
t_on = DUTY * T_rep
pts_period = max(int(round(T_rep / DT)), 50)
dt_actual = T_rep / pts_period
t_rise = min(20e-12, t_on / 4.0)

I_off = I_BIAS_FACTOR * I_th
I_on = I_PEAK_FACTOR * I_th

print("  Generating ground truth (100k pulses at 5 GHz, free-running)...")
print("  Compiling JIT solver...", end="", flush=True)
t0 = time.time()
_ = simulate_pulses(
    100, 10, 100, 1e-12,
    I_off, I_on, 30e-12, 7e-12,
    laser.V, laser.Gamma, laser.v_g, laser.a, laser.N_tr, laser.epsilon,
    laser.A, laser.B, laser.C, laser.tau_p, laser.beta_sp, laser.alpha_H, q,
    0.0, 0)
print(f" done ({time.time()-t0:.1f}s)")

t0 = time.time()
gt_phi, gt_peak_S, gt_samp_S, gt_peak_k = simulate_pulses(
    N_PULSES_GT + N_DISCARD, N_DISCARD, pts_period, dt_actual,
    I_off, I_on, t_on, t_rise,
    laser.V, laser.Gamma, laser.v_g, laser.a, laser.N_tr, laser.epsilon,
    laser.A, laser.B, laser.C, laser.tau_p, laser.beta_sp, laser.alpha_H, q,
    0.0, 42)
t_gt = time.time() - t0
print(f"  Ground truth generated in {t_gt:.1f}s")

# Ground truth statistics
gt_peak_P = laser.output_power(np.maximum(gt_peak_S, 0))
gt_r1, gt_dphi = compute_r1(gt_phi)

gt_mean_S = np.mean(gt_peak_S)
gt_std_S = np.std(gt_peak_S)
gt_jitter = np.std(gt_peak_k * dt_actual)

print(f"  Ground truth stats:")
print(f"    mean(peak_S) = {gt_mean_S:.3e} m^-3")
print(f"    std(peak_S)  = {gt_std_S:.3e} m^-3")
print(f"    r1           = {gt_r1:.4f}")
print(f"    jitter       = {gt_jitter*1e12:.2f} ps")


# ── Step 2: Neural SDE model ────────────────────────────────────────────────

class NoiseNetwork(nn.Module):
    """Small network: f_noise(N, S, t) -> (sigma_N, sigma_S, sigma_phi).

    Outputs are positive (softplus activation) and represent noise amplitudes.
    """

    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 3),
            nn.Softplus(),  # ensure positive outputs
        )
        # Initialize to roughly correct scale
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, N_norm, S_norm, t_norm):
        """
        Inputs are normalized: N/N_tr, S/S_ref, t/T_rep.
        Outputs: (sigma_N, sigma_S, sigma_phi) in physical units (scaled).
        """
        x = torch.stack([N_norm, S_norm, t_norm], dim=-1)
        return self.net(x)


class NeuralSDE(nn.Module):
    """
    Neural SDE for gain-switched laser.
    Deterministic part: standard rate equations (hardcoded).
    Stochastic part: learned noise amplitudes.
    """

    def __init__(self, laser_params):
        super().__init__()
        self.noise_net = NoiseNetwork(hidden=32)

        # Store laser parameters as buffers
        self.register_buffer('V', torch.tensor(laser_params.V, dtype=torch.float64))
        self.register_buffer('Gamma', torch.tensor(laser_params.Gamma, dtype=torch.float64))
        self.register_buffer('v_g', torch.tensor(laser_params.v_g, dtype=torch.float64))
        self.register_buffer('a_gain', torch.tensor(laser_params.a, dtype=torch.float64))
        self.register_buffer('N_tr', torch.tensor(laser_params.N_tr, dtype=torch.float64))
        self.register_buffer('eps', torch.tensor(laser_params.epsilon, dtype=torch.float64))
        self.register_buffer('A_nr', torch.tensor(laser_params.A, dtype=torch.float64))
        self.register_buffer('B_rad', torch.tensor(laser_params.B, dtype=torch.float64))
        self.register_buffer('C_aug', torch.tensor(laser_params.C, dtype=torch.float64))
        self.register_buffer('tau_p', torch.tensor(laser_params.tau_p, dtype=torch.float64))
        self.register_buffer('beta_sp', torch.tensor(laser_params.beta_sp, dtype=torch.float64))
        self.register_buffer('alpha_H', torch.tensor(laser_params.alpha_H, dtype=torch.float64))
        self.register_buffer('q_e', torch.tensor(q, dtype=torch.float64))

        # Normalization scales
        self.register_buffer('N_scale', torch.tensor(laser_params.N_tr, dtype=torch.float64))
        self.register_buffer('S_scale', torch.tensor(1e21, dtype=torch.float64))

    def current_waveform(self, t, I_off, I_on, t_on, t_rise, T_rep):
        """Raised cosine waveform (differentiable)."""
        # Periodic time within one period
        t_mod = t % T_rep

        # Piecewise smooth: rising, high, falling, low
        # Use sigmoid approximation for differentiability
        k = 50.0 / t_rise  # sharpness

        rise = torch.sigmoid(k * t_mod) * torch.sigmoid(k * (t_on - t_mod))
        I = I_off + (I_on - I_off) * rise
        return I

    def forward(self, batch_size, n_steps, dt, I_off, I_on, t_on, t_rise, T_rep):
        """
        Run batch_size independent pulse simulations.
        Uses Euler-Maruyama with reparameterization trick.

        Returns: peak_S (batch_size,), peak_phi (batch_size,), peak_k (batch_size,)
        """
        sqrt_dt = np.sqrt(dt)

        # Initial conditions
        N = torch.full((batch_size,), self.N_tr.item() * 1.2, dtype=torch.float64)
        Er = torch.full((batch_size,), np.sqrt(1e10), dtype=torch.float64)
        Ei = torch.zeros(batch_size, dtype=torch.float64)

        # Track peak photon density per pulse
        best_S = torch.zeros(batch_size, dtype=torch.float64)
        best_phi = torch.zeros(batch_size, dtype=torch.float64)
        best_k = torch.zeros(batch_size, dtype=torch.float64)

        for k in range(n_steps):
            t = k * dt
            t_mod = t % T_rep

            # Current waveform (non-differentiable but fast)
            if t_mod < t_rise:
                f = 0.5 * (1.0 - np.cos(np.pi * t_mod / t_rise))
                I_cur = I_off + (I_on - I_off) * f
            elif t_mod < t_on - t_rise:
                I_cur = I_on
            elif t_mod < t_on:
                f = 0.5 * (1.0 - np.cos(np.pi * (t_on - t_mod) / t_rise))
                I_cur = I_off + (I_on - I_off) * f
            else:
                I_cur = I_off

            S = Er * Er + Ei * Ei
            S = torch.clamp(S, min=1e-10)

            # Gain
            g = self.a_gain * (N - self.N_tr) / (1.0 + self.eps * S)

            # Recombination
            R_sp = self.A_nr * N + self.B_rad * N * N + self.C_aug * N * N * N
            R_sp_mode = self.beta_sp * self.B_rad * N * N

            # Net gain for field
            gamma = 0.5 * (self.Gamma * self.v_g * g - 1.0 / self.tau_p)

            # Field update (deterministic)
            Er_new = (1.0 + gamma * dt) * Er - gamma * self.alpha_H * dt * Ei
            Ei_new = (1.0 + gamma * dt) * Ei + gamma * self.alpha_H * dt * Er
            Er = Er_new
            Ei = Ei_new

            # Neural noise (reparameterization trick)
            N_norm = N / self.N_scale
            S_norm = S / self.S_scale
            t_norm = torch.full_like(N, t_mod / T_rep)

            sigma = self.noise_net(N_norm, S_norm, t_norm)  # (batch, 3)
            sigma_N = sigma[:, 0] * 1e12   # scale to physical units
            sigma_S = sigma[:, 1] * 1e5    # field noise scale
            sigma_phi = sigma[:, 2] * 1e5  # field noise scale

            # Reparameterization: noise = sigma * epsilon, epsilon ~ N(0,1)
            eps_Er = torch.randn(batch_size, dtype=torch.float64)
            eps_Ei = torch.randn(batch_size, dtype=torch.float64)
            eps_N = torch.randn(batch_size, dtype=torch.float64)

            Er = Er + sigma_S * sqrt_dt * eps_Er
            Ei = Ei + sigma_phi * sqrt_dt * eps_Ei

            # Carrier density update
            dN = (I_cur / (self.q_e * self.V) - R_sp
                  - self.Gamma * self.v_g * g * S) * dt
            N = N + dN + sigma_N * sqrt_dt * eps_N

            # Track peak
            S_now = Er * Er + Ei * Ei
            mask = S_now > best_S
            best_S = torch.where(mask, S_now, best_S)
            phi_now = torch.atan2(Ei, Er)
            best_phi = torch.where(mask, phi_now, best_phi)
            best_k = torch.where(mask, torch.full_like(best_k, float(k)), best_k)

        return best_S, best_phi, best_k


# ── Step 3: Training ─────────────────────────────────────────────────────────

print("\n  Building Neural SDE model...")
model = NeuralSDE(laser).double()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

# Training parameters
N_EPOCHS = 600
BATCH_SIZE = 200           # pulses per forward pass
N_STEPS = pts_period       # steps per pulse period (one pulse)
N_PERIODS_WARMUP = 3       # warmup periods before collecting statistics
N_PERIODS_COLLECT = 5      # periods to collect statistics from

# Ground truth targets (tensors)
target_mean_S = torch.tensor(gt_mean_S, dtype=torch.float64)
target_std_S = torch.tensor(gt_std_S, dtype=torch.float64)
target_r1 = torch.tensor(gt_r1, dtype=torch.float64)
target_jitter = torch.tensor(gt_jitter, dtype=torch.float64)

# Loss weights
w_mean = 1.0
w_std = 1.0
w_r1 = 10.0    # phase correlation is important
w_jitter = 1.0

print(f"  Training: {N_EPOCHS} epochs, batch_size={BATCH_SIZE}")
print(f"  Per batch: {N_PERIODS_WARMUP} warmup + {N_PERIODS_COLLECT} collect periods")
print(f"  Steps per period: {N_STEPS}")

t_train_start = time.time()
loss_history = {'total': [], 'mean_S': [], 'std_S': [], 'r1': [], 'jitter': []}


def run_neural_sde_batch(model, batch_size, n_periods):
    """Run neural SDE for multiple periods, return per-pulse peaks."""
    all_peak_S = []
    all_peak_phi = []
    all_peak_k = []

    # Initial conditions
    N = torch.full((batch_size,), laser.N_tr * 1.2, dtype=torch.float64)
    Er = torch.full((batch_size,), np.sqrt(1e10), dtype=torch.float64)
    Ei = torch.zeros(batch_size, dtype=torch.float64)

    sqrt_dt = np.sqrt(dt_actual)

    for period in range(n_periods):
        best_S = torch.zeros(batch_size, dtype=torch.float64)
        best_phi = torch.zeros(batch_size, dtype=torch.float64)
        best_k = torch.zeros(batch_size, dtype=torch.float64)

        for k in range(N_STEPS):
            t_mod = k * dt_actual

            # Current waveform
            if t_mod < t_rise:
                f = 0.5 * (1.0 - np.cos(np.pi * t_mod / t_rise))
                I_cur = I_off + (I_on - I_off) * f
            elif t_mod < t_on - t_rise:
                I_cur = I_on
            elif t_mod < t_on:
                f = 0.5 * (1.0 - np.cos(np.pi * (t_on - t_mod) / t_rise))
                I_cur = I_off + (I_on - I_off) * f
            else:
                I_cur = I_off

            S = Er * Er + Ei * Ei
            S = torch.clamp(S, min=1e-10)

            g = model.a_gain * (N - model.N_tr) / (1.0 + model.eps * S)
            R_sp = model.A_nr * N + model.B_rad * N * N + model.C_aug * N * N * N

            gamma = 0.5 * (model.Gamma * model.v_g * g - 1.0 / model.tau_p)

            # Field update
            Er_new = (1.0 + gamma * dt_actual) * Er - gamma * model.alpha_H * dt_actual * Ei
            Ei_new = (1.0 + gamma * dt_actual) * Ei + gamma * model.alpha_H * dt_actual * Er
            Er = Er_new
            Ei = Ei_new

            # Neural noise
            N_norm = N / model.N_scale
            S_norm = S / model.S_scale
            t_norm_val = torch.full_like(N, t_mod / T_rep)

            sigma = model.noise_net(N_norm, S_norm, t_norm_val)
            sigma_N = sigma[:, 0] * 1e12
            sigma_S = sigma[:, 1] * 1e5
            sigma_phi = sigma[:, 2] * 1e5

            eps_Er = torch.randn(batch_size, dtype=torch.float64)
            eps_Ei = torch.randn(batch_size, dtype=torch.float64)
            eps_N = torch.randn(batch_size, dtype=torch.float64)

            Er = Er + sigma_S * sqrt_dt * eps_Er
            Ei = Ei + sigma_phi * sqrt_dt * eps_Ei

            dN = (I_cur / (model.q_e * model.V) - R_sp
                  - model.Gamma * model.v_g * g * S) * dt_actual
            N = N + dN + sigma_N * sqrt_dt * eps_N

            # Track peak
            S_now = Er * Er + Ei * Ei
            mask = S_now > best_S
            best_S = torch.where(mask, S_now, best_S)
            phi_now = torch.atan2(Ei, Er)
            best_phi = torch.where(mask, phi_now, best_phi)
            best_k = torch.where(mask, torch.full_like(best_k, float(k)), best_k)

        # After warmup, collect statistics
        if period >= N_PERIODS_WARMUP:
            all_peak_S.append(best_S)
            all_peak_phi.append(best_phi)
            all_peak_k.append(best_k)

    # Stack results
    peak_S_all = torch.stack(all_peak_S, dim=0)   # (n_collect, batch)
    peak_phi_all = torch.stack(all_peak_phi, dim=0)
    peak_k_all = torch.stack(all_peak_k, dim=0)

    return peak_S_all, peak_phi_all, peak_k_all


print("\n  Training loop...")
for epoch in range(N_EPOCHS):
    model.train()

    # Run neural SDE
    peak_S_all, peak_phi_all, peak_k_all = run_neural_sde_batch(
        model, BATCH_SIZE, N_PERIODS_WARMUP + N_PERIODS_COLLECT)

    # Flatten across periods and batch
    peak_S_flat = peak_S_all.flatten()
    peak_phi_flat = peak_phi_all.flatten()
    peak_k_flat = peak_k_all.flatten()

    # Compute statistics
    pred_mean_S = peak_S_flat.mean()
    pred_std_S = peak_S_flat.std()

    # Phase correlation: compute across consecutive periods for same batch element
    # Use consecutive peaks within same batch element
    if peak_phi_all.shape[0] > 1:
        dphi_consecutive = peak_phi_all[1:] - peak_phi_all[:-1]
        # Wrap to [-pi, pi]
        dphi_wrapped = torch.atan2(torch.sin(dphi_consecutive),
                                    torch.cos(dphi_consecutive))
        exp_dphi = torch.exp(1j * dphi_wrapped.to(torch.complex128))
        pred_r1 = torch.abs(exp_dphi.mean()).to(torch.float64)
    else:
        pred_r1 = torch.tensor(0.0, dtype=torch.float64)

    # Timing jitter
    pred_jitter = peak_k_flat.std() * dt_actual

    # Moment-matching loss
    loss_mean = w_mean * ((pred_mean_S - target_mean_S) / target_mean_S) ** 2
    loss_std = w_std * ((pred_std_S - target_std_S) / target_std_S) ** 2
    loss_r1 = w_r1 * (pred_r1 - target_r1) ** 2
    loss_jitter = w_jitter * ((pred_jitter - target_jitter) / target_jitter) ** 2

    loss = loss_mean + loss_std + loss_r1 + loss_jitter

    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    loss_history['total'].append(loss.item())
    loss_history['mean_S'].append(loss_mean.item())
    loss_history['std_S'].append(loss_std.item())
    loss_history['r1'].append(loss_r1.item())
    loss_history['jitter'].append(loss_jitter.item())

    if (epoch + 1) % 100 == 0:
        print(f"    Epoch {epoch+1:4d}: loss={loss.item():.4f} "
              f"(mean={loss_mean.item():.4f}, std={loss_std.item():.4f}, "
              f"r1={loss_r1.item():.4f}, jit={loss_jitter.item():.4f})")
        print(f"      pred: mean_S={pred_mean_S.item():.3e}, "
              f"std_S={pred_std_S.item():.3e}, "
              f"r1={pred_r1.item():.4f}, jitter={pred_jitter.item()*1e12:.2f}ps")

t_train = time.time() - t_train_start
print(f"\n  Training complete: {t_train:.1f}s")


# ── Step 4: Compare learned vs analytical noise ──────────────────────────────

print("\n  Comparing learned vs analytical noise amplitudes...")

model.eval()

# Evaluate noise network at a grid of (N, S) values
N_grid = np.linspace(0.8 * laser.N_tr, 2.5 * laser.N_tr, 50)
S_grid = np.linspace(1e18, 1e22, 50)

# Fix t_norm = 0.5 (mid-pulse)
t_norm_fixed = 0.5

# Evaluate at varying N (fixed S at steady-state level)
S_fixed = gt_mean_S
learned_sigma_N_vs_N = np.zeros(len(N_grid))
learned_sigma_S_vs_N = np.zeros(len(N_grid))
analytical_sigma_N_vs_N = np.zeros(len(N_grid))
analytical_sigma_S_vs_N = np.zeros(len(N_grid))

with torch.no_grad():
    for i, N_val in enumerate(N_grid):
        N_norm = torch.tensor(N_val / laser.N_tr, dtype=torch.float64)
        S_norm = torch.tensor(S_fixed / 1e21, dtype=torch.float64)
        t_norm = torch.tensor(t_norm_fixed, dtype=torch.float64)

        sigma = model.noise_net(N_norm.unsqueeze(0), S_norm.unsqueeze(0),
                                t_norm.unsqueeze(0))
        learned_sigma_N_vs_N[i] = sigma[0, 0].item() * 1e12  # physical units
        learned_sigma_S_vs_N[i] = sigma[0, 1].item() * 1e5

        # Analytical Langevin
        R_sp = laser.A * N_val + laser.B * N_val**2 + laser.C * N_val**3
        R_sp_mode = laser.beta_sp * laser.B * N_val**2
        analytical_sigma_N_vs_N[i] = np.sqrt(2 * R_sp)
        analytical_sigma_S_vs_N[i] = np.sqrt(R_sp_mode)  # field noise amplitude

# Evaluate at varying S (fixed N at threshold-level)
N_fixed = laser.N_tr * 1.5
learned_sigma_N_vs_S = np.zeros(len(S_grid))
learned_sigma_S_vs_S = np.zeros(len(S_grid))
analytical_sigma_N_vs_S = np.zeros(len(S_grid))
analytical_sigma_S_vs_S = np.zeros(len(S_grid))

with torch.no_grad():
    for i, S_val in enumerate(S_grid):
        N_norm = torch.tensor(N_fixed / laser.N_tr, dtype=torch.float64)
        S_norm = torch.tensor(S_val / 1e21, dtype=torch.float64)
        t_norm = torch.tensor(t_norm_fixed, dtype=torch.float64)

        sigma = model.noise_net(N_norm.unsqueeze(0), S_norm.unsqueeze(0),
                                t_norm.unsqueeze(0))
        learned_sigma_N_vs_S[i] = sigma[0, 0].item() * 1e12
        learned_sigma_S_vs_S[i] = sigma[0, 1].item() * 1e5

        # Analytical (N is fixed, so R_sp is constant)
        R_sp = laser.A * N_fixed + laser.B * N_fixed**2 + laser.C * N_fixed**3
        R_sp_mode = laser.beta_sp * laser.B * N_fixed**2
        analytical_sigma_N_vs_S[i] = np.sqrt(2 * R_sp)
        analytical_sigma_S_vs_S[i] = np.sqrt(R_sp_mode)


# ── Step 5: Generate statistics from trained model ───────────────────────────

print("  Generating statistics from trained neural SDE...")

model.eval()
with torch.no_grad():
    # Run larger batch for statistics comparison
    nsde_peak_S, nsde_peak_phi, nsde_peak_k = run_neural_sde_batch(
        model, 500, N_PERIODS_WARMUP + 10)

nsde_peak_S_flat = nsde_peak_S.flatten().numpy()
nsde_peak_phi_flat = nsde_peak_phi.flatten().numpy()
nsde_peak_P = laser.output_power(np.maximum(nsde_peak_S_flat, 0))

# Phase differences from neural SDE
nsde_dphi_all = []
for b in range(nsde_peak_phi.shape[1]):
    phi_seq = nsde_peak_phi[:, b].numpy()
    _, dphi = compute_r1(phi_seq)
    nsde_dphi_all.extend(dphi.tolist())
nsde_dphi = np.array(nsde_dphi_all)


# ── Figure 1: Noise amplitudes ───────────────────────────────────────────────

fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))
fig1.suptitle('Learned vs Analytical Langevin Noise Amplitudes', fontsize=13)

# sigma_N vs N
ax = axes1[0, 0]
ax.plot(N_grid / laser.N_tr, learned_sigma_N_vs_N, 'b-', lw=2, label='Learned')
ax.plot(N_grid / laser.N_tr, analytical_sigma_N_vs_N, 'r--', lw=2, label='Analytical')
ax.set_xlabel('$N / N_{tr}$')
ax.set_ylabel('$\\sigma_N$ (arb. units)')
ax.set_title(f'Carrier noise vs N (S = {S_fixed:.1e} m$^{{-3}}$)')
ax.legend()
ax.grid(True, alpha=0.3)

# sigma_S vs N
ax = axes1[0, 1]
ax.plot(N_grid / laser.N_tr, learned_sigma_S_vs_N, 'b-', lw=2, label='Learned')
ax.plot(N_grid / laser.N_tr, analytical_sigma_S_vs_N, 'r--', lw=2, label='Analytical')
ax.set_xlabel('$N / N_{tr}$')
ax.set_ylabel('$\\sigma_E$ (arb. units)')
ax.set_title(f'Field noise vs N (S = {S_fixed:.1e} m$^{{-3}}$)')
ax.legend()
ax.grid(True, alpha=0.3)

# sigma_N vs S
ax = axes1[1, 0]
ax.semilogx(S_grid, learned_sigma_N_vs_S, 'b-', lw=2, label='Learned')
ax.semilogx(S_grid, analytical_sigma_N_vs_S, 'r--', lw=2, label='Analytical')
ax.set_xlabel('$S$ (m$^{-3}$)')
ax.set_ylabel('$\\sigma_N$ (arb. units)')
ax.set_title(f'Carrier noise vs S (N = {N_fixed/laser.N_tr:.1f}$N_{{tr}}$)')
ax.legend()
ax.grid(True, alpha=0.3)

# sigma_S vs S
ax = axes1[1, 1]
ax.semilogx(S_grid, learned_sigma_S_vs_S, 'b-', lw=2, label='Learned')
ax.semilogx(S_grid, analytical_sigma_S_vs_S, 'r--', lw=2, label='Analytical')
ax.set_xlabel('$S$ (m$^{-3}$)')
ax.set_ylabel('$\\sigma_E$ (arb. units)')
ax.set_title(f'Field noise vs S (N = {N_fixed/laser.N_tr:.1f}$N_{{tr}}$)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_fig(fig1, os.path.join(OUT_DIR, 'noise_amplitudes.png'))


# ── Figure 2: Statistics comparison ──────────────────────────────────────────

fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
fig2.suptitle('Ground Truth (Langevin) vs Neural SDE: Pulse Statistics', fontsize=13)

# Peak power histogram
ax = axes2[0]
gt_P_mW = gt_peak_P * 1e3
nsde_P_mW = nsde_peak_P * 1e3
bins_P = np.linspace(
    min(gt_P_mW.min(), nsde_P_mW.min()),
    max(gt_P_mW.max(), nsde_P_mW.max()), 80)
ax.hist(gt_P_mW, bins=bins_P, density=True, alpha=0.6, color='blue',
        label=f'Langevin (mean={np.mean(gt_P_mW):.2f} mW)')
ax.hist(nsde_P_mW, bins=bins_P, density=True, alpha=0.6, color='orange',
        label=f'Neural SDE (mean={np.mean(nsde_P_mW):.2f} mW)')
ax.set_xlabel('Peak Power (mW)')
ax.set_ylabel('Probability Density')
ax.set_title('Peak Power Distribution')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Phase difference histogram
ax = axes2[1]
bins_phi = np.linspace(-np.pi, np.pi, 80)
ax.hist(gt_dphi, bins=bins_phi, density=True, alpha=0.6, color='blue',
        label=f'Langevin (r1={gt_r1:.4f})')
nsde_r1 = np.abs(np.mean(np.exp(1j * nsde_dphi))) if len(nsde_dphi) > 0 else 0
ax.hist(nsde_dphi, bins=bins_phi, density=True, alpha=0.6, color='orange',
        label=f'Neural SDE (r1={nsde_r1:.4f})')
ax.set_xlabel('$\\Delta\\phi$ (rad)')
ax.set_ylabel('Probability Density')
ax.set_title('Phase Difference Distribution')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Timing jitter comparison (peak position)
ax = axes2[2]
gt_timing_ps = gt_peak_k * dt_actual * 1e12
nsde_timing_ps = nsde_peak_k.flatten().numpy() * dt_actual * 1e12
bins_t = np.linspace(
    min(gt_timing_ps.min(), nsde_timing_ps.min()),
    max(gt_timing_ps.max(), nsde_timing_ps.max()), 80)
ax.hist(gt_timing_ps, bins=bins_t, density=True, alpha=0.6, color='blue',
        label=f'Langevin ($\\sigma$={np.std(gt_timing_ps):.1f} ps)')
ax.hist(nsde_timing_ps, bins=bins_t, density=True, alpha=0.6, color='orange',
        label=f'Neural SDE ($\\sigma$={np.std(nsde_timing_ps):.1f} ps)')
ax.set_xlabel('Peak Position (ps)')
ax.set_ylabel('Probability Density')
ax.set_title('Timing Distribution')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_fig(fig2, os.path.join(OUT_DIR, 'statistics_comparison.png'))


# ── Figure 3: Training loss ──────────────────────────────────────────────────

fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
fig3.suptitle('Neural SDE Training: Moment-Matching Loss', fontsize=13)

epochs = np.arange(1, N_EPOCHS + 1)

# Total loss
ax = axes3[0]
ax.semilogy(epochs, loss_history['total'], 'k-', lw=1.5, label='Total')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Total Loss')
ax.grid(True, alpha=0.3)
ax.legend()

# Individual components
ax = axes3[1]
ax.semilogy(epochs, loss_history['mean_S'], '-', lw=1.2, label='$\\langle S \\rangle$')
ax.semilogy(epochs, loss_history['std_S'], '-', lw=1.2, label='$\\sigma_S$')
ax.semilogy(epochs, loss_history['r1'], '-', lw=1.2, label='$r_1$')
ax.semilogy(epochs, loss_history['jitter'], '-', lw=1.2, label='Jitter')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss Component')
ax.set_title('Individual Loss Components')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_fig(fig3, os.path.join(OUT_DIR, 'training_loss.png'))


# ── Summary ──────────────────────────────────────────────────────────────────

# Compare learned vs analytical at operating point
N_op = laser.N_tr * 1.5  # approximate operating point
R_sp_op = laser.A * N_op + laser.B * N_op**2 + laser.C * N_op**3
R_sp_mode_op = laser.beta_sp * laser.B * N_op**2
analytical_sigma_N_op = np.sqrt(2 * R_sp_op)
analytical_sigma_S_op = np.sqrt(R_sp_mode_op)

with torch.no_grad():
    N_norm_op = torch.tensor(N_op / laser.N_tr, dtype=torch.float64)
    S_norm_op = torch.tensor(gt_mean_S / 1e21, dtype=torch.float64)
    t_norm_op = torch.tensor(0.5, dtype=torch.float64)
    sigma_op = model.noise_net(N_norm_op.unsqueeze(0), S_norm_op.unsqueeze(0),
                                t_norm_op.unsqueeze(0))
    learned_sigma_N_op = sigma_op[0, 0].item() * 1e12
    learned_sigma_S_op = sigma_op[0, 1].item() * 1e5

print("\n" + "=" * 70)
print("  Neural ODE Noise Model Summary")
print("=" * 70)
print(f"  Ground truth: {N_PULSES_GT} pulses at {F_REP*1e-9:.0f} GHz ({t_gt:.1f}s)")
print(f"  Training:     {N_EPOCHS} epochs ({t_train:.1f}s)")
print(f"  Final loss:   {loss_history['total'][-1]:.4f}")
print()
print(f"  Statistics comparison (ground truth vs neural SDE):")
print(f"    mean(peak_S):  {gt_mean_S:.3e} vs {nsde_peak_S_flat.mean():.3e}")
print(f"    std(peak_S):   {gt_std_S:.3e} vs {nsde_peak_S_flat.std():.3e}")
print(f"    r1:            {gt_r1:.4f} vs {nsde_r1:.4f}")
print(f"    jitter:        {gt_jitter*1e12:.2f} ps vs "
      f"{np.std(nsde_peak_k.flatten().numpy() * dt_actual)*1e12:.2f} ps")
print()
print(f"  Noise amplitude comparison at N = 1.5*N_tr:")
print(f"    sigma_N:  learned={learned_sigma_N_op:.3e}, "
      f"analytical={analytical_sigma_N_op:.3e} "
      f"(ratio={learned_sigma_N_op/analytical_sigma_N_op:.2f})")
print(f"    sigma_E:  learned={learned_sigma_S_op:.3e}, "
      f"analytical={analytical_sigma_S_op:.3e} "
      f"(ratio={learned_sigma_S_op/analytical_sigma_S_op:.2f})")
print()
total_time = time.time() - t0
print(f"  Total runtime: {total_time:.0f}s ({total_time/60:.1f} min)")
print("=" * 70)
