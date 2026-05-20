"""Test: does the complex field solver show coherence destruction threshold?

Sweeps S_inj directly at 10 GHz gain switching and measures pulse-to-pulse
phase standard deviation. A threshold should appear where std(dphi) jumps
from ~small (correlated) to ~pi/sqrt(3) (random).
"""
import numpy as np

from core.dfb_laser import DFBLaserParams, q, h, c
from gsdfb.plotting import setup_plotting

import matplotlib.pyplot as plt
setup_plotting()
from core.sld_injection import (
    SLDParams, InjectionParams,
    solve_sld_steady_state, sld_to_injection_field,
    solve_transient_injection_stochastic,
)
from core.gain_switched_interference import (
    GainSwitchParams, make_gain_switch_current, simulate_pulse_train,
)

laser = DFBLaserParams()
I_th = laser.threshold_current()
sld = SLDParams()

gs = GainSwitchParams(
    f_rep=10e9,
    duty=0.30,
    I_bias_factor=0.9,
    I_peak_factor=5.0,
    t_rise=20e-12,
    n_periods=60,
    n_discard=10,
)

PTS = 2000

print("=" * 65)
print("  Coherence Threshold Test — Complex Field Solver")
print("=" * 65)
print(f"  f_rep = {gs.f_rep*1e-9:.0f} GHz, tau_p = {laser.tau_p*1e12:.1f} ps")
print(f"  T_rep = {gs.T_rep*1e12:.0f} ps, T_off = {(1-gs.duty)*gs.T_rep*1e12:.0f} ps")
print(f"  Pulses for statistics: {gs.n_steady}")

R_sp_est = laser.beta_sp * laser.B * laser.N_tr**2
print(f"  R_sp_mode(N_tr) = {R_sp_est:.2e} m^-3 s^-1")


def extract_pulse_phase_std(sol, gs, pts=PTS):
    idx_start = gs.n_discard * pts
    S = sol.y[1, idx_start:]
    phi = sol.y[2, idx_start:]
    phases = []
    for n in range(gs.n_steady):
        s = n * pts
        e = (n + 1) * pts
        if e > len(S):
            break
        peak = s + np.argmax(S[s:e])
        phases.append(phi[peak])
    phases = np.array(phases)
    dphi = np.diff(phases)
    dphi = np.angle(np.exp(1j * dphi))
    return np.std(dphi), phases


# Free-running reference
print(f"\n  Free-running reference...")
data_free = simulate_pulse_train(laser, gs, pts_per_period=PTS, seed=42)
phases_free = []
for n in range(gs.n_steady):
    s = n * PTS
    e = (n + 1) * PTS
    if e > len(data_free['S']):
        break
    peak = s + np.argmax(data_free['S'][s:e])
    phases_free.append(data_free['phi'][peak])
phases_free = np.array(phases_free)
dphi_free = np.angle(np.exp(1j * np.diff(phases_free)))
std_free = np.std(dphi_free)
print(f"  std(dphi) = {std_free:.4f} rad")

# Also check S during off phase
S_min = []
for n in range(gs.n_steady):
    s = n * PTS
    on_end = s + int(gs.duty * PTS)
    off_end = (n + 1) * PTS
    if off_end > len(data_free['S']):
        break
    S_min.append(np.min(data_free['S'][on_end:off_end]))
print(f"  S_min (off phase) = {np.mean(S_min):.2e} m^-3")
print(f"  S_peak = {np.max(data_free['S']):.2e} m^-3")


# Sweep S_inj
S_inj_values = np.logspace(16, 23, 15)

print(f"\n  Sweeping S_inj ({len(S_inj_values)} values)...")
print(f"  {'S_inj':>12s}  {'R_SLD/R_sp':>10s}  {'std(dphi)':>10s}")
print(f"  {'----':>12s}  {'----------':>10s}  {'---------':>10s}")

results = []
for S_inj in S_inj_values:
    inj = InjectionParams(eta_coupling=0.1)
    inj.compute_derived(laser)

    I_func = make_gain_switch_current(gs, I_th)
    t_total = gs.t_sim
    n_pts = gs.n_periods * PTS
    t_eval = np.linspace(0, t_total, n_pts)

    sol = solve_transient_injection_stochastic(
        laser, I_func, inj, S_inj,
        sld_tau_coh=sld.tau_coh,
        t_span=[0, t_total], t_eval=t_eval, seed=42,
    )

    std_dphi, phases = extract_pulse_phase_std(sol, gs)
    R_SLD = S_inj / laser.tau_p
    ratio = R_SLD / R_sp_est

    results.append({'S_inj': S_inj, 'std_dphi': std_dphi, 'ratio': ratio})
    print(f"  {S_inj:12.2e}  {ratio:10.1f}  {std_dphi:10.4f}")


# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

S_vals = [r['S_inj'] for r in results]
std_vals = [r['std_dphi'] for r in results]
ratio_vals = [r['ratio'] for r in results]

for ax, x_vals, xlabel in [
    (axes[0], S_vals, 'Injected photon density S_inj (m$^{-3}$)'),
    (axes[1], ratio_vals, 'R$_{SLD}$ / R$_{sp,mode}$'),
]:
    ax.semilogx(x_vals, std_vals, 'ro-', lw=2, ms=6, label='SLD injected')
    ax.axhline(std_free, color='C0', ls='--', lw=1.5,
               label=f'Free-running: {std_free:.3f} rad')
    ax.axhline(np.pi / np.sqrt(3), color='gray', ls=':', lw=1,
               label=f'Uniform random: {np.pi/np.sqrt(3):.2f} rad')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Pulse-to-pulse phase std (rad)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 2.5)

fig.suptitle(
    f'Coherence Destruction Threshold — Complex Field Model\n'
    f'(10 GHz gain switching, {gs.n_steady} pulses, '
    f'I_bias={gs.I_bias_factor}×I_th, I_peak={gs.I_peak_factor}×I_th)',
    fontsize=12,
)
plt.tight_layout()
fig.savefig('images/coherence_threshold/coherence_threshold_test.png', dpi=150)
print(f"\n  Saved: images/coherence_threshold/coherence_threshold_test.png")
plt.close('all')
