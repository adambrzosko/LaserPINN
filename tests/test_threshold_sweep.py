"""Sweep eta_coupling at 10 GHz to find coherence destruction threshold.

Uses realistic SLD parameters with configurable acceptance bandwidth.
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

sld_result = solve_sld_steady_state(sld, 150e-3)
P_sld = sld_result['P_out']

print("=" * 65)
print("  Threshold Sweep: eta_coupling at 10 GHz")
print("=" * 65)
print(f"  P_sld = {P_sld*1e3:.2f} mW, sld BW = {sld.delta_nu*1e-12:.0f} THz")
print(f"  tau_p = {laser.tau_p*1e12:.1f} ps, T_off = {(1-gs.duty)*gs.T_rep*1e12:.0f} ps")

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


def run_sweep(acceptance_bw, etas, label):
    print(f"\n  --- {label} (acceptance BW = {acceptance_bw*1e-9:.0f} GHz) ---")
    spec_frac = min(acceptance_bw / sld.delta_nu, 1.0)
    print(f"  spectral fraction = {spec_frac:.4f}")

    results = []
    for eta in etas:
        inj = InjectionParams(eta_coupling=eta)
        inj.compute_derived(laser)
        S_inj, _ = sld_to_injection_field(sld, P_sld, laser, inj,
                                           acceptance_bandwidth=acceptance_bw)

        I_func = make_gain_switch_current(gs, I_th)
        t_total = gs.t_sim
        n_pts = gs.n_periods * PTS
        t_eval = np.linspace(0, t_total, n_pts)

        sol = solve_transient_injection_stochastic(
            laser, I_func, inj, S_inj,
            sld_tau_coh=sld.tau_coh,
            t_span=[0, t_total], t_eval=t_eval, seed=42,
        )

        idx_start = gs.n_discard * PTS
        S_data = sol.y[1, idx_start:]
        phi_data = sol.y[2, idx_start:]
        phases = []
        for n in range(gs.n_steady):
            s_idx = n * PTS
            e_idx = (n + 1) * PTS
            if e_idx > len(S_data):
                break
            peak = s_idx + np.argmax(S_data[s_idx:e_idx])
            phases.append(phi_data[peak])
        phases = np.array(phases)
        dphi = np.angle(np.exp(1j * np.diff(phases)))
        std_dphi = np.std(dphi)

        results.append({'eta': eta, 'S_inj': S_inj, 'std_dphi': std_dphi})
        print(f"    eta={eta:.3f}  S_inj={S_inj:.2e}  std(dphi)={std_dphi:.4f}")

    return results


cold_cavity_bw = laser.v_g * (laser.alpha_i + laser.alpha_m) / (2 * np.pi)
etas = np.array([0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 1.0])

bw_configs = [
    (cold_cavity_bw, 'Cold-cavity linewidth'),
    (500e9, '500 GHz'),
    (2e12, '2 THz (gain BW)'),
]

all_results = {}
for bw, label in bw_configs:
    all_results[label] = run_sweep(bw, etas, label)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

colors = ['C0', 'C1', 'C2']
for (label, res), color in zip(all_results.items(), colors):
    etas_plot = [r['eta'] for r in res]
    stds_plot = [r['std_dphi'] for r in res]
    sinjs = [r['S_inj'] for r in res]

    axes[0].plot(etas_plot, stds_plot, 'o-', color=color, lw=2, ms=5,
                 label=label)
    axes[1].semilogx(sinjs, stds_plot, 'o-', color=color, lw=2, ms=5,
                     label=label)

for ax in axes:
    ax.axhline(std_free, color='gray', ls='--', lw=1.5,
               label=f'Free-running ({std_free:.2f} rad)')
    ax.axhline(np.pi / np.sqrt(3), color='red', ls=':', lw=1,
               label=f'Uniform random ({np.pi/np.sqrt(3):.2f} rad)')
    ax.set_ylabel('Pulse-to-pulse phase std (rad)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 2.5)

axes[0].set_xlabel('Coupling efficiency η')
axes[1].set_xlabel('S_inj (m$^{-3}$)')

fig.suptitle(
    f'Coherence Threshold vs Coupling — 10 GHz Gain Switching\n'
    f'P_sld = {P_sld*1e3:.1f} mW, I_bias = {gs.I_bias_factor}×I_th, '
    f'I_peak = {gs.I_peak_factor}×I_th',
    fontsize=12,
)
plt.tight_layout()
fig.savefig('images/coherence_threshold/coherence_threshold_vs_eta.png', dpi=150)
print(f"\n  Saved: images/coherence_threshold/coherence_threshold_vs_eta.png")
plt.close('all')
