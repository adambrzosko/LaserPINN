"""
Run the Part 1 frequency sweep at three SLD acceptance bandwidths:
  0.6 nm  (cavity resonance width, Δν_cav ≈ 77 GHz at λ=1547 nm)
  3.0 nm  (intermediate)
  8.0 nm  (current default — broadband carrier-noise channel)

Saves per-bandwidth scalar JSON files and a comparison figure.
"""

import sys, os, time as _time, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.dfb_laser import make_laser, q, h, c
from core.million_pulse_comparison import simulate_pulses_waveform
from gsdfb import compute_r1, setup_plotting, save_fig
from gsdfb.analysis import (
    phase_randomisation_quality, absolute_jitter, amzi_outputs,
)
from gsdfb.plotting import img_dir

# ── Drive parameters (same as paper_10ghz_simulation.py) ────────────
I_DC = 45.5e-3
V_RF_AMP = 3.7
Z_MATCH = 50.0
I_RF = V_RF_AMP / Z_MATCH
DT_TARGET = 0.5e-12


def make_paper_laser():
    return make_laser('dfb', lambda0=1547e-9, L=150e-6)


def build_sine_waveform(pts_period, dt, f_rep, I_dc, I_rf):
    t = np.arange(pts_period) * dt
    waveform = I_dc + I_rf * np.sin(2.0 * np.pi * f_rep * t)
    return np.maximum(waveform, 0.0).astype(np.float64)


def sld_power_to_sinj(P_sld_mW, laser, acceptance_bw_nm, sld_bw_nm=33.0):
    spectral_frac = acceptance_bw_nm / sld_bw_nm
    coupling_loss = 0.5
    total_coupling = spectral_frac * coupling_loss
    P_coupled_W = total_coupling * P_sld_mW * 1e-3
    nu0 = c / laser.lambda0
    S_inj = P_coupled_W * laser.tau_p / (h * nu0 * laser.V)
    return S_inj


def run_config(laser, f_rep, S_inj, n_pulses, n_discard=200, seed=42):
    T_rep = 1.0 / f_rep
    pts = max(int(round(T_rep / DT_TARGET)), 100)
    dt = T_rep / pts
    waveform = build_sine_waveform(pts, dt, f_rep, I_DC, I_RF)

    pk_phi, pk_S, samp_S, pk_k = simulate_pulses_waveform(
        n_pulses + n_discard, n_discard, pts, dt,
        waveform,
        laser.V, laser.Gamma, laser.v_g, laser.a, laser.N_tr, laser.epsilon,
        laser.A, laser.B, laser.C, laser.tau_p, laser.beta_sp, laser.alpha_H, q,
        S_inj, seed)

    r1, dphi = compute_r1(pk_phi)
    pq = phase_randomisation_quality(pk_phi)
    sigma_t, t_peak = absolute_jitter(pk_k, dt)
    pk_P = laser.output_power(np.maximum(pk_S, 0))
    mean_P = float(np.mean(pk_P))
    cv_P = float(np.std(pk_P) / mean_P) if mean_P > 0 else 0.0
    I_A, I_B, eta = amzi_outputs(pk_P, pk_phi)

    max_lag = min(50, n_pulses // 10)
    autocorr = np.zeros(max_lag)
    eta_ac = np.zeros(max_lag)
    eta_centered = eta - np.mean(eta)
    eta_var = float(np.var(eta))
    for k in range(max_lag):
        if k == 0:
            autocorr[k] = 1.0
            eta_ac[k] = 1.0
        else:
            autocorr[k] = float(np.abs(
                np.mean(np.exp(1j * (pk_phi[k:] - pk_phi[:-k])))))
            if eta_var > 0:
                eta_ac[k] = float(
                    np.mean(eta_centered[k:] * eta_centered[:-k])) / eta_var

    return dict(
        r1=r1, dphi=dphi, pq=pq,
        sigma_t=sigma_t, mean_P=mean_P, cv_P=cv_P,
        pk_phi=pk_phi, pk_P=pk_P, eta=eta,
        autocorr=autocorr, eta_autocorr=eta_ac,
        kl=pq['kl'], ks_stat=pq['ks_stat'],
    )


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    setup_plotting()

    laser = make_paper_laser()
    out = img_dir('acceptance_bw_comparison_v2')
    data_dir = os.path.join(out, 'data')
    os.makedirs(data_dir, exist_ok=True)

    N_PULSES = 1_000_000
    P_sld_mW = 19.0
    freqs = [1e9, 2e9, 5e9, 8e9, 10e9]
    freq_labels = ['1 GHz', '2 GHz', '5 GHz', '8 GHz', '10 GHz']
    bw_list = [0.6, 3.0, 8.0]

    # JIT warmup
    print("Compiling JIT...", end="", flush=True)
    t0 = _time.time()
    _wf = build_sine_waveform(50, 1e-12, 10e9, I_DC, I_RF)
    simulate_pulses_waveform(
        100, 10, 50, 1e-12, _wf,
        laser.V, laser.Gamma, laser.v_g, laser.a, laser.N_tr, laser.epsilon,
        laser.A, laser.B, laser.C, laser.tau_p, laser.beta_sp, laser.alpha_H, q,
        0.0, 0)
    del _wf
    print(f" {_time.time()-t0:.1f}s")

    # ── Run free-running once ───────────────────────────────────────
    free_results = {}
    print(f"\nFree-running ({N_PULSES//1000}k pulses)")
    for f_rep, flabel in zip(freqs, freq_labels):
        seed = hash(f'{flabel}_free') % 100000
        print(f"  {flabel:8s} ...", end="", flush=True)
        t0 = _time.time()
        r = run_config(laser, f_rep, 0.0, N_PULSES, seed=seed)
        print(f" {_time.time()-t0:.1f}s  r1={r['r1']:.4f}")
        free_results[flabel] = r

    # ── Run SLD-injected at each bandwidth ──────────────────────────
    sld_results = {}  # keyed by (bw_nm, flabel)
    for bw_nm in bw_list:
        S_inj = sld_power_to_sinj(P_sld_mW, laser, bw_nm)
        print(f"\nSLD-injected, acceptance BW = {bw_nm} nm, "
              f"S_inj = {S_inj:.2e} m^-3 ({N_PULSES//1000}k pulses)")
        for f_rep, flabel in zip(freqs, freq_labels):
            seed = hash(f'{flabel}_sld_{bw_nm}') % 100000
            print(f"  {flabel:8s} ...", end="", flush=True)
            t0 = _time.time()
            r = run_config(laser, f_rep, S_inj, N_PULSES, seed=seed)
            print(f" {_time.time()-t0:.1f}s  r1={r['r1']:.4f}  "
                  f"KL={r['kl']:.4f}  jitter={r['sigma_t']*1e12:.1f}ps")
            sld_results[(bw_nm, flabel)] = r

    # ── Save scalar data per bandwidth ──────────────────────────────
    for bw_nm in bw_list:
        scalars = {}
        for flabel in freq_labels:
            rf = free_results[flabel]
            scalars[f'{flabel}_free'] = dict(
                r1=float(rf['r1']), kl=float(rf['kl']),
                sigma_t_ps=float(rf['sigma_t'] * 1e12),
                cv_P=float(rf['cv_P']),
            )
            rs = sld_results[(bw_nm, flabel)]
            scalars[f'{flabel}_sld'] = dict(
                r1=float(rs['r1']), kl=float(rs['kl']),
                sigma_t_ps=float(rs['sigma_t'] * 1e12),
                cv_P=float(rs['cv_P']),
            )
        fname = f'{data_dir}/scalars_bw{bw_nm:.1f}nm.json'
        with open(fname, 'w') as f:
            json.dump(scalars, f, indent=2)
        print(f"Saved {fname}")

    # ── Save raw arrays (eta, dphi, autocorrelation) ──────────────
    # Free-running (shared across all BWs)
    free_arrays = {}
    for flabel in freq_labels:
        rf = free_results[flabel]
        free_arrays[f'{flabel}_eta'] = rf['eta']
        free_arrays[f'{flabel}_dphi'] = rf['dphi']
        free_arrays[f'{flabel}_autocorr'] = rf['autocorr']
        free_arrays[f'{flabel}_eta_autocorr'] = rf['eta_autocorr']
    np.savez_compressed(f'{data_dir}/free_running.npz',
                        freq_labels=np.array(freq_labels),
                        N_PULSES=N_PULSES,
                        **free_arrays)
    print(f"Saved {data_dir}/free_running.npz")

    # Per-bandwidth SLD-injected arrays
    for bw_nm in bw_list:
        bw_arrays = {}
        for flabel in freq_labels:
            rs = sld_results[(bw_nm, flabel)]
            bw_arrays[f'{flabel}_eta'] = rs['eta']
            bw_arrays[f'{flabel}_dphi'] = rs['dphi']
            bw_arrays[f'{flabel}_autocorr'] = rs['autocorr']
            bw_arrays[f'{flabel}_eta_autocorr'] = rs['eta_autocorr']
        np.savez_compressed(
            f'{data_dir}/sld_bw{bw_nm:.1f}nm.npz',
            freq_labels=np.array(freq_labels),
            N_PULSES=N_PULSES,
            bw_nm=bw_nm,
            P_sld_mW=P_sld_mW,
            **bw_arrays)
        print(f"Saved {data_dir}/sld_bw{bw_nm:.1f}nm.npz")

    # ── Summary table ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  r1 summary (SLD-injected, {P_sld_mW:.0f} mW)")
    print(f"  {'Freq':>8s}", end="")
    for bw in bw_list:
        print(f"  {'bw='+str(bw)+'nm':>12s}", end="")
    print()
    for flabel in freq_labels:
        print(f"  {flabel:>8s}", end="")
        for bw in bw_list:
            r1 = sld_results[(bw, flabel)]['r1']
            print(f"  {r1:12.4f}", end="")
        print()

    # ── Comparison figure: η histograms ─────────────────────────────
    n_freq = len(freqs)
    n_bw = len(bw_list)
    fig, axes = plt.subplots(n_bw, n_freq, figsize=(4 * n_freq, 3.5 * n_bw))
    fig.suptitle(
        f'SLD-injected AMZI $\\eta$ distribution at {P_sld_mW:.0f} mW\n'
        f'Acceptance bandwidth comparison',
        fontsize=13, y=0.98)

    eta_th = np.linspace(0.001, 0.999, 500)
    arcsine_pdf = 1.0 / (np.pi * np.sqrt(eta_th * (1.0 - eta_th)))

    for row, bw_nm in enumerate(bw_list):
        for col, flabel in enumerate(freq_labels):
            ax = axes[row, col]
            r = sld_results[(bw_nm, flabel)]
            ax.hist(r['eta'], bins=200, density=True, alpha=0.7,
                    color=f'C{row}',
                    label=f'r$_1$={r["r1"]:.3f}')
            ax.plot(eta_th, arcsine_pdf, 'k--', lw=1, alpha=0.5)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 12)
            if row == 0:
                ax.set_title(flabel, fontsize=11)
            if col == 0:
                ax.set_ylabel(f'BW = {bw_nm} nm')
            if row == n_bw - 1:
                ax.set_xlabel('$\\eta$')
            ax.legend(fontsize=7, loc='upper center')
            ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    figpath = f'{out}/acceptance_bw_comparison.png'
    fig.savefig(figpath, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {figpath}")
    plt.close(fig)

    # ── Phase distribution figure: Δφ histograms ──────────────────
    fig2, axes2 = plt.subplots(n_bw, n_freq, figsize=(4 * n_freq, 3.5 * n_bw))
    fig2.suptitle(
        f'SLD-injected phase difference $\\Delta\\varphi$ distribution at {P_sld_mW:.0f} mW\n'
        f'Acceptance bandwidth comparison',
        fontsize=13, y=0.98)

    for row, bw_nm in enumerate(bw_list):
        for col, flabel in enumerate(freq_labels):
            ax = axes2[row, col]
            r = sld_results[(bw_nm, flabel)]
            dphi_wrapped = np.mod(r['dphi'], 2 * np.pi)
            ax.hist(dphi_wrapped, bins=200, density=True, alpha=0.7,
                    color=f'C{row}',
                    label=f'r$_1$={r["r1"]:.3f}')
            ax.axhline(1.0 / (2 * np.pi), color='k', ls='--', lw=1, alpha=0.5,
                       label='Uniform')
            ax.set_xlim(0, 2 * np.pi)
            ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
            ax.set_xticklabels(['0', '$\\pi/2$', '$\\pi$', '$3\\pi/2$', '$2\\pi$'])
            if row == 0:
                ax.set_title(flabel, fontsize=11)
            if col == 0:
                ax.set_ylabel(f'BW = {bw_nm} nm')
            if row == n_bw - 1:
                ax.set_xlabel('$\\Delta\\varphi$ (rad)')
            ax.legend(fontsize=7, loc='upper right')
            ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    figpath2 = f'{out}/acceptance_bw_phase_distribution.png'
    fig2.savefig(figpath2, dpi=150, bbox_inches='tight')
    print(f"Figure saved: {figpath2}")
    plt.close(fig2)
