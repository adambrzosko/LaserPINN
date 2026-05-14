"""
Timing jitter analysis of gain-switched pulse trains, 1-10 GHz.

Extracts the peak arrival time within each period from 1M-pulse
simulations.  Computes:
  - Absolute jitter  sigma_t  (ps)
  - Period jitter  sigma_T  (variation of peak-to-peak interval)
  - Allan deviation of the timing
  - Correlation between timing jitter and phase / amplitude jitter

Key prediction: SLD injection REDUCES timing jitter (more consistent
seed photon number → less turn-on delay variation) while INCREASING
phase jitter.  This creates an engineering trade-off.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time as _time

from dfb_laser import DFBLaserParams, q, h, c
from sld_injection import (
    SLDParams, InjectionParams,
    solve_sld_steady_state, sld_to_injection_field,
)
from million_pulse_comparison import simulate_pulses


# ── Jitter metrics ───────────────────────────────────────────────────────────

def absolute_jitter(peak_k, dt):
    """RMS timing jitter: std of peak arrival time within period."""
    t_peak = peak_k.astype(np.float64) * dt
    return np.std(t_peak), t_peak


def period_jitter(peak_k, dt, T_rep):
    """Period jitter: std of peak-to-peak interval deviation from T_rep.

    The true peak-to-peak interval is T_rep + (t_peak[n+1] - t_peak[n]).
    Period jitter = std(t_peak[n+1] - t_peak[n]).
    """
    t_peak = peak_k.astype(np.float64) * dt
    dt_peak = np.diff(t_peak)
    return np.std(dt_peak), dt_peak


def allan_deviation(peak_k, dt, T_rep, max_m=1000):
    """Overlapping Allan deviation of fractional timing.

    y_n = (peak-to-peak interval - T_rep) / T_rep
    """
    t_peak = peak_k.astype(np.float64) * dt
    y = np.diff(t_peak) / T_rep

    n = len(y)
    ms = np.unique(np.geomspace(1, min(n // 4, max_m), 40).astype(int))
    ms = ms[ms >= 1]

    adev = np.zeros(len(ms))
    for i, m in enumerate(ms):
        # Overlapping Allan variance
        ybar = np.convolve(y, np.ones(m) / m, mode='valid')
        diff = np.diff(ybar)
        adev[i] = np.sqrt(0.5 * np.mean(diff**2))

    tau = ms * T_rep
    return tau, adev


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 70)
    print("  Timing Jitter Analysis — 1M Pulses, 1-10 GHz")
    print("=" * 70)

    laser = DFBLaserParams()
    I_th = laser.threshold_current()
    sld = SLDParams()

    sld_result = solve_sld_steady_state(sld, 150e-3)
    P_sld = sld_result['P_out']
    inj = InjectionParams(eta_coupling=0.10)
    inj.compute_derived(laser)
    S_INJ, _ = sld_to_injection_field(sld, P_sld, laser, inj,
                                       acceptance_bandwidth=500e9)

    N_PULSES = 1_000_000
    N_DISCARD = 1000
    DT = 1.0e-12
    DUTY = 0.30
    I_BIAS_FACTOR = 0.9
    I_PEAK_FACTOR = 5.0

    I_off = I_BIAS_FACTOR * I_th
    I_on = I_PEAK_FACTOR * I_th

    freqs = np.arange(1, 11) * 1e9
    f_ghz = freqs * 1e-9

    print(f"  S_inj = {S_INJ:.2e} m^-3,  {N_PULSES/1e6:.0f}M pulses per run")

    # Warmup
    print("\n  Compiling JIT solver...")
    t0 = _time.time()
    _w = simulate_pulses(
        100, 10, 100, 1e-12,
        I_off, I_on, 30e-12, 7e-12,
        laser.V, laser.Gamma, laser.v_g, laser.a, laser.N_tr, laser.epsilon,
        laser.A, laser.B, laser.C, laser.tau_p, laser.beta_sp, laser.alpha_H, q,
        0.0, 0)
    del _w
    print(f"  Compiled in {_time.time()-t0:.1f}s")

    data = {}
    total_t0 = _time.time()

    for f_rep in freqs:
        T_rep = 1.0 / f_rep
        t_on = DUTY * T_rep
        pts = max(int(round(T_rep / DT)), 50)
        dt = T_rep / pts
        t_rise = min(20e-12, t_on / 4.0)

        for case, s_inj in [('free', 0.0), ('sld', S_INJ)]:
            key = f"{f_rep*1e-9:.0f}GHz_{case}"
            seed = 42 + int(f_rep * 1e-9) * 100 + (0 if case == 'free' else 1)

            print(f"  {key:14s} ...", end="", flush=True)
            t0 = _time.time()

            phi, pk_S, sm_S, pk_k = simulate_pulses(
                N_PULSES + N_DISCARD, N_DISCARD, pts, dt,
                I_off, I_on, t_on, t_rise,
                laser.V, laser.Gamma, laser.v_g, laser.a,
                laser.N_tr, laser.epsilon,
                laser.A, laser.B, laser.C,
                laser.tau_p, laser.beta_sp, laser.alpha_H, q,
                s_inj, seed)

            elapsed = _time.time() - t0

            pk_P = laser.output_power(np.maximum(pk_S, 0))

            # Jitter metrics
            sig_t, t_peak = absolute_jitter(pk_k, dt)
            sig_T, dt_peak = period_jitter(pk_k, dt, T_rep)

            # Phase metrics
            dphi = np.angle(np.exp(1j * np.diff(phi)))
            r1 = np.abs(np.mean(np.exp(1j * dphi)))
            sig_phi = np.std(dphi)

            # Allan deviation
            tau_ad, adev = allan_deviation(pk_k, dt, T_rep)

            data[key] = dict(
                f_rep=f_rep, T_rep=T_rep, dt=dt, case=case,
                phi=phi, pk_P=pk_P, pk_k=pk_k,
                t_peak=t_peak, dt_peak=dt_peak,
                sig_t=sig_t, sig_T=sig_T,
                r1=r1, sig_phi=sig_phi,
                tau_ad=tau_ad, adev=adev,
            )

            print(f" {elapsed:5.1f}s  sig_t={sig_t*1e12:5.2f}ps  "
                  f"sig_T={sig_T*1e12:5.2f}ps  "
                  f"sig_phi={sig_phi:.3f}rad  r1={r1:.4f}")

    total_elapsed = _time.time() - total_t0
    print(f"\n  Total: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")

    # ── Figure 1: Jitter vs frequency — main result ──────────────────────

    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))
    fig1.suptitle(
        'Timing Jitter vs Repetition Rate — 1M Pulses\n'
        f'I$_{{bias}}$={I_BIAS_FACTOR}$\\times$I$_{{th}}$, '
        f'I$_{{peak}}$={I_PEAK_FACTOR}$\\times$I$_{{th}}$, '
        f'S$_{{inj}}$={S_INJ:.1e} m$^{{-3}}$',
        fontsize=12)

    for case, color, marker, label in [
        ('free', 'C0', 'o', 'Free-running'),
        ('sld', 'C3', 's', 'SLD-injected'),
    ]:
        sig_t_vals = [data[f"{f:.0f}GHz_{case}"]['sig_t'] * 1e12
                      for f in f_ghz]
        sig_T_vals = [data[f"{f:.0f}GHz_{case}"]['sig_T'] * 1e12
                      for f in f_ghz]
        sig_phi_vals = [data[f"{f:.0f}GHz_{case}"]['sig_phi']
                        for f in f_ghz]

        axes1[0].plot(f_ghz, sig_t_vals, f'{marker}-', color=color,
                      lw=2, ms=7, label=label)
        axes1[1].plot(f_ghz, sig_T_vals, f'{marker}-', color=color,
                      lw=2, ms=7, label=label)
        axes1[2].plot(f_ghz, sig_phi_vals, f'{marker}-', color=color,
                      lw=2, ms=7, label=label)

    axes1[0].set_ylabel('Absolute jitter $\\sigma_t$ (ps)')
    axes1[0].set_title('Peak arrival jitter')
    axes1[1].set_ylabel('Period jitter $\\sigma_T$ (ps)')
    axes1[1].set_title('Peak-to-peak interval jitter')
    axes1[2].set_ylabel('Phase jitter $\\sigma_{\\Delta\\phi}$ (rad)')
    axes1[2].set_title('Phase jitter')
    axes1[2].axhline(np.pi / np.sqrt(3), color='gray', ls=':', alpha=0.5)

    for ax in axes1:
        ax.set_xlabel('Repetition rate (GHz)')
        ax.set_xticks(f_ghz)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig1.savefig('images/timing_jitter/jitter_vs_freq.png', dpi=150)
    print("  Saved: images/timing_jitter/jitter_vs_freq.png")

    # ── Figure 2: Peak arrival distributions ─────────────────────────────

    sel = [1, 3, 5, 7, 10]
    fig2, axes2 = plt.subplots(2, len(sel), figsize=(22, 8))
    fig2.suptitle(
        'Peak Arrival Time Distribution Within Period — 1M Pulses',
        fontsize=13)

    for col, fg in enumerate(sel):
        for row, (case, color, label) in enumerate([
            ('free', 'C0', 'Free-running'),
            ('sld', 'C3', 'SLD-injected'),
        ]):
            key = f"{fg}GHz_{case}"
            d = data[key]
            t_ps = d['t_peak'] * 1e12
            t_on_ps = DUTY * d['T_rep'] * 1e12

            ax = axes2[row, col]
            ax.hist(t_ps, bins=200, density=True, color=color,
                    alpha=0.7, edgecolor='none')
            ax.axvline(np.mean(t_ps), color='k', ls='--', lw=1, alpha=0.7)
            ax.axvline(t_on_ps, color='gray', ls=':', lw=1, alpha=0.5,
                       label=f'$t_{{on}}$ = {t_on_ps:.0f} ps')
            ax.set_title(
                f'{fg} GHz — {label}\n'
                f'$\\sigma_t$ = {d["sig_t"]*1e12:.2f} ps, '
                f'$\\langle t \\rangle$ = {np.mean(t_ps):.1f} ps',
                fontsize=10)
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel('Probability density')
            ax.legend(fontsize=8)

        axes2[1, col].set_xlabel('Peak time (ps)')

    plt.tight_layout()
    fig2.savefig('images/timing_jitter/arrival_distributions.png', dpi=150)
    print("  Saved: images/timing_jitter/arrival_distributions.png")

    # ── Figure 3: Trade-off — timing jitter vs phase jitter ──────────────

    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
    fig3.suptitle(
        'Timing–Phase Jitter Trade-off Under SLD Injection',
        fontsize=13)

    # Panel 1: parametric curve (sig_t vs sig_phi)
    ax = axes3[0]
    for case, color, marker, label in [
        ('free', 'C0', 'o', 'Free-running'),
        ('sld', 'C3', 's', 'SLD-injected'),
    ]:
        sig_t = [data[f"{f:.0f}GHz_{case}"]['sig_t'] * 1e12
                 for f in f_ghz]
        sig_p = [data[f"{f:.0f}GHz_{case}"]['sig_phi']
                 for f in f_ghz]
        ax.plot(sig_p, sig_t, f'{marker}-', color=color, lw=2, ms=7,
                label=label)
        for i, fg in enumerate(f_ghz):
            if fg in [1, 3, 5, 10]:
                ax.annotate(f'{fg:.0f}',
                            (sig_p[i], sig_t[i]),
                            textcoords="offset points",
                            xytext=(6, 4), fontsize=8, color=color)

    ax.set_xlabel('Phase jitter $\\sigma_{\\Delta\\phi}$ (rad)')
    ax.set_ylabel('Timing jitter $\\sigma_t$ (ps)')
    ax.set_title('Parametric: each point is one $f_{rep}$ (GHz label)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: jitter ratio (SLD / free)
    ax = axes3[1]
    ratio_t = []
    ratio_phi = []
    for fg in f_ghz:
        st_f = data[f"{fg:.0f}GHz_free"]['sig_t']
        st_s = data[f"{fg:.0f}GHz_sld"]['sig_t']
        sp_f = data[f"{fg:.0f}GHz_free"]['sig_phi']
        sp_s = data[f"{fg:.0f}GHz_sld"]['sig_phi']
        ratio_t.append(st_s / max(st_f, 1e-30))
        ratio_phi.append(sp_s / max(sp_f, 1e-30))

    ax.plot(f_ghz, ratio_t, 'o-', color='C4', lw=2, ms=7,
            label='Timing jitter ratio')
    ax.plot(f_ghz, ratio_phi, 's-', color='C1', lw=2, ms=7,
            label='Phase jitter ratio')
    ax.axhline(1.0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Repetition rate (GHz)')
    ax.set_ylabel('Jitter ratio (SLD / free-running)')
    ax.set_xticks(f_ghz)
    ax.set_title('SLD impact: < 1 = reduction, > 1 = increase')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig3.savefig('images/timing_jitter/jitter_tradeoff.png', dpi=150)
    print("  Saved: images/timing_jitter/jitter_tradeoff.png")

    # ── Figure 4: Correlations at 10 GHz ─────────────────────────────────

    fig4, axes4 = plt.subplots(2, 3, figsize=(18, 10))
    fig4.suptitle(
        'Timing–Phase–Amplitude Correlations at 10 GHz — '
        '10k pulse sample',
        fontsize=13)

    N_SHOW = 10_000  # subsample for scatter visibility

    for row, (case, color, label) in enumerate([
        ('free', 'C0', 'Free-running'),
        ('sld', 'C3', 'SLD-injected'),
    ]):
        key = f"10GHz_{case}"
        d = data[key]

        dt_ps = np.diff(d['t_peak'][:N_SHOW + 1]) * 1e12
        dphi = np.angle(np.exp(1j * np.diff(d['phi'][:N_SHOW + 1])))
        dP_mW = np.diff(d['pk_P'][:N_SHOW + 1]) * 1e3
        t_ps = d['t_peak'][:N_SHOW] * 1e12

        # Scatter: dt vs dphi
        ax = axes4[row, 0]
        ax.scatter(dphi, dt_ps, s=0.2, alpha=0.3, color=color, rasterized=True)
        ax.set_xlabel('$\\Delta\\phi$ (rad)')
        ax.set_ylabel('$\\Delta t_{peak}$ (ps)')
        ax.set_title(f'{label}: timing vs phase change')
        rho = np.corrcoef(dphi, dt_ps)[0, 1]
        ax.text(0.05, 0.95, f'$\\rho$ = {rho:.3f}',
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', fc='white', alpha=0.8))
        ax.grid(True, alpha=0.3)

        # Scatter: dt vs dP
        ax = axes4[row, 1]
        ax.scatter(dP_mW, dt_ps, s=0.2, alpha=0.3, color=color, rasterized=True)
        ax.set_xlabel('$\\Delta P_{peak}$ (mW)')
        ax.set_ylabel('$\\Delta t_{peak}$ (ps)')
        ax.set_title(f'{label}: timing vs power change')
        rho2 = np.corrcoef(dP_mW, dt_ps)[0, 1]
        ax.text(0.05, 0.95, f'$\\rho$ = {rho2:.3f}',
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', fc='white', alpha=0.8))
        ax.grid(True, alpha=0.3)

        # Period jitter time series (first 200 pulses)
        ax = axes4[row, 2]
        ax.plot(t_ps[:200] - np.mean(t_ps), color=color, lw=0.8)
        ax.set_xlabel('Pulse number')
        ax.set_ylabel('$t_{peak} - \\langle t \\rangle$ (ps)')
        ax.set_title(f'{label}: timing deviation trace')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig4.savefig('images/timing_jitter/correlations_10GHz.png', dpi=150)
    print("  Saved: images/timing_jitter/correlations_10GHz.png")

    # ── Figure 5: Allan deviation ────────────────────────────────────────

    fig5, axes5 = plt.subplots(1, 2, figsize=(14, 6))
    fig5.suptitle(
        'Allan Deviation of Peak Timing — Selected Frequencies',
        fontsize=13)

    sel_ad = [1, 3, 5, 10]
    for ax, (case, label) in zip(axes5, [
        ('free', 'Free-running'), ('sld', 'SLD-injected')
    ]):
        for fg in sel_ad:
            key = f"{fg}GHz_{case}"
            d = data[key]
            tau_s = d['tau_ad']
            ax.loglog(tau_s * 1e9, d['adev'],
                      lw=1.5, label=f'{fg} GHz')

        ax.set_xlabel('Averaging time $\\tau$ (ns)')
        ax.set_ylabel('Allan deviation $\\sigma_y(\\tau)$')
        ax.set_title(label)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which='both')

        # Reference slope for white noise: tau^{-1/2}
        tau_ref = np.array([1e-9, 1e-6])
        ax.loglog(tau_ref * 1e9, 0.01 * np.sqrt(tau_ref[0] / tau_ref),
                  'k--', lw=0.8, alpha=0.4, label='$\\tau^{-1/2}$')

    plt.tight_layout()
    fig5.savefig('images/timing_jitter/allan_deviation.png', dpi=150)
    print("  Saved: images/timing_jitter/allan_deviation.png")

    # ── Summary table ────────────────────────────────────────────────────

    plt.close('all')

    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"  {'f_rep':>5s}  {'case':8s}  {'sig_t':>7s}  {'sig_T':>7s}  "
          f"{'sig_phi':>7s}  {'r1':>6s}  {'<t_pk>':>7s}")
    print(f"  {'(GHz)':>5s}  {'':8s}  {'(ps)':>7s}  {'(ps)':>7s}  "
          f"{'(rad)':>7s}  {'':>6s}  {'(ps)':>7s}")
    print("  " + "-" * 60)

    for fg in f_ghz:
        for case in ['free', 'sld']:
            key = f"{fg:.0f}GHz_{case}"
            d = data[key]
            print(f"  {fg:5.0f}  {case:8s}  "
                  f"{d['sig_t']*1e12:7.2f}  {d['sig_T']*1e12:7.2f}  "
                  f"{d['sig_phi']:7.3f}  {d['r1']:6.4f}  "
                  f"{np.mean(d['t_peak'])*1e12:7.1f}")

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)
