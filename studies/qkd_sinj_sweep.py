"""
Sweep S_inj to find the minimum injection for secure QKD at each f_rep.

For BB84 decoy-state, security proofs require phase-randomised pulses:
r1 = |<exp(i*dphi)>| < 0.01.  This script finds the S_inj threshold
at 3, 5, and 10 GHz and quantifies the timing jitter cost.

Uses 500k pulses per point (r1 reliable to ~0.002).
"""
import numpy as np
import time as _time

from core.dfb_laser import DFBLaserParams, q, h, c
from core.sld_injection import (
    SLDParams, InjectionParams,
    solve_sld_steady_state, sld_to_injection_field,
)
from core.million_pulse_comparison import simulate_pulses
from gsdfb.analysis import compute_metrics as _compute_metrics


def compute_metrics(phi, pk_S, pk_k, dt, T_rep, laser):
    """Return dict of QKD-relevant metrics from one simulation run."""
    m = _compute_metrics(phi, pk_S, pk_k, dt, laser)
    # Provide 'cv' key for backward compatibility with the rest of this script
    m['cv'] = m['cv_P']
    return m


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator
    from gsdfb import setup_plotting, save_fig
    setup_plotting()

    print("=" * 70)
    print("  S_inj Sweep for QKD Security Threshold")
    print("=" * 70)

    laser = DFBLaserParams()
    I_th = laser.threshold_current()
    sld = SLDParams()

    N_PULSES = 500_000
    N_DISCARD = 500
    DT = 1.0e-12
    DUTY = 0.30
    I_BIAS_FACTOR = 0.9
    I_PEAK_FACTOR = 5.0

    I_off = I_BIAS_FACTOR * I_th
    I_on = I_PEAK_FACTOR * I_th

    # S_inj values to sweep — covers eta=0.01/cold-cavity to eta=1.0/full-BW
    S_inj_values = np.logspace(17, 22, 25)

    # Repetition rates to test
    target_freqs = [3e9, 5e9, 10e9]

    # Physical reference points for annotation
    sld_result = solve_sld_steady_state(sld, 150e-3)
    P_sld = sld_result['P_out']
    ref_points = {}
    for eta, bw, label in [
        (0.10, 500e9,  'η=0.1, 500 GHz'),
        (1.00, 500e9,  'η=1.0, 500 GHz'),
        (0.10, 5e12,   'η=0.1, 5 THz'),
        (1.00, 2e12,   'η=1.0, 2 THz'),
        (1.00, 10e12,  'η=1.0, full BW'),
    ]:
        inj = InjectionParams(eta_coupling=eta)
        inj.compute_derived(laser)
        s, _ = sld_to_injection_field(sld, P_sld, laser, inj,
                                       acceptance_bandwidth=bw)
        ref_points[label] = s

    print(f"  {N_PULSES/1e3:.0f}k pulses per point, "
          f"{len(S_inj_values)} S_inj values, "
          f"{len(target_freqs)} frequencies")
    print(f"  S_inj range: {S_inj_values[0]:.1e} — {S_inj_values[-1]:.1e} m^-3")
    print(f"\n  Reference S_inj values:")
    for label, s in ref_points.items():
        print(f"    {s:.2e}  ← {label}")

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

    # ── Free-running baselines ───────────────────────────────────────────

    baselines = {}
    print("\n  Free-running baselines:")
    for f_rep in target_freqs:
        T_rep = 1.0 / f_rep
        t_on = DUTY * T_rep
        pts = max(int(round(T_rep / DT)), 50)
        dt = T_rep / pts
        t_rise = min(20e-12, t_on / 4.0)
        seed = 42 + int(f_rep * 1e-9) * 100

        phi, pk_S, _, pk_k = simulate_pulses(
            N_PULSES + N_DISCARD, N_DISCARD, pts, dt,
            I_off, I_on, t_on, t_rise,
            laser.V, laser.Gamma, laser.v_g, laser.a,
            laser.N_tr, laser.epsilon,
            laser.A, laser.B, laser.C,
            laser.tau_p, laser.beta_sp, laser.alpha_H, q,
            0.0, seed)

        m = compute_metrics(phi, pk_S, pk_k, dt, T_rep, laser)
        baselines[f_rep] = m
        print(f"    {f_rep*1e-9:.0f} GHz: r1={m['r1']:.4f}  "
              f"sig_t={m['sig_t']*1e12:.2f} ps  sig_phi={m['sig_phi']:.3f} rad")

    # ── Sweep ────────────────────────────────────────────────────────────

    results = {f: [] for f in target_freqs}
    total_t0 = _time.time()

    for f_rep in target_freqs:
        T_rep = 1.0 / f_rep
        t_on = DUTY * T_rep
        pts = max(int(round(T_rep / DT)), 50)
        dt = T_rep / pts
        t_rise = min(20e-12, t_on / 4.0)
        fg = f_rep * 1e-9

        print(f"\n  === {fg:.0f} GHz (T_rep={T_rep*1e12:.0f} ps, "
              f"pts={pts}, dt={dt*1e12:.2f} ps) ===")
        print(f"  {'S_inj':>12s}  {'r1':>7s}  {'sig_phi':>8s}  "
              f"{'sig_t':>8s}  {'CV':>6s}  {'KL':>7s}")

        for i, s_inj in enumerate(S_inj_values):
            seed = 42 + int(fg) * 100 + i + 1

            t0 = _time.time()
            phi, pk_S, _, pk_k = simulate_pulses(
                N_PULSES + N_DISCARD, N_DISCARD, pts, dt,
                I_off, I_on, t_on, t_rise,
                laser.V, laser.Gamma, laser.v_g, laser.a,
                laser.N_tr, laser.epsilon,
                laser.A, laser.B, laser.C,
                laser.tau_p, laser.beta_sp, laser.alpha_H, q,
                s_inj, seed)
            elapsed = _time.time() - t0

            m = compute_metrics(phi, pk_S, pk_k, dt, T_rep, laser)
            m['S_inj'] = s_inj
            results[f_rep].append(m)

            flag = ' <-- SECURE' if m['r1'] < 0.01 else ''
            print(f"  {s_inj:12.2e}  {m['r1']:7.4f}  "
                  f"{m['sig_phi']:8.3f}  {m['sig_t']*1e12:8.2f}  "
                  f"{m['cv']*100:6.2f}  {m['kl']:7.3f}{flag}")

    total = _time.time() - total_t0
    print(f"\n  Total sweep: {total:.0f}s ({total/60:.1f} min)")

    # ── Find thresholds ──────────────────────────────────────────────────

    R1_THRESHOLD = 0.01
    thresholds = {}

    print(f"\n  Security threshold (r1 < {R1_THRESHOLD}):")
    for f_rep in target_freqs:
        fg = f_rep * 1e-9
        res = results[f_rep]
        s_vals = np.array([r['S_inj'] for r in res])
        r1_vals = np.array([r['r1'] for r in res])

        # Find first crossing below threshold
        secure_mask = r1_vals < R1_THRESHOLD
        if np.any(secure_mask):
            idx = np.argmax(secure_mask)
            s_thresh = s_vals[idx]

            # Interpolate in log space for better estimate
            if idx > 0:
                r1_above = r1_vals[idx - 1]
                r1_below = r1_vals[idx]
                s_above = s_vals[idx - 1]
                s_below = s_vals[idx]
                frac = ((np.log10(R1_THRESHOLD) - np.log10(r1_above)) /
                        (np.log10(r1_below) - np.log10(r1_above)))
                s_thresh_interp = 10**(np.log10(s_above) +
                                       frac * (np.log10(s_below) - np.log10(s_above)))
            else:
                s_thresh_interp = s_thresh

            # Metrics at threshold
            m_thresh = res[idx]
            bl = baselines[f_rep]

            thresholds[f_rep] = dict(
                S_inj=s_thresh_interp,
                S_inj_actual=s_thresh,
                idx=idx,
                metrics=m_thresh,
            )

            print(f"    {fg:.0f} GHz:  S_inj >= {s_thresh_interp:.2e} m^-3")
            print(f"           sig_t = {m_thresh['sig_t']*1e12:.1f} ps "
                  f"(free: {bl['sig_t']*1e12:.1f} ps, "
                  f"{m_thresh['sig_t']/bl['sig_t']:.1f}x)")
            print(f"           CV = {m_thresh['cv']*100:.1f}% "
                  f"(free: {bl['cv']*100:.1f}%)")
        else:
            thresholds[f_rep] = None
            print(f"    {fg:.0f} GHz:  NOT ACHIEVED in sweep range")

    # Map threshold S_inj back to physical parameters
    print(f"\n  Physical parameter mapping (to reach threshold S_inj):")
    for f_rep in target_freqs:
        fg = f_rep * 1e-9
        t = thresholds[f_rep]
        if t is None:
            print(f"    {fg:.0f} GHz: threshold not reached")
            continue
        s_need = t['S_inj']
        print(f"    {fg:.0f} GHz (need S_inj >= {s_need:.2e}):")
        for label, s_ref in ref_points.items():
            if s_ref >= s_need * 0.5:
                print(f"      {label}: S_inj = {s_ref:.2e} {'✓' if s_ref >= s_need else '✗'}")

    # ── Figure 1: r1 vs S_inj — the main result ─────────────────────────

    fig1, ax1 = plt.subplots(figsize=(10, 6))

    colors = {3e9: 'C0', 5e9: 'C1', 10e9: 'C3'}
    for f_rep in target_freqs:
        fg = f_rep * 1e-9
        res = results[f_rep]
        s_vals = [r['S_inj'] for r in res]
        r1_vals = [r['r1'] for r in res]

        col = colors[f_rep]
        ax1.semilogx(s_vals, r1_vals, 'o-', color=col, lw=2, ms=5,
                     label=f'{fg:.0f} GHz')

        # Mark threshold crossing
        t = thresholds[f_rep]
        if t is not None:
            ax1.axvline(t['S_inj'], color=col, ls=':', lw=1, alpha=0.4)
            ax1.annotate(f"  {t['S_inj']:.1e}",
                         xy=(t['S_inj'], R1_THRESHOLD),
                         fontsize=8, color=col, va='bottom')

        # Free-running baseline
        bl = baselines[f_rep]
        ax1.axhline(bl['r1'], color=col, ls='--', lw=0.8, alpha=0.3)

    # Security threshold
    ax1.axhline(R1_THRESHOLD, color='green', ls='-', lw=2, alpha=0.6,
                label=f'Security threshold ($r_1$ = {R1_THRESHOLD})')
    ax1.axhline(0.1, color='orange', ls='--', lw=1, alpha=0.4,
                label='Marginal ($r_1$ = 0.1)')

    # Reference S_inj annotations (top axis)
    ref_colors = ['gray'] * len(ref_points)
    for (label, s), rc in zip(ref_points.items(), ref_colors):
        ax1.axvline(s, color='gray', ls=':', lw=0.6, alpha=0.3)
        ax1.text(s, ax1.get_ylim()[1] * 0.95, f'  {label}',
                 fontsize=6.5, color='gray', rotation=90,
                 va='top', ha='left')

    ax1.set_xlabel('Injected photon density $S_{inj}$ (m$^{-3}$)')
    ax1.set_ylabel('Phase correlation $r_1$ = $|\\langle e^{i\\Delta\\phi}\\rangle|$')
    ax1.set_title(
        'Minimum SLD Injection for Secure Phase Randomisation\n'
        f'500k pulses, I$_{{bias}}$={I_BIAS_FACTOR}×I$_{{th}}$, '
        f'I$_{{peak}}$={I_PEAK_FACTOR}×I$_{{th}}$',
        fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.02, 1.05)
    ax1.set_xlim(S_inj_values[0], S_inj_values[-1])

    plt.tight_layout()
    save_fig(fig1, 'images/qkd_source/sinj_sweep_r1.png', close=False)

    # ── Figure 2: Cost of injection — timing + phase jitter ──────────────

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5.5))
    fig2.suptitle(
        'Cost of SLD Injection — Jitter Penalty at Each $S_{inj}$',
        fontsize=13)

    for f_rep in target_freqs:
        fg = f_rep * 1e-9
        res = results[f_rep]
        s_vals = [r['S_inj'] for r in res]
        col = colors[f_rep]
        bl = baselines[f_rep]
        t = thresholds[f_rep]

        # Timing jitter
        sig_t_vals = [r['sig_t'] * 1e12 for r in res]
        axes2[0].semilogx(s_vals, sig_t_vals, 'o-', color=col, lw=2, ms=4,
                          label=f'{fg:.0f} GHz')
        axes2[0].axhline(bl['sig_t'] * 1e12, color=col, ls='--', lw=0.8, alpha=0.4)

        # Phase jitter
        sig_p_vals = [r['sig_phi'] for r in res]
        axes2[1].semilogx(s_vals, sig_p_vals, 'o-', color=col, lw=2, ms=4,
                          label=f'{fg:.0f} GHz')
        axes2[1].axhline(bl['sig_phi'], color=col, ls='--', lw=0.8, alpha=0.4)

        # Intensity CV
        cv_vals = [r['cv'] * 100 for r in res]
        axes2[2].semilogx(s_vals, cv_vals, 'o-', color=col, lw=2, ms=4,
                          label=f'{fg:.0f} GHz')
        axes2[2].axhline(bl['cv'] * 100, color=col, ls='--', lw=0.8, alpha=0.4)

        # Mark threshold
        if t is not None:
            for ax in axes2:
                ax.axvline(t['S_inj'], color=col, ls=':', lw=1, alpha=0.3)

    axes2[0].set_ylabel('Timing jitter $\\sigma_t$ (ps)')
    axes2[0].set_title('Peak arrival jitter')
    axes2[1].set_ylabel('Phase jitter $\\sigma_{\\Delta\\phi}$ (rad)')
    axes2[1].set_title('Phase jitter (want → $\\pi/\\sqrt{3}$)')
    axes2[1].axhline(np.pi / np.sqrt(3), color='gray', ls=':', lw=1, alpha=0.4,
                     label='Uniform random')
    axes2[2].set_ylabel('Intensity CV (%)')
    axes2[2].set_title('Pulse-to-pulse intensity fluctuation')

    for ax in axes2:
        ax.set_xlabel('$S_{inj}$ (m$^{-3}$)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(S_inj_values[0], S_inj_values[-1])

    plt.tight_layout()
    save_fig(fig2, 'images/qkd_source/sinj_sweep_cost.png', close=False)

    # ── Figure 3: Security–jitter trade-off ──────────────────────────────

    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
    fig3.suptitle(
        'Security vs Performance — Parametric in $S_{inj}$\n'
        'Each point is one $S_{inj}$ value; arrows show increasing injection',
        fontsize=12)

    for f_rep in target_freqs:
        fg = f_rep * 1e-9
        res = results[f_rep]
        col = colors[f_rep]
        t = thresholds[f_rep]

        r1_vals = [r['r1'] for r in res]
        sig_t_vals = [r['sig_t'] * 1e12 for r in res]
        cv_vals = [r['cv'] * 100 for r in res]

        axes3[0].plot(r1_vals, sig_t_vals, 'o-', color=col, lw=1.5, ms=4,
                      label=f'{fg:.0f} GHz')
        axes3[1].plot(r1_vals, cv_vals, 'o-', color=col, lw=1.5, ms=4,
                      label=f'{fg:.0f} GHz')

        # Mark threshold point
        if t is not None:
            m = t['metrics']
            axes3[0].plot(m['r1'], m['sig_t'] * 1e12, '*', color=col,
                          ms=15, zorder=5, markeredgecolor='k', markeredgewidth=0.5)
            axes3[1].plot(m['r1'], m['cv'] * 100, '*', color=col,
                          ms=15, zorder=5, markeredgecolor='k', markeredgewidth=0.5)

    for ax in axes3:
        ax.axvline(R1_THRESHOLD, color='green', ls='-', lw=2, alpha=0.6,
                   label=f'$r_1$ = {R1_THRESHOLD}')
        ax.axvspan(0, R1_THRESHOLD, color='green', alpha=0.05)
        ax.set_xlabel('Phase correlation $r_1$')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.02, 1.02)

    axes3[0].set_ylabel('Timing jitter $\\sigma_t$ (ps)')
    axes3[0].set_title('Timing jitter cost of security')
    axes3[0].text(0.005, axes3[0].get_ylim()[1] * 0.9, 'SECURE',
                  fontsize=10, color='green', alpha=0.5, ha='center')

    axes3[1].set_ylabel('Intensity CV (%)')
    axes3[1].set_title('Intensity noise cost of security')

    plt.tight_layout()
    save_fig(fig3, 'images/qkd_source/sinj_sweep_tradeoff.png', close=False)

    plt.close('all')

    # ── Final summary ────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("  Summary: Minimum S_inj for Secure BB84 (r1 < 0.01)")
    print("=" * 70)
    print(f"  {'f_rep':>5s}  {'S_inj_min':>12s}  {'sig_t':>8s}  "
          f"{'sig_t_free':>10s}  {'ratio':>6s}  {'CV':>6s}")
    print(f"  {'(GHz)':>5s}  {'(m^-3)':>12s}  {'(ps)':>8s}  "
          f"{'(ps)':>10s}  {'':>6s}  {'(%)':>6s}")
    print("  " + "-" * 55)

    for f_rep in target_freqs:
        fg = f_rep * 1e-9
        bl = baselines[f_rep]
        t = thresholds[f_rep]
        if t is not None:
            m = t['metrics']
            ratio = m['sig_t'] / bl['sig_t']
            print(f"  {fg:5.0f}  {t['S_inj']:12.2e}  {m['sig_t']*1e12:8.1f}  "
                  f"{bl['sig_t']*1e12:10.1f}  {ratio:6.1f}x  "
                  f"{m['cv']*100:6.1f}")
        else:
            print(f"  {fg:5.0f}  {'> '+f'{S_inj_values[-1]:.0e}':>12s}  "
                  f"{'—':>8s}  {bl['sig_t']*1e12:10.1f}  {'—':>6s}  {'—':>6s}")

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)
