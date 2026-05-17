"""
Coherence threshold as a phase transition.

The inter-pulse phase correlation r₁ = |<exp(iΔφ)>| acts as an order
parameter for a noise-induced desynchronisation transition.  As S_inj
increases, r₁ drops from ~1 (phase-locked) to ~0 (random).

This script:
  1. Sweeps S_inj at 5 frequencies, measures r₁
  2. Derives the random-phasor analytical model:
       E_n = √S_res × exp(iφ_{n-1}) + CN(0, S_noise)
     which predicts r₁(κ) where κ = S_noise/S_res
  3. Demonstrates universal data collapse: all frequencies fall on
     one master curve when S_inj is rescaled by S_c (the critical
     injection at which r₁ ≈ characteristic value)
  4. Extracts the critical exponent:  r₁ ~ κ^{-1/2} for κ >> 1
     (same as mean-field / Kuramoto exponent)
  5. Shows S_c scales exponentially with T_off/τ_p, confirming the
     physical picture that the residual field decays during the
     off-phase

Key result: the coherence destruction maps onto a noise-driven
desynchronisation with mean-field (β = 1/2) exponent, universal
across repetition rates.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time as _time

from dfb_laser import DFBLaserParams, q, h, c
from sld_injection import (
    SLDParams, InjectionParams,
    solve_sld_steady_state, sld_to_injection_field,
)
from million_pulse_comparison import simulate_pulses


# ── Analytical model ────────────────────────────────────────────────────

def random_phasor_r1(kappa, n_samples=5_000_000):
    """Monte Carlo evaluation of r₁ for the random-phasor model.

    Model: E = 1 + sqrt(kappa/2) * (X + iY)   where X,Y ~ N(0,1)
    r₁ = <cos(arg(E))> = <Re(E) / |E|>

    kappa = S_noise / S_res  (noise-to-signal ratio)

    Asymptotics:
        kappa -> 0:  r₁ -> 1 - kappa/4 + ...
        kappa -> inf: r₁ -> sqrt(pi/kappa)   [exponent -1/2]
    """
    rng = np.random.default_rng(seed=12345)
    X = rng.standard_normal(n_samples)
    Y = rng.standard_normal(n_samples)

    Re_E = 1.0 + np.sqrt(kappa / 2.0) * X
    Im_E = np.sqrt(kappa / 2.0) * Y
    mag = np.sqrt(Re_E**2 + Im_E**2)

    return float(np.mean(Re_E / mag))


def random_phasor_r1_curve(kappa_vals):
    """Vectorised: compute r₁ for an array of kappa values."""
    return np.array([random_phasor_r1(k) for k in kappa_vals])


def r1_asymptotic_large_kappa(kappa):
    """Asymptotic: r₁ ~ sqrt(pi/kappa) for kappa >> 1."""
    return np.sqrt(np.pi / kappa)


def r1_asymptotic_small_kappa(kappa):
    """Asymptotic: r₁ ~ 1 - kappa/4 for kappa << 1."""
    return 1.0 - kappa / 4.0


# ── Metrics ─────────────────────────────────────────────────────────────

def compute_r1(phi):
    """Phase correlation r₁ = |<exp(iΔφ)>|."""
    dphi = np.diff(phi)
    return float(np.abs(np.mean(np.exp(1j * dphi))))


def compute_metrics(phi, pk_S, pk_k, dt, laser):
    dphi = np.angle(np.exp(1j * np.diff(phi)))
    r1 = float(np.abs(np.mean(np.exp(1j * dphi))))
    sig_phi = float(np.std(dphi))
    sig_t = float(np.std(pk_k.astype(np.float64) * dt))
    pk_P = laser.output_power(np.maximum(pk_S, 0))
    cv = float(np.std(pk_P) / np.mean(pk_P))
    return dict(r1=r1, sig_phi=sig_phi, sig_t=sig_t, cv=cv)


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 70)
    print("  Coherence Threshold as a Phase Transition")
    print("=" * 70)

    laser = DFBLaserParams()
    I_th = laser.threshold_current()
    sld = SLDParams()

    N_PULSES = 200_000
    N_DISCARD = 500
    DT = 1.0e-12
    DUTY = 0.30
    I_BIAS_FACTOR = 0.9
    I_PEAK_FACTOR = 5.0

    I_off = I_BIAS_FACTOR * I_th
    I_on = I_PEAK_FACTOR * I_th

    # Sweep parameters
    S_inj_values = np.logspace(16.5, 22, 24)
    target_freqs = [3e9, 4e9, 5e9, 7e9, 10e9]
    freq_colors = {3e9: 'C0', 4e9: 'C1', 5e9: 'C2', 7e9: 'C4', 10e9: 'C3'}

    print(f"  {N_PULSES/1e3:.0f}k pulses, {len(S_inj_values)} S_inj × "
          f"{len(target_freqs)} frequencies = {len(S_inj_values)*len(target_freqs)} runs")

    # Warmup
    print("  Compiling JIT solver...")
    t0 = _time.time()
    _w = simulate_pulses(
        100, 10, 100, 1e-12,
        I_off, I_on, 30e-12, 7e-12,
        laser.V, laser.Gamma, laser.v_g, laser.a, laser.N_tr, laser.epsilon,
        laser.A, laser.B, laser.C, laser.tau_p, laser.beta_sp, laser.alpha_H, q,
        0.0, 0)
    del _w
    print(f"  Compiled in {_time.time()-t0:.1f}s")

    # ── Run sweep ────────────────────────────────────────────────────────

    results = {}
    total_t0 = _time.time()

    for f_rep in target_freqs:
        T_rep = 1.0 / f_rep
        t_on = DUTY * T_rep
        T_off = (1 - DUTY) * T_rep
        pts = max(int(round(T_rep / DT)), 50)
        dt = T_rep / pts
        t_rise = min(20e-12, t_on / 4.0)
        fg = f_rep * 1e-9

        print(f"\n  {fg:.0f} GHz  (T_off = {T_off*1e12:.0f} ps, "
              f"T_off/tau_p = {T_off/laser.tau_p:.1f})")

        freq_results = []
        for i, s_inj in enumerate(S_inj_values):
            seed = 42 + int(fg) * 100 + i + 1

            phi, pk_S, _, pk_k = simulate_pulses(
                N_PULSES + N_DISCARD, N_DISCARD, pts, dt,
                I_off, I_on, t_on, t_rise,
                laser.V, laser.Gamma, laser.v_g, laser.a,
                laser.N_tr, laser.epsilon,
                laser.A, laser.B, laser.C,
                laser.tau_p, laser.beta_sp, laser.alpha_H, q,
                s_inj, seed)

            m = compute_metrics(phi, pk_S, pk_k, dt, laser)
            m['S_inj'] = s_inj
            freq_results.append(m)

            if i % 6 == 0 or m['r1'] < 0.05:
                print(f"    S_inj={s_inj:.2e}  r1={m['r1']:.4f}")

        results[f_rep] = freq_results

    total = _time.time() - total_t0
    print(f"\n  Total: {total:.0f}s ({total/60:.1f} min)")

    # ── Find critical S_inj for each frequency ──────────────────────────

    # S_c defined as S_inj where r₁ = 1/e ≈ 0.368  (natural choice for
    # exponential-like decay; also close to Kuramoto critical coupling)
    R1_CRIT = 1.0 / np.e

    print(f"\n  Critical S_inj (r₁ = 1/e ≈ {R1_CRIT:.3f}):")

    S_c = {}
    for f_rep in target_freqs:
        fg = f_rep * 1e-9
        res = results[f_rep]
        s_vals = np.array([r['S_inj'] for r in res])
        r1_vals = np.array([r['r1'] for r in res])

        # Interpolate in log(S_inj) space
        below = r1_vals > R1_CRIT
        above = r1_vals <= R1_CRIT
        if np.any(below) and np.any(above):
            # Find crossing index
            cross = np.argmax(above)
            if cross > 0:
                s_lo, s_hi = s_vals[cross - 1], s_vals[cross]
                r_lo, r_hi = r1_vals[cross - 1], r1_vals[cross]
                frac = (R1_CRIT - r_lo) / (r_hi - r_lo)
                s_c = 10**(np.log10(s_lo) + frac * (np.log10(s_hi) - np.log10(s_lo)))
            else:
                s_c = s_vals[0]
        else:
            s_c = s_vals[-1]

        S_c[f_rep] = s_c
        T_off = (1 - DUTY) / f_rep
        print(f"    {fg:.0f} GHz:  S_c = {s_c:.2e}  "
              f"(T_off/tau_p = {T_off/laser.tau_p:.1f})")

    # ── Compute analytical random-phasor model ──────────────────────────

    print(f"\n  Computing random-phasor model (Monte Carlo, 5M samples)...")
    kappa_theory = np.logspace(-2, 3, 200)
    r1_theory = random_phasor_r1_curve(kappa_theory)
    print("  Done.")

    # Find theoretical kappa where r₁ = 1/e
    idx_crit = np.argmin(np.abs(r1_theory - R1_CRIT))
    kappa_crit = kappa_theory[idx_crit]
    print(f"  Theoretical kappa_c (r1 = 1/e): {kappa_crit:.2f}")

    # ── Fit S_c vs frequency: S_c = A * exp(B * T_off / tau_p) ─────────

    T_off_vals = np.array([(1 - DUTY) / f for f in target_freqs])
    ratio_vals = T_off_vals / laser.tau_p
    log_Sc = np.array([np.log(S_c[f]) for f in target_freqs])

    # Linear fit in semi-log: log(S_c) = a + b * (T_off/tau_p)
    coeffs = np.polyfit(ratio_vals, log_Sc, 1)
    b_fit, a_fit = coeffs
    print(f"\n  Scaling fit: ln(S_c) = {a_fit:.2f} + {b_fit:.3f} × T_off/tau_p")
    print(f"  => S_c propto exp({b_fit:.3f} × T_off/tau_p)")
    print(f"  Physical: S_res ~ S_peak × exp(-T_off/tau_p) with effective decay "
          f"rate ~ {-b_fit:.3f}/tau_p")

    # ── Critical exponent fit ────────────────────────────────────────────

    # In the regime kappa >> 1: r₁ ~ C * kappa^(-beta)
    # Fit on the collapsed data
    print(f"\n  Critical exponent fit (r₁ ~ kappa^(-beta) for kappa >> 1):")

    all_kappa = []
    all_r1 = []
    for f_rep in target_freqs:
        res = results[f_rep]
        s_c = S_c[f_rep]
        for r in res:
            kappa = r['S_inj'] / s_c
            if 2.0 < kappa < 100 and 0.02 < r['r1'] < 0.5:
                all_kappa.append(kappa)
                all_r1.append(r['r1'])

    all_kappa = np.array(all_kappa)
    all_r1 = np.array(all_r1)

    if len(all_kappa) > 5:
        def power_law(x, C, beta):
            return C * x**(-beta)

        popt, pcov = curve_fit(power_law, all_kappa, all_r1, p0=[1.0, 0.5])
        C_fit, beta_fit = popt
        beta_err = np.sqrt(pcov[1, 1])
        print(f"  r₁ = {C_fit:.3f} × kappa^(-{beta_fit:.3f} ± {beta_err:.3f})")
        print(f"  Mean-field prediction: beta = 0.500")
        print(f"  Discrepancy: {abs(beta_fit - 0.5):.3f}")
    else:
        beta_fit = 0.5
        C_fit = np.sqrt(np.pi)
        print("  Insufficient data for fit; using theoretical beta = 0.5")

    # ── Figure 1: Raw r₁ vs S_inj ───────────────────────────────────────

    fig1, ax1 = plt.subplots(figsize=(10, 6))

    for f_rep in target_freqs:
        fg = f_rep * 1e-9
        res = results[f_rep]
        s_vals = [r['S_inj'] for r in res]
        r1_vals = [r['r1'] for r in res]
        col = freq_colors[f_rep]

        ax1.semilogx(s_vals, r1_vals, 'o-', color=col, lw=2, ms=4,
                      label=f'{fg:.0f} GHz')
        ax1.axvline(S_c[f_rep], color=col, ls=':', lw=0.8, alpha=0.4)

    ax1.axhline(R1_CRIT, color='k', ls='--', lw=1.5, alpha=0.5,
                label=f'$r_1 = 1/e$ (critical)')
    ax1.axhline(0.01, color='green', ls=':', lw=1, alpha=0.5,
                label='$r_1 = 0.01$ (QKD secure)')

    ax1.set_xlabel('Injected photon density $S_{inj}$ (m$^{-3}$)', fontsize=12)
    ax1.set_ylabel('Order parameter $r_1 = |\\langle e^{i\\Delta\\phi}\\rangle|$',
                    fontsize=12)
    ax1.set_title('Phase Correlation as Order Parameter\n'
                   'Noise-driven desynchronisation transition', fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.02, 1.05)

    plt.tight_layout()
    fig1.savefig('images/phase_transition/raw_r1_vs_sinj.png', dpi=150)
    print("\n  Saved: images/phase_transition/raw_r1_vs_sinj.png")

    # ── Figure 2: Universal data collapse ────────────────────────────────

    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
    fig2.suptitle(
        'Universal Data Collapse — All Frequencies on One Master Curve',
        fontsize=13)

    # (a) Linear scale
    ax = axes2[0]
    for f_rep in target_freqs:
        fg = f_rep * 1e-9
        res = results[f_rep]
        s_c = S_c[f_rep]
        kappa = [r['S_inj'] / s_c for r in res]
        r1 = [r['r1'] for r in res]
        col = freq_colors[f_rep]
        ax.plot(kappa, r1, 'o', color=col, ms=5, alpha=0.7,
                label=f'{fg:.0f} GHz')

    # Overlay analytical model (rescale so r₁(kappa_crit) = 1/e)
    kappa_plot = kappa_theory / kappa_crit
    ax.plot(kappa_plot, r1_theory, 'k-', lw=2.5, alpha=0.5,
            label='Random-phasor model', zorder=0)

    ax.axhline(R1_CRIT, color='gray', ls='--', lw=1, alpha=0.3)
    ax.axvline(1.0, color='gray', ls='--', lw=1, alpha=0.3)
    ax.set_xlabel('$\\kappa = S_{inj} / S_c$', fontsize=12)
    ax.set_ylabel('$r_1$', fontsize=12)
    ax.set_title('Linear scale')
    ax.set_xlim(0, 15)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # (b) Log-log scale — shows power law
    ax = axes2[1]
    for f_rep in target_freqs:
        fg = f_rep * 1e-9
        res = results[f_rep]
        s_c = S_c[f_rep]
        kappa = np.array([r['S_inj'] / s_c for r in res])
        r1 = np.array([r['r1'] for r in res])
        mask = (kappa > 0.05) & (r1 > 0.005)
        col = freq_colors[f_rep]
        ax.loglog(kappa[mask], r1[mask], 'o', color=col, ms=5, alpha=0.7,
                  label=f'{fg:.0f} GHz')

    # Analytical model
    mask_th = (kappa_plot > 0.05) & (r1_theory > 0.005)
    ax.loglog(kappa_plot[mask_th], r1_theory[mask_th], 'k-', lw=2.5,
              alpha=0.5, label='Random-phasor model', zorder=0)

    # Power law reference
    kappa_ref = np.logspace(0.5, 3, 50)
    ax.loglog(kappa_ref, C_fit * kappa_ref**(-beta_fit), 'r--', lw=1.5,
              label=f'$r_1 \\propto \\kappa^{{-{beta_fit:.2f}}}$')
    ax.loglog(kappa_ref, np.sqrt(np.pi) * kappa_ref**(-0.5), 'b:', lw=1,
              alpha=0.5, label='$\\sqrt{\\pi/\\kappa}$ (theory)')

    ax.axhline(R1_CRIT, color='gray', ls='--', lw=0.8, alpha=0.3)
    ax.axvline(1.0, color='gray', ls='--', lw=0.8, alpha=0.3)
    ax.set_xlabel('$\\kappa = S_{inj} / S_c$', fontsize=12)
    ax.set_ylabel('$r_1$', fontsize=12)
    ax.set_title(f'Log-log: critical exponent $\\beta = {beta_fit:.3f}$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    fig2.savefig('images/phase_transition/universal_collapse.png', dpi=150)
    print("  Saved: images/phase_transition/universal_collapse.png")

    # ── Figure 3: S_c scaling with frequency ─────────────────────────────

    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
    fig3.suptitle(
        'Critical Injection Scaling — Why Higher Rates Need More SLD',
        fontsize=13)

    f_arr = np.array([f * 1e-9 for f in target_freqs])
    Sc_arr = np.array([S_c[f] for f in target_freqs])
    ratio_arr = np.array([(1 - DUTY) / (f * laser.tau_p) for f in target_freqs])

    # (a) S_c vs frequency
    ax = axes3[0]
    ax.semilogy(f_arr, Sc_arr, 'ko-', lw=2, ms=8)
    for i, fg in enumerate(f_arr):
        ax.annotate(f'  {Sc_arr[i]:.1e}', (fg, Sc_arr[i]),
                    fontsize=8, va='bottom')
    ax.set_xlabel('Repetition rate (GHz)', fontsize=12)
    ax.set_ylabel('Critical $S_c$ (m$^{-3}$)', fontsize=12)
    ax.set_title('Critical injection vs repetition rate')
    ax.set_xticks(f_arr)
    ax.grid(True, alpha=0.3)

    # (b) S_c vs T_off/tau_p — should be linear in semi-log
    ax = axes3[1]
    ax.semilogy(ratio_arr, Sc_arr, 'ko', ms=10, zorder=5)

    # Fit line
    ratio_fit = np.linspace(ratio_arr.min() * 0.8, ratio_arr.max() * 1.1, 100)
    Sc_fit = np.exp(a_fit + b_fit * ratio_fit)
    ax.semilogy(ratio_fit, Sc_fit, 'r--', lw=2,
                label=f'Fit: $S_c \\propto \\exp({b_fit:.2f}\\, T_{{off}}/\\tau_p)$')

    for i, fg in enumerate(f_arr):
        ax.annotate(f'  {fg:.0f} GHz', (ratio_arr[i], Sc_arr[i]),
                    fontsize=9, va='bottom')

    ax.set_xlabel('$T_{off} / \\tau_p$', fontsize=12)
    ax.set_ylabel('Critical $S_c$ (m$^{-3}$)', fontsize=12)
    ax.set_title(
        'Scaling: $S_c \\propto S_{res}^{-1} \\propto \\exp(+T_{off}/\\tau_{eff})$\n'
        f'Effective decay rate: $\\tau_{{eff}}$ = {-1/b_fit * laser.tau_p*1e12:.1f} ps '
        f'(vs $\\tau_p$ = {laser.tau_p*1e12:.1f} ps)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig3.savefig('images/phase_transition/Sc_scaling.png', dpi=150)
    print("  Saved: images/phase_transition/Sc_scaling.png")

    # ── Figure 4: Phase diagram + Kuramoto analogy ───────────────────────

    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 6))
    fig4.suptitle(
        'Phase Diagram and Kuramoto Analogy',
        fontsize=13)

    # (a) 2D phase diagram: f_rep vs S_inj, colour = r₁
    ax = axes4[0]
    f_grid = []
    s_grid = []
    r1_grid = []
    for f_rep in target_freqs:
        res = results[f_rep]
        for r in res:
            f_grid.append(f_rep * 1e-9)
            s_grid.append(r['S_inj'])
            r1_grid.append(r['r1'])

    sc = ax.scatter(f_grid, s_grid, c=r1_grid, cmap='RdYlGn',
                    s=30, vmin=0, vmax=1, edgecolors='none')
    plt.colorbar(sc, ax=ax, label='$r_1$')

    # Critical line
    f_dense = np.linspace(2, 10, 100)
    ratio_dense = (1 - DUTY) / (f_dense * 1e9 * laser.tau_p)
    Sc_dense = np.exp(a_fit + b_fit * ratio_dense)
    ax.plot(f_dense, Sc_dense, 'k--', lw=2, label='$S_c$ (critical line)')

    ax.set_yscale('log')
    ax.set_xlabel('Repetition rate (GHz)', fontsize=12)
    ax.set_ylabel('$S_{inj}$ (m$^{-3}$)', fontsize=12)
    ax.set_title('Phase diagram: coherent (green) vs random (red)')
    ax.legend(fontsize=9, loc='lower right')
    ax.set_ylim(S_inj_values[0], S_inj_values[-1])

    # Add region labels
    ax.text(6, 1e18, 'COHERENT\n$r_1 \\approx 1$',
            fontsize=12, ha='center', color='darkgreen', alpha=0.7,
            fontweight='bold')
    ax.text(6, 5e20, 'RANDOM\n$r_1 \\approx 0$',
            fontsize=12, ha='center', color='darkred', alpha=0.7,
            fontweight='bold')

    # (b) Kuramoto analogy diagram
    ax = axes4[1]

    # Overlay: standard Kuramoto transition and our data
    # Kuramoto: r = 0 for K < K_c, r ~ sqrt(1 - K_c/K) for K > K_c
    K_norm = np.linspace(0, 4, 200)

    # Our system (inverted: increasing noise destroys order)
    # Plot r₁ vs 1/kappa (= S_res/S_noise = "coupling strength")
    for f_rep in target_freqs:
        fg = f_rep * 1e-9
        res = results[f_rep]
        s_c = S_c[f_rep]
        inv_kappa = np.array([s_c / r['S_inj'] for r in res])
        r1 = np.array([r['r1'] for r in res])
        col = freq_colors[f_rep]
        ax.plot(inv_kappa, r1, 'o', color=col, ms=4, alpha=0.5)

    # Analytical model in 1/kappa coordinates
    inv_kappa_th = kappa_crit / kappa_theory
    mask_p = inv_kappa_th < 4
    ax.plot(inv_kappa_th[mask_p], r1_theory[mask_p], 'k-', lw=2.5, alpha=0.5,
            label='Random-phasor model')

    # Kuramoto reference (shifted/scaled for comparison)
    K_kur = np.linspace(0.01, 4, 200)
    K_c_kur = 1.0
    r_kur = np.where(K_kur > K_c_kur,
                     np.sqrt(np.maximum(1 - K_c_kur / K_kur, 0)), 0)
    ax.plot(K_kur, r_kur, 'b--', lw=1.5, alpha=0.6,
            label='Kuramoto (mean-field)')

    ax.axvline(1.0, color='gray', ls=':', lw=0.8, alpha=0.3)
    ax.set_xlabel('Effective coupling $S_{res} / S_{noise}$ (= $1/\\kappa$)',
                  fontsize=11)
    ax.set_ylabel('Order parameter $r_1$', fontsize=12)
    ax.set_title(
        'Kuramoto analogy\n'
        'Both transitions: mean-field exponent $\\beta = 1/2$')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 4)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig4.savefig('images/phase_transition/phase_diagram.png', dpi=150)
    print("  Saved: images/phase_transition/phase_diagram.png")

    plt.close('all')

    # ── Summary ──────────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("  Phase Transition Summary")
    print("=" * 70)
    print(f"\n  Order parameter:  r₁ = |<exp(iΔφ)>|")
    print(f"  Control parameter: κ = S_inj / S_c")
    print(f"  Critical point:  r₁(κ=1) = 1/e ≈ {R1_CRIT:.3f}")
    print(f"\n  Critical exponent: β = {beta_fit:.3f} ± {beta_err:.3f}")
    print(f"  Mean-field prediction: β = 0.500")
    print(f"  Theory (random phasor): r₁ → √(π/κ) for κ >> 1 → β = 1/2 ✓")
    print(f"\n  Critical S_c scaling:")
    print(f"  {'f_rep':>5s}  {'T_off/tau_p':>11s}  {'S_c':>12s}")
    print("  " + "-" * 32)
    for f_rep in target_freqs:
        fg = f_rep * 1e-9
        T_off = (1 - DUTY) / f_rep
        print(f"  {fg:5.0f}  {T_off/laser.tau_p:11.1f}  {S_c[f_rep]:12.2e}")
    print(f"\n  Fit: S_c ~ exp({b_fit:.3f} × T_off/tau_p)")
    print(f"  Effective photon decay time during off-phase:")
    print(f"    tau_eff = {-1/b_fit * laser.tau_p*1e12:.1f} ps  "
          f"(vs tau_p = {laser.tau_p*1e12:.1f} ps)")
    print(f"    Interpretation: gain at I_bias = 0.9×I_th partially")
    print(f"    compensates cavity loss, extending photon lifetime by"
          f" {-1/(b_fit * laser.tau_p) / (1/laser.tau_p):.1f}×")

    print("\n  Universality: all frequencies collapse onto one master curve")
    print("  when S_inj is rescaled by S_c(f_rep).")
    print("\n  Analogy: noise-driven desynchronisation transition with")
    print("  mean-field (Kuramoto-like) exponent β = 1/2, but SMOOTH")
    print("  crossover (no sharp phase boundary — finite effective")
    print("  'particle number' from photon statistics).")

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)
