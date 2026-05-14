"""
QKD source characterisation of gain-switched DFB laser, 1-10 GHz.

Computes security-relevant metrics for BB84 decoy-state QKD:
  - Phase randomisation quality (KL divergence from uniform, r1)
  - Photon number statistics (Fano factor, compound Poisson distribution)
  - Pulse-to-pulse intensity correlations (autocorrelation)
  - Secure key rate vs distance for 3-intensity decoy-state BB84
  - Security verdict: which (f_rep, source) configurations are usable

Key result: standard security proofs REQUIRE phase-randomised pulses.
Free-running gain switching only achieves this at low f_rep (< 3 GHz);
SLD injection extends the secure operating range to higher rates.

Uses 1M-pulse numba solver for statistical convergence.
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


# ── QKD metrics ─────────────────────────────────────────────────────────

def phase_randomisation_quality(phi, n_bins=128):
    """Assess how uniform the inter-pulse phase distribution is.

    Returns:
        kl: KL divergence D_KL(p || uniform) in bits — 0 = perfectly uniform.
        ks_stat: Kolmogorov-Smirnov statistic vs uniform on [-pi, pi].
        r1: first-order phase correlation |<exp(i*dphi)>|.
        dphi: array of wrapped phase differences.
    """
    dphi = np.angle(np.exp(1j * np.diff(phi)))

    # r1
    r1 = np.abs(np.mean(np.exp(1j * dphi)))

    # KL divergence from uniform
    hist, _ = np.histogram(dphi, bins=n_bins, range=(-np.pi, np.pi))
    hist = hist.astype(np.float64) + 1e-10        # avoid log(0)
    p = hist / hist.sum()
    q_unif = np.ones(n_bins) / n_bins
    kl = float(np.sum(p * np.log2(p / q_unif)))

    # KS statistic
    dphi_sorted = np.sort(dphi)
    n = len(dphi_sorted)
    cdf_emp = np.arange(1, n + 1) / n
    cdf_unif = (dphi_sorted + np.pi) / (2 * np.pi)
    ks_stat = float(np.max(np.abs(cdf_emp - cdf_unif)))

    return kl, ks_stat, r1, dphi


def photon_number_statistics(pk_P, target_mu=0.5):
    """Compute photon number distribution assuming coherent-state pulses.

    Each pulse has peak power P_k. After attenuation to mean mu:
        mu_k = target_mu * P_k / <P>

    The compound distribution P(n) = <Poisson(n | mu_k)> accounts for
    intensity fluctuations across pulses.

    Returns:
        fano: compound Fano factor  F = 1 + mu * CV^2
        cv: coefficient of variation of pulse peak powers
        mu_k: per-pulse attenuated mean photon number array
        pn: compound P(n) for n = 0..n_max
    """
    mu_k = target_mu * pk_P / np.mean(pk_P)
    cv = float(np.std(pk_P) / np.mean(pk_P))

    # Compound Fano: F = 1 + <mu> * CV^2(mu_k)
    fano = 1.0 + target_mu * cv**2

    # Compound Poisson via histogram of mu_k
    n_max = max(int(target_mu * 5) + 10, 15)
    n_vals = np.arange(n_max + 1)
    mu_hist, mu_edges = np.histogram(mu_k, bins=500, density=True)
    mu_cen = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    dmu = mu_edges[1] - mu_edges[0]

    pn = np.zeros(n_max + 1)
    for n in n_vals:
        log_poisson = -mu_cen + n * np.log(mu_cen + 1e-30) - _log_factorial(n)
        pn[n] = np.sum(np.exp(log_poisson) * mu_hist * dmu)

    return fano, cv, mu_k, pn


def _log_factorial(n):
    """log(n!) via lookup or Stirling for large n."""
    import math
    if n <= 20:
        return np.log(float(math.factorial(int(n))))
    # Stirling
    return n * np.log(n) - n + 0.5 * np.log(2 * np.pi * n)


def intensity_autocorrelation(pk_P, max_lag=50):
    """Normalised autocorrelation of pulse peak powers."""
    x = pk_P - np.mean(pk_P)
    var = np.var(pk_P)
    if var < 1e-30:
        return np.arange(max_lag + 1), np.zeros(max_lag + 1)
    acf = np.zeros(max_lag + 1)
    n = len(x)
    for lag in range(max_lag + 1):
        acf[lag] = np.mean(x[:n - lag] * x[lag:]) / var
    return np.arange(max_lag + 1), acf


def bb84_decoy_key_rate(mu_s, mu_w, eta_ch, eta_det, p_dark, e_mis,
                        f_ec=1.16):
    """Asymptotic BB84 3-intensity decoy-state key rate per pulse.

    Signal mu_s, weak decoy mu_w, vacuum decoy mu_v = 0.

    Returns R in bits/pulse (>=0).
    """
    eta = eta_ch * eta_det

    # Gains
    Q_s = 1 - (1 - p_dark) * np.exp(-eta * mu_s)
    Q_w = 1 - (1 - p_dark) * np.exp(-eta * mu_w)
    Q_v = p_dark  # vacuum

    # Overall QBER for signal
    E_s = (e_mis * eta * mu_s * np.exp(-eta * mu_s) + 0.5 * p_dark) / Q_s

    # Single-photon yield lower bound
    Y1 = max((Q_w * np.exp(mu_w) - Q_v) / mu_w, 0.0)
    Y1 = min(Y1, 1.0)

    # Single-photon gain
    Q1 = Y1 * mu_s * np.exp(-mu_s)

    # Single-photon QBER upper bound
    e1 = min((E_s * Q_s - 0.5 * Q_v) / max(Q1, 1e-30), 0.5)
    e1 = max(e1, 0.0)

    # Binary entropy
    def H(x):
        x = np.clip(x, 1e-10, 1 - 1e-10)
        return -x * np.log2(x) - (1 - x) * np.log2(1 - x)

    R = max(Q1 * (1 - H(e1)) - f_ec * Q_s * H(E_s), 0.0)
    return R


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 70)
    print("  QKD Source Characterisation — 1M Pulses, 1-10 GHz")
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

    lam = laser.lambda0
    nu = c / lam
    E_photon = h * nu

    print(f"  S_inj = {S_INJ:.2e} m^-3")
    print(f"  lambda = {lam*1e9:.1f} nm, E_photon = {E_photon*1e19:.3f}e-19 J")
    print(f"  {N_PULSES/1e6:.0f}M pulses per run\n")

    # ── Warmup JIT ───────────────────────────────────────────────────────
    print("  Compiling JIT solver...")
    t0 = _time.time()
    _w = simulate_pulses(
        100, 10, 100, 1e-12,
        I_off, I_on, 30e-12, 7e-12,
        laser.V, laser.Gamma, laser.v_g, laser.a, laser.N_tr, laser.epsilon,
        laser.A, laser.B, laser.C, laser.tau_p, laser.beta_sp, laser.alpha_H, q,
        0.0, 0)
    del _w
    print(f"  Compiled in {_time.time()-t0:.1f}s\n")

    # ── Run simulations ─────────────────────────────────────────────────
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

            # Phase
            kl, ks, r1, dphi = phase_randomisation_quality(phi)

            # Photon stats
            fano, cv, mu_k, pn = photon_number_statistics(pk_P, target_mu=0.5)

            # Intensity correlations
            lags, acf = intensity_autocorrelation(pk_P)

            data[key] = dict(
                f_rep=f_rep, T_rep=T_rep, case=case,
                phi=phi, pk_P=pk_P, dphi=dphi,
                kl=kl, ks=ks, r1=r1,
                fano=fano, cv=cv, mu_k=mu_k, pn=pn,
                lags=lags, acf=acf,
            )

            print(f" {elapsed:5.1f}s  r1={r1:.4f}  "
                  f"KL={kl:.4f}bit  F={fano:.4f}  CV={cv*100:.2f}%")

    total_elapsed = _time.time() - total_t0
    print(f"\n  Total: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")

    # ── Figure 1: Phase randomisation quality ────────────────────────────

    fig1, axes1 = plt.subplots(2, 5, figsize=(24, 8), sharey='row')
    fig1.suptitle(
        'Phase Randomisation Quality — $\\Delta\\phi$ Distribution (1M pulses)\n'
        'Uniform (dashed line) $=$ secure for BB84;  peaked $=$ information leakage',
        fontsize=13)

    sel_phase = [1, 2, 3, 5, 10]
    for col, fg in enumerate(sel_phase):
        for row, (case, color, label) in enumerate([
            ('free', 'C0', 'Free-running'),
            ('sld', 'C3', 'SLD-injected'),
        ]):
            key = f"{fg}GHz_{case}"
            d = data[key]
            ax = axes1[row, col]

            ax.hist(d['dphi'], bins=128, range=(-np.pi, np.pi),
                    density=True, color=color, alpha=0.7, edgecolor='none')
            ax.axhline(1 / (2 * np.pi), color='k', ls='--', lw=1, alpha=0.5)
            ax.set_xlim(-np.pi, np.pi)
            verdict = ('RANDOM' if d['r1'] < 0.01
                       else 'PARTIAL' if d['r1'] < 0.1
                       else 'CORREL.')
            ax.set_title(
                f'{fg} GHz — {label}\n'
                f'$r_1$={d["r1"]:.3f}  KL={d["kl"]:.3f}b  [{verdict}]',
                fontsize=9)
            ax.grid(True, alpha=0.2)
            if col == 0:
                ax.set_ylabel('Prob. density')

        axes1[1, col].set_xlabel('$\\Delta\\phi$ (rad)')

    plt.tight_layout()
    fig1.savefig('images/qkd_source/phase_randomisation.png', dpi=150)
    print("  Saved: images/qkd_source/phase_randomisation.png")

    # ── Figure 2: Source metrics vs frequency ────────────────────────────

    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle(
        'QKD Source Metrics vs Repetition Rate — 1M Pulses\n'
        f'$I_{{bias}}$={I_BIAS_FACTOR}$\\times I_{{th}}$, '
        f'$I_{{peak}}$={I_PEAK_FACTOR}$\\times I_{{th}}$, '
        f'$S_{{inj}}$={S_INJ:.1e} m$^{{-3}}$',
        fontsize=12)

    for case, color, marker, label in [
        ('free', 'C0', 'o', 'Free-running'),
        ('sld', 'C3', 's', 'SLD-injected'),
    ]:
        r1_vals  = [data[f"{f:.0f}GHz_{case}"]['r1']  for f in f_ghz]
        kl_vals  = [data[f"{f:.0f}GHz_{case}"]['kl']  for f in f_ghz]
        fano_vals = [data[f"{f:.0f}GHz_{case}"]['fano'] for f in f_ghz]
        cv_vals  = [data[f"{f:.0f}GHz_{case}"]['cv'] * 100 for f in f_ghz]

        axes2[0, 0].semilogy(f_ghz, r1_vals, f'{marker}-', color=color,
                             lw=2, ms=7, label=label)
        axes2[0, 1].semilogy(f_ghz, kl_vals, f'{marker}-', color=color,
                             lw=2, ms=7, label=label)
        axes2[1, 0].plot(f_ghz, fano_vals, f'{marker}-', color=color,
                         lw=2, ms=7, label=label)
        axes2[1, 1].plot(f_ghz, cv_vals, f'{marker}-', color=color,
                         lw=2, ms=7, label=label)

    # Thresholds / references
    axes2[0, 0].axhline(0.01, color='green', ls=':', lw=1.5, alpha=0.6,
                         label='$r_1$ = 0.01 (secure)')
    axes2[0, 0].axhline(0.1,  color='orange', ls=':', lw=1.5, alpha=0.6,
                         label='$r_1$ = 0.1 (marginal)')
    axes2[1, 0].axhline(1.0, color='gray', ls=':', lw=1.0, alpha=0.5,
                         label='Poissonian ($F$ = 1)')

    axes2[0, 0].set_ylabel('Phase correlation $r_1$')
    axes2[0, 0].set_title('Phase correlation (lower = more random)')
    axes2[0, 1].set_ylabel('KL divergence (bits)')
    axes2[0, 1].set_title('KL divergence from uniform (lower = better)')
    axes2[1, 0].set_ylabel('Fano factor $F$')
    axes2[1, 0].set_title('Photon number excess noise ($F=1$: Poissonian)')
    axes2[1, 1].set_ylabel('Intensity CV (%)')
    axes2[1, 1].set_title('Pulse-to-pulse intensity fluctuation')

    for ax in axes2.flat:
        ax.set_xlabel('Repetition rate (GHz)')
        ax.set_xticks(f_ghz)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig2.savefig('images/qkd_source/source_metrics.png', dpi=150)
    print("  Saved: images/qkd_source/source_metrics.png")

    # ── Figure 3: Intensity autocorrelation ──────────────────────────────

    sel_freqs = [1, 3, 5, 10]
    fig3, axes3 = plt.subplots(2, len(sel_freqs), figsize=(20, 8))
    fig3.suptitle(
        'Pulse Intensity Autocorrelation — Side-Channel Diagnostic\n'
        'Non-zero values at lag $>$ 0 leak information between pulses  '
        '(gray dashed = 95% CI for white noise)',
        fontsize=12)

    ci95 = 1.96 / np.sqrt(N_PULSES)

    for col, fg in enumerate(sel_freqs):
        for row, (case, color, label) in enumerate([
            ('free', 'C0', 'Free-running'),
            ('sld', 'C3', 'SLD-injected'),
        ]):
            key = f"{fg}GHz_{case}"
            d = data[key]
            ax = axes3[row, col]

            ax.bar(d['lags'][1:], d['acf'][1:], color=color, alpha=0.7,
                   width=0.8)
            ax.axhline(0, color='k', lw=0.5)
            ax.axhline( ci95, color='gray', ls='--', lw=0.8, alpha=0.5)
            ax.axhline(-ci95, color='gray', ls='--', lw=0.8, alpha=0.5)
            ax.set_title(f'{fg} GHz — {label}', fontsize=10)
            ymax = max(0.02, np.max(np.abs(d['acf'][1:])) * 1.5)
            ax.set_ylim(-ymax * 0.3, ymax)
            ax.grid(True, alpha=0.2)
            if col == 0:
                ax.set_ylabel('Autocorrelation')

        axes3[1, col].set_xlabel('Lag (pulses)')

    plt.tight_layout()
    fig3.savefig('images/qkd_source/intensity_correlations.png', dpi=150)
    print("  Saved: images/qkd_source/intensity_correlations.png")

    # ── Figure 4: Secure key rate ────────────────────────────────────────

    # Channel parameters
    ALPHA_FIBER = 0.2       # dB/km
    ETA_DET     = 0.10      # detector efficiency
    P_DARK      = 1e-6      # dark count prob per gate
    E_MIS       = 0.01      # optical misalignment error
    MU_S        = 0.5       # signal intensity
    MU_W        = 0.1       # weak decoy intensity
    F_EC        = 1.16      # error correction efficiency
    R1_SECURE   = 0.01      # r1 threshold for secure phase randomisation

    distances = np.linspace(0, 200, 500)

    fig4, axes4 = plt.subplots(1, 3, figsize=(21, 6))
    fig4.suptitle(
        'BB84 Decoy-State Secure Key Rate\n'
        f'$\\mu_s$={MU_S}, $\\mu_w$={MU_W}, '
        f'$\\eta_{{det}}$={ETA_DET}, $p_{{dark}}$={P_DARK:.0e}, '
        f'$f_{{EC}}$={F_EC}   |   '
        f'Security requires $r_1 < {R1_SECURE}$',
        fontsize=12)

    # (a) Key rate vs distance  — free vs SLD at selected frequencies
    for case, color_base, ls, panel in [
        ('free', 'Blues', '-',  axes4[0]),
        ('sld',  'Reds',  '-',  axes4[1]),
    ]:
        cmap = plt.get_cmap(color_base)
        for i, fg in enumerate(sel_freqs):
            key = f"{fg}GHz_{case}"
            d = data[key]
            f_rep = d['f_rep']
            r1 = d['r1']
            secure = r1 < R1_SECURE

            rates = np.array([
                bb84_decoy_key_rate(
                    MU_S, MU_W,
                    10**(-ALPHA_FIBER * L / 10),
                    ETA_DET, P_DARK, E_MIS, F_EC
                ) * f_rep
                for L in distances
            ])

            color = cmap(0.3 + 0.6 * i / (len(sel_freqs) - 1))
            lw = 2 if secure else 1.2
            alpha = 1.0 if secure else 0.35
            style = '-' if secure else ':'
            sec_tag = '' if secure else ' [INSECURE]'

            mask = rates > 0
            if np.any(mask):
                panel.semilogy(distances[mask], rates[mask] / 1e6,
                               ls=style, color=color, lw=lw, alpha=alpha,
                               label=f'{fg} GHz  r$_1$={r1:.3f}{sec_tag}')

        panel.set_xlabel('Distance (km)')
        panel.set_ylabel('Key rate (Mbit/s)')
        title = 'Free-running' if case == 'free' else 'SLD-injected'
        panel.set_title(title)
        panel.legend(fontsize=8, loc='upper right')
        panel.grid(True, alpha=0.3, which='both')
        panel.set_xlim(0, 200)

    # (c) Effective secure rate vs f_rep at fixed distances
    ax = axes4[2]
    fixed_dists = [20, 50, 100]
    markers = ['o', 's', '^']

    for L, mk in zip(fixed_dists, markers):
        eta_ch = 10**(-ALPHA_FIBER * L / 10)
        R_pp = bb84_decoy_key_rate(MU_S, MU_W, eta_ch,
                                    ETA_DET, P_DARK, E_MIS, F_EC)

        for case, color, ls, label_case in [
            ('free', 'C0', '-', 'Free'),
            ('sld',  'C3', '--', 'SLD'),
        ]:
            rates_eff = []
            for fg in f_ghz:
                key = f"{fg:.0f}GHz_{case}"
                r1 = data[key]['r1']
                f_rep = data[key]['f_rep']
                if r1 < R1_SECURE:
                    rates_eff.append(R_pp * f_rep / 1e6)
                else:
                    rates_eff.append(0)

            ax.plot(f_ghz, rates_eff, f'{mk}{ls}', color=color,
                    lw=2, ms=7, alpha=0.8,
                    label=f'{label_case} @ {L} km')

    ax.set_xlabel('Repetition rate (GHz)')
    ax.set_ylabel('Secure key rate (Mbit/s)')
    ax.set_title('Effective secure rate\n(0 if $r_1 > 0.01$)')
    ax.set_xticks(f_ghz)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig4.savefig('images/qkd_source/key_rate.png', dpi=150)
    print("  Saved: images/qkd_source/key_rate.png")

    # ── Figure 5: Photon number distribution ─────────────────────────────

    fig5, axes5 = plt.subplots(2, len(sel_freqs), figsize=(20, 8))
    fig5.suptitle(
        f'Photon Number Distribution ($\\mu$ = {MU_S}) — Compound Poisson\n'
        'Deviations from ideal Poisson (gray) reveal intensity fluctuations',
        fontsize=12)

    for col, fg in enumerate(sel_freqs):
        for row, (case, color, label) in enumerate([
            ('free', 'C0', 'Free-running'),
            ('sld', 'C3', 'SLD-injected'),
        ]):
            key = f"{fg}GHz_{case}"
            d = data[key]
            ax = axes5[row, col]

            n_show = min(len(d['pn']), 8)
            n_vals = np.arange(n_show)

            # Ideal Poisson
            import math
            pn_ideal = np.exp(-MU_S) * MU_S**n_vals
            for i in range(n_show):
                pn_ideal[i] /= float(math.factorial(i))

            width = 0.35
            ax.bar(n_vals - width / 2, d['pn'][:n_show], width,
                   color=color, alpha=0.7,
                   label=f'{label} ($F$={d["fano"]:.4f})')
            ax.bar(n_vals + width / 2, pn_ideal, width,
                   color='gray', alpha=0.5, label='Poisson')

            ax.set_title(f'{fg} GHz — {label}', fontsize=10)
            ax.set_xticks(n_vals)
            if col == 0:
                ax.set_ylabel('$P(n)$')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.2)

        axes5[1, col].set_xlabel('Photon number $n$')

    plt.tight_layout()
    fig5.savefig('images/qkd_source/photon_number_distribution.png', dpi=150)
    print("  Saved: images/qkd_source/photon_number_distribution.png")

    plt.close('all')

    # ── Summary table ────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("  QKD Source Summary")
    print("=" * 70)
    print(f"  {'f_rep':>5s}  {'case':8s}  {'r1':>7s}  {'KL':>8s}  "
          f"{'Fano':>7s}  {'CV':>6s}  {'ACF(1)':>9s}  {'secure':>7s}")
    print(f"  {'(GHz)':>5s}  {'':8s}  {'':>7s}  {'(bits)':>8s}  "
          f"{'':>7s}  {'(%)':>6s}  {'':>9s}  {'':>7s}")
    print("  " + "-" * 64)

    for fg in f_ghz:
        for case in ['free', 'sld']:
            key = f"{fg:.0f}GHz_{case}"
            d = data[key]
            secure = 'YES' if d['r1'] < R1_SECURE else 'no'
            print(f"  {fg:5.0f}  {case:8s}  "
                  f"{d['r1']:7.4f}  {d['kl']:8.4f}  "
                  f"{d['fano']:7.4f}  {d['cv']*100:6.2f}  "
                  f"{d['acf'][1]:9.5f}  {secure:>7s}")

    # Phase randomisation verdict
    print(f"\n  Phase Randomisation Assessment (threshold r1 < {R1_SECURE}):")
    for fg in f_ghz:
        kf = f"{fg:.0f}GHz_free"
        ks = f"{fg:.0f}GHz_sld"
        r1_f = data[kf]['r1']
        r1_s = data[ks]['r1']
        vf = 'RANDOM' if r1_f < 0.01 else ('PARTIAL' if r1_f < 0.1 else 'CORREL.')
        vs = 'RANDOM' if r1_s < 0.01 else ('PARTIAL' if r1_s < 0.1 else 'CORREL.')
        print(f"    {fg:.0f} GHz:  free -> {vf:8s} (r1={r1_f:.4f})  |  "
              f"sld -> {vs:8s} (r1={r1_s:.4f})")

    # Maximum secure repetition rate
    max_f_free = 0
    max_f_sld  = 0
    for fg in f_ghz:
        if data[f"{fg:.0f}GHz_free"]['r1'] < R1_SECURE:
            max_f_free = fg
        if data[f"{fg:.0f}GHz_sld"]['r1'] < R1_SECURE:
            max_f_sld = fg

    print(f"\n  Maximum secure repetition rate:")
    print(f"    Free-running: {max_f_free:.0f} GHz")
    print(f"    SLD-injected: {max_f_sld:.0f} GHz")

    if max_f_sld > max_f_free:
        speedup = max_f_sld / max(max_f_free, 1)
        print(f"    -> SLD extends secure range by {speedup:.0f}x")

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)
