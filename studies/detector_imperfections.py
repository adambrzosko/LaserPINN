"""
Impact of detector imperfections on the single-photodiode g^(m)(0) estimator.

Backs Section 6.4 of the thesis.  The single-detector method estimates the
m-th order coherence from the moments of the integrated pulse intensity,

    g^(m)_S(0) ~= <I^m> / <I>^m ,

so every fluctuation the detection chain adds to I biases the estimate.  This
study quantifies two imperfections by simulation:

  1. Additive electronic noise, applied per sample as N(0, sigma_e^2).
  2. A baseline (DC setpoint) error, applied as a constant offset per sample.

Method
------
A gain-switched DFB pulse train is simulated once with the stochastic rate
equations (Langevin noise included), and the integrated intensity of each
pulse, I_signal(j) = sum_k S_k(j), is retained.  Imperfections are then
applied analytically at the level of the integrated quantity, which is exact
and avoids storing N_pulses x N_s samples:

  - N_s independent per-sample perturbations of variance sigma_e^2 contribute
    a variance N_s * sigma_e^2 to the integral, so the integrated noise is a
    single draw from N(0, N_s sigma_e^2).
  - Subtracting delta * S_bar from every sample subtracts N_s * delta * S_bar
    = delta * <I> from the integral.

Defining the SNR as the ratio of integrated variances,

    SNR_dB = 10 log10( Var[I_signal] / Var[I_e] ),

gives Var[I_e] = Var[I_signal] / 10^(SNR/10), in which N_s cancels.  The
results are therefore independent of the number of samples per window, which
removes the ambiguity in the original analysis code.

Outputs (images/detector_imperfections/):
    noisy_noiseless.png       intensity distribution with and without noise
    shifted_unshifted.png     intensity distribution with and without offset
    noisy_shifted.png         both imperfections combined
    SNRmeas_variance.png      true / estimated noise and measured variance
    estimatedSNR.png          true against inferred SNR
    noise_recovery.png        quantitative test of the noise estimate (TODO 1)
    gm_noise.png              g2, g3, g4 bias from noise            (TODO 2)
    gm_noise_shift.png        g2, g3, g4 bias from noise + offset   (TODO 2)
    correction_efficacy.png   raw against noise-corrected estimates

Run from the Simulations/ root:
    python3 -m studies.detector_imperfections
"""
import os
import numpy as np
import time as _time
from numba import njit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core.dfb_laser import DFBLaserParams, q


# ── Solver: integrated intensity per pulse ───────────────────────────────────

@njit
def simulate_integrated(n_pulses, n_discard, pts_period, dt,
                        I_off, I_on, t_on, t_rise,
                        V, Gamma, v_g, a_gain, N_tr, eps,
                        A, B, C_aug, tau_p, beta_sp, alpha_H, q_e,
                        seed):
    """Gain-switched DFB, returning the integrated intensity of each pulse.

    Same field-domain Euler-Maruyama scheme as
    core.million_pulse_comparison.simulate_pulses (E = E_r + i E_i, with
    spontaneous emission entering as a complex Gaussian increment and the
    carrier equation carrying its own Langevin term), but accumulating
    sum_k S_k over the period instead of extracting the peak.
    """
    np.random.seed(seed)
    sqrt_half_dt = np.sqrt(dt / 2.0)

    n_out = n_pulses - n_discard
    integrated = np.empty(n_out)

    N = N_tr * 1.2
    Er = np.sqrt(1e10)
    Ei = 0.0

    for p in range(n_pulses):
        acc = 0.0

        for k in range(pts_period):
            t = k * dt

            if t < t_rise:
                f = 0.5 * (1.0 - np.cos(np.pi * t / t_rise))
                I_cur = I_off + (I_on - I_off) * f
            elif t < t_on - t_rise:
                I_cur = I_on
            elif t < t_on:
                f = 0.5 * (1.0 - np.cos(np.pi * (t_on - t) / t_rise))
                I_cur = I_off + (I_on - I_off) * f
            else:
                I_cur = I_off

            S = Er * Er + Ei * Ei
            if S < 1e-10:
                S = 1e-10

            g = a_gain * (N - N_tr) / (1.0 + eps * S)
            R_sp = A * N + B * N * N + C_aug * N * N * N
            R_sp_mode = beta_sp * B * N * N

            gamma = 0.5 * (Gamma * v_g * g - 1.0 / tau_p)

            Er_new = (1.0 + gamma * dt) * Er - gamma * alpha_H * dt * Ei
            Ei_new = (1.0 + gamma * dt) * Ei + gamma * alpha_H * dt * Er
            Er = Er_new
            Ei = Ei_new

            amp_sp = np.sqrt(R_sp_mode) * sqrt_half_dt
            Er += amp_sp * np.random.randn()
            Ei += amp_sp * np.random.randn()

            dN = (I_cur / (q_e * V) - R_sp - Gamma * v_g * g * S) * dt
            FN = np.sqrt(2.0 * R_sp * dt) * np.random.randn()
            N += dN + FN

            acc += Er * Er + Ei * Ei

        if p >= n_discard:
            integrated[p - n_discard] = acc

    return integrated


# ── Estimator and noise correction ───────────────────────────────────────────

def g_m(I, m):
    """Single-photodiode estimator g^(m)_S(0) = <I^m> / <I>^m."""
    return np.mean(I ** m) / np.mean(I) ** m


def correct(I, I_e):
    """Second-order noise correction, Eqs. (6.13)-(6.14) of the thesis.

    I_corr = eta (I + x), with eta = sqrt(1 - V_e/V_I) and
    x = [<I>(1-eta) - <I_e>] / eta.  Returns None where V_e >= V_I, for
    which the correction has no real solution.
    """
    V_I, V_e = np.var(I), np.var(I_e)
    if V_e >= V_I:
        return None
    eta = np.sqrt(1.0 - V_e / V_I)
    x = (np.mean(I) * (1.0 - eta) - np.mean(I_e)) / eta
    return eta * (I + x)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    OUT = 'images/detector_imperfections'
    os.makedirs(OUT, exist_ok=True)

    print("=" * 70)
    print("  Detector imperfections and the single-photodiode estimator")
    print("=" * 70)

    laser = DFBLaserParams()
    I_th = laser.threshold_current()

    # Operating point: 1 GHz gain switching, matching the Chapter 6 experiment.
    N_PULSES = int(os.environ.get('DI_PULSES', 1_000_000))
    N_DISCARD = 1_000
    F_REP = 1.0e9
    DT = 1.0e-12
    DUTY = 0.30
    # Operating point chosen to reproduce the 19 mA setting of Table 6.2,
    # the near-Poissonian source on which the experimental SNR sweep of
    # Section 6.6.3 was performed.  Gives g2 = 1.00057, g3 = 1.0017,
    # g4 = 1.0034 against the measured 1.0007, 1.002, 1.004.
    I_BIAS_FACTOR = 0.95
    I_PEAK_FACTOR = 1.60

    DELTA = 0.30          # baseline offset, as a fraction of the mean amplitude
    ORDERS = (2, 3, 4)
    SNR_DB = np.arange(-4.0, 40.1, float(os.environ.get('DI_STEP', 1.0)))
    N_REAL = int(os.environ.get('DI_REAL', 40))   # realisations per SNR, for CIs
    RNG = np.random.default_rng(20260903)

    I_off = I_BIAS_FACTOR * I_th
    I_on = I_PEAK_FACTOR * I_th
    T_rep = 1.0 / F_REP
    t_on = DUTY * T_rep
    N_s = int(round(T_rep / DT))          # samples spanning the window
    dt = T_rep / N_s
    t_rise = min(20e-12, t_on / 4.0)

    print(f"\n  I_th = {I_th*1e3:.2f} mA, bias = {I_BIAS_FACTOR}*I_th, "
          f"peak = {I_PEAK_FACTOR}*I_th")
    print(f"  f_rep = {F_REP/1e9:.0f} GHz, N_s = {N_s} samples/window, "
          f"dt = {dt*1e12:.2f} ps")
    print(f"  {N_PULSES/1e6:.0f}M pulses, {len(SNR_DB)} SNR points, "
          f"{N_REAL} realisations each")

    print("\n  Compiling JIT solver...")
    t0 = _time.time()
    _ = simulate_integrated(50, 5, 100, 1e-12, I_off, I_on, 30e-12, 7e-12,
                            laser.V, laser.Gamma, laser.v_g, laser.a,
                            laser.N_tr, laser.epsilon, laser.A, laser.B,
                            laser.C, laser.tau_p, laser.beta_sp,
                            laser.alpha_H, q, 0)
    del _
    print(f"  Compiled in {_time.time()-t0:.1f} s")

    # The pulse train is expensive and depends on none of the sweep settings,
    # so it is cached.  Delete the .npy to force a re-simulation.
    cache = os.path.join(OUT, f'I_signal_{N_PULSES}_{I_BIAS_FACTOR}'
                              f'_{I_PEAK_FACTOR}.npy')
    if os.path.exists(cache):
        I_sig = np.load(cache)
        print(f"  Loaded cached pulse train: {cache}")
    else:
        print("  Simulating pulse train ...", end="", flush=True)
        t0 = _time.time()
        I_sig = simulate_integrated(
            N_PULSES + N_DISCARD, N_DISCARD, N_s, dt,
            I_off, I_on, t_on, t_rise,
            laser.V, laser.Gamma, laser.v_g, laser.a, laser.N_tr,
            laser.epsilon, laser.A, laser.B, laser.C, laser.tau_p,
            laser.beta_sp, laser.alpha_H, q, 20260903)
        print(f" {_time.time()-t0:.1f} s")
        np.save(cache, I_sig)
        if os.environ.get('DI_CACHE_ONLY'):
            print("  Cache written, stopping (DI_CACHE_ONLY set).")
            raise SystemExit(0)

    V_sig = np.var(I_sig)
    mean_sig = np.mean(I_sig)
    g_true = {m: g_m(I_sig, m) for m in ORDERS}

    print(f"\n  <I> = {mean_sig:.4e},  Var[I] = {V_sig:.4e}")
    print("  Noise-free source:")
    for m in ORDERS:
        print(f"    g^({m})(0) = {g_true[m]:.6f}")

    # ── SNR sweep ────────────────────────────────────────────────────────────

    n_snr = len(SNR_DB)
    res = {k: np.zeros((n_snr, N_REAL)) for k in
           ('V_meas', 'V_e_meas', 'V_e_true', 'snr_est')}
    for m in ORDERS:
        res[f'g{m}_noise'] = np.zeros((n_snr, N_REAL))
        res[f'g{m}_shift'] = np.zeros((n_snr, N_REAL))
        res[f'g{m}_corr'] = np.full((n_snr, N_REAL), np.nan)

    offset = DELTA * mean_sig      # = N_s * delta * S_bar

    print(f"\n  Sweeping SNR ...", end="", flush=True)
    t0 = _time.time()
    for i, snr in enumerate(SNR_DB):
        V_e = V_sig / 10.0 ** (snr / 10.0)
        sd = np.sqrt(V_e)
        for r in range(N_REAL):
            noise = RNG.normal(0.0, sd, I_sig.size)
            I_e = RNG.normal(0.0, sd, I_sig.size)   # pre-pulse noise record
            I_meas = I_sig + noise
            I_shift = I_sig - offset + noise

            res['V_meas'][i, r] = np.var(I_meas)
            res['V_e_meas'][i, r] = np.var(I_e)
            res['V_e_true'][i, r] = V_e
            V_sig_est = np.var(I_meas) - np.var(I_e)
            res['snr_est'][i, r] = (10.0 * np.log10(V_sig_est / np.var(I_e))
                                    if V_sig_est > 0 else np.nan)

            I_corr = correct(I_meas, I_e)
            for m in ORDERS:
                res[f'g{m}_noise'][i, r] = g_m(I_meas, m)
                res[f'g{m}_shift'][i, r] = g_m(I_shift, m)
                if I_corr is not None:
                    res[f'g{m}_corr'][i, r] = g_m(I_corr, m)
    print(f" {_time.time()-t0:.1f} s")

    mean = {k: np.nanmean(v, axis=1) for k, v in res.items()}

    # ── Quantitative behaviour of the estimator (Section 6.4) ────────────────
    #
    # For additive zero-mean noise independent of the signal, <I_meas> = <I>
    # and Var[I_meas] = Var[I] + Var[I_e], so the second-order estimator obeys
    #
    #     g2_meas - 1 = (Var[I] + Var[I_e]) / <I>^2 ,
    #     g2_true - 1 =  Var[I]             / <I>^2 ,
    #
    # and therefore the bias, expressed as a fraction of the excess the
    # measurement is trying to resolve, is exactly
    #
    #     (g2_meas - g2_true) / (g2_true - 1) = Var[I_e]/Var[I] = 10^(-SNR/10),
    #
    # independent of the source.  This turns a qualitative "agrees above about
    # 13 dB" into an exact statement: 13 dB is a 5% criterion.  The law is
    # checked numerically below and the higher orders are measured against it.

    excess_bias = {m: (mean[f'g{m}_noise'] - g_true[m]) / (g_true[m] - 1.0)
                   for m in ORDERS}
    analytic = 10.0 ** (-SNR_DB / 10.0)
    dev2 = np.abs(excess_bias[2] - analytic) / analytic

    # Inferred signal variance: a small difference of two large numbers at low
    # SNR, which is what actually limits the correction.
    V_sig_est = res['V_meas'] - res['V_e_meas']
    vs_err = (V_sig_est - V_sig) / V_sig
    vs_mean = vs_err.mean(axis=1)
    vs_lo = np.percentile(vs_err, 2.5, axis=1)
    vs_hi = np.percentile(vs_err, 97.5, axis=1)

    corr_bias = {}
    for m in ORDERS:
        cb = (res[f'g{m}_corr'] - g_true[m]) / (g_true[m] - 1.0)
        corr_bias[m] = (np.nanmean(cb, axis=1),
                        np.nanpercentile(cb, 2.5, axis=1),
                        np.nanpercentile(cb, 97.5, axis=1))

    TOL = 0.05      # 5% of the excess

    def threshold(curve):
        for k_ in range(n_snr):
            if np.all(np.abs(curve[k_:]) < TOL):
                return SNR_DB[k_]
        return np.nan

    thr_raw = {m: threshold(excess_bias[m]) for m in ORDERS}
    thr_corr = {m: threshold(corr_bias[m][0]) for m in ORDERS}

    print("\n" + "-" * 78)
    print("  Bias as a fraction of the excess (g^(m) - 1)")
    print("-" * 78)
    print(f"  {'SNR':>5} {'analytic':>10} " +
          " ".join(f"{'g%d raw' % m:>10}" for m in ORDERS) + " " +
          " ".join(f"{'g%d corr' % m:>10}" for m in ORDERS))
    for k_, snr in enumerate(SNR_DB):
        if snr % 5 == 0:
            print(f"  {snr:5.0f} {analytic[k_]:10.2e} " +
                  " ".join(f"{excess_bias[m][k_]:10.2e}" for m in ORDERS) + " " +
                  " ".join(f"{corr_bias[m][0][k_]:10.2e}" for m in ORDERS))

    print(f"\n  Analytic law (g2): max relative deviation = "
          f"{np.nanmax(dev2)*100:.2f}%, median = {np.nanmedian(dev2)*100:.2f}%")
    print(f"  SNR for bias < {TOL*100:.0f}% of the excess:")
    for m in ORDERS:
        print(f"    m = {m}:  uncorrected {thr_raw[m]:5.0f} dB     "
              f"corrected {thr_corr[m]:5.0f} dB")

    print("\n" + "-" * 78)
    print("  Inferred signal variance, relative error")
    print("-" * 78)
    for k_, snr in enumerate(SNR_DB):
        if snr % 5 == 0:
            print(f"  {snr:5.0f} dB   {vs_mean[k_]:+9.2e} "
                  f"[{vs_lo[k_]:+.2e}, {vs_hi[k_]:+.2e}]")

    print("\n" + "-" * 78)
    print(f"  Baseline offset alone (no noise), delta = {DELTA:.2f}")
    print("-" * 78)
    I_sh_only = I_sig - offset
    for m in ORDERS:
        gm_sh = g_m(I_sh_only, m)
        print(f"    g^({m}): {g_true[m]:.6f} -> {gm_sh:.6f}   "
              f"({(gm_sh - g_true[m]) / (g_true[m] - 1.0) * 100:+.1f}% of excess)")

    # ── Figures ──────────────────────────────────────────────────────────────
    # Deliberately plain matplotlib, matching the existing Chapter 6 figures
    # rather than the serif gsdfb house style used in Chapter 5.
    plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 200,
                         'savefig.bbox': 'tight'})

    def save(fig, name):
        p = os.path.join(OUT, name)
        fig.savefig(p)
        plt.close(fig)
        print(f"  Saved: {p}")

    SNR_DEMO = 15.0
    V_e_demo = V_sig / 10.0 ** (SNR_DEMO / 10.0)
    rng_d = np.random.default_rng(7)
    noise_d = rng_d.normal(0.0, np.sqrt(V_e_demo), I_sig.size)
    I_noisy = I_sig + noise_d
    I_shifted = I_sig - offset
    I_both = I_sig - offset + noise_d

    bins = np.linspace(min(I_both.min(), I_sig.min()) * 0.98,
                       max(I_noisy.max(), I_sig.max()) * 1.02, 200)

    for name, (a, la), (b, lb) in [
            ('noisy_noiseless.png', (I_noisy, 'Noisy'), (I_sig, 'Noiseless')),
            ('shifted_unshifted.png', (I_shifted, 'Shifted'), (I_sig, 'Unshifted')),
            ('noisy_shifted.png', (I_both, 'Noisy and shifted'), (I_sig, 'Ideal'))]:
        fig, ax = plt.subplots(figsize=(5.6, 4.2))
        ax.hist(a, bins=bins, alpha=0.85, label=la)
        ax.hist(b, bins=bins, alpha=0.55, label=lb)
        ax.set_xlabel('Integrated intensity $I$ (arb.)')
        ax.set_ylabel('Counts')
        ax.legend()
        save(fig, name)

    # Variance of the measured, estimated-noise and true-noise distributions
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    ax.plot(SNR_DB, mean['V_e_true'], '-', c='tab:blue', lw=1.4,
            label='Noise (true)')
    ax.plot(SNR_DB, mean['V_e_meas'], '.', c='tab:orange',
            label='Noise (estimated)')
    ax.plot(SNR_DB, mean['V_meas'], '.', c='tab:green', label='Measured')
    ax.axhline(V_sig, ls='--', c='k', lw=1, label='Signal (noise-free)')
    ax.set_xlabel('SNR [dB]')
    ax.set_ylabel('Variance')
    ax.legend()
    save(fig, 'SNRmeas_variance.png')

    # True against inferred SNR
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    ax.plot(SNR_DB, SNR_DB, '-', c='tab:blue', lw=1.4, label='SNR set')
    ax.plot(SNR_DB, mean['snr_est'], '.', c='tab:orange', label='SNR inferred')
    ax.set_xlabel('SNR set [dB]')
    ax.set_ylabel('SNR [dB]')
    ax.legend()
    save(fig, 'estimatedSNR.png')

    # Quantitative behaviour: analytic law and the inferred signal variance
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), constrained_layout=True)
    ax = axes[0]
    for m, mk, ms in zip(ORDERS, ('o', 's', '^'), (9, 5, 3)):
        ax.plot(SNR_DB, np.abs(excess_bias[m]), mk, ms=ms, mfc='none',
                label=f'$g^{{({m})}}$')
    ax.plot(SNR_DB, analytic, '-', c='k', lw=1.2,
            label=r'$10^{-\mathrm{SNR}/10}$')
    ax.axhline(TOL, ls=':', c='tab:red', lw=1.2, label=f'{TOL*100:.0f}% of excess')
    ax.set_yscale('log')
    ax.set_xlabel('SNR [dB]')
    ax.set_ylabel(r'$|g^{(m)}_{\rm meas}-g^{(m)}|\,/\,(g^{(m)}-1)$')
    ax.legend()

    ax = axes[1]
    ax.plot(SNR_DB, vs_mean, '.', color='tab:green')
    ax.fill_between(SNR_DB, vs_lo, vs_hi, alpha=0.25, color='tab:green',
                    label='95% interval')
    ax.axhline(0.0, ls='--', c='k', lw=1)
    ax.set_yscale('symlog', linthresh=1e-4)
    ax.set_xlabel('SNR [dB]')
    ax.set_ylabel(r'Inferred $\mathrm{Var}[I_{\rm signal}]$, relative error')
    ax.legend()
    save(fig, 'noise_recovery.png')

    # g^(m) bias, noise only
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), constrained_layout=True)
    for ax, m in zip(axes, ORDERS):
        ax.plot(SNR_DB, mean[f'g{m}_noise'], '.', label=f'$g^{{({m})}}$')
        ax.axhline(g_true[m], ls='--', c='k', lw=1, label='No noise')
        ax.set_xlabel('SNR [dB]')
        ax.set_ylabel(f'$g^{{({m})}}(0)$')
        ax.legend()
    save(fig, 'gm_noise.png')

    # g^(m) bias, noise and offset
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), constrained_layout=True)
    for ax, m in zip(axes, ORDERS):
        ax.plot(SNR_DB, mean[f'g{m}_noise'], '.', label='Noise')
        ax.plot(SNR_DB, mean[f'g{m}_shift'], '.', label='Noise and offset')
        ax.axhline(g_true[m], ls='--', c='k', lw=1, label='Neither')
        ax.set_xlabel('SNR [dB]')
        ax.set_ylabel(f'$g^{{({m})}}(0)$')
        ax.legend()
    save(fig, 'gm_noise_shift.png')

    # Relative bias, all orders on one axis
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    for m, mk, ms in zip(ORDERS, ('o', 's', '^'), (9, 5, 3)):
        ax.plot(SNR_DB, (mean[f'g{m}_noise'] / g_true[m] - 1) * 100, mk,
                ms=ms, mfc='none', label=f'$g^{{({m})}}$')
    ax.axhline(0.0, ls='--', c='k', lw=1)
    ax.set_xlabel('SNR [dB]')
    ax.set_ylabel('Relative bias [%]')
    ax.set_yscale('symlog', linthresh=0.01)
    ax.legend()
    save(fig, 'gm_relative_bias.png')

    # Efficacy of the noise correction
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), constrained_layout=True)
    for ax, m in zip(axes, ORDERS):
        ax.plot(SNR_DB, mean[f'g{m}_noise'], '.', label='Uncorrected')
        ax.plot(SNR_DB, mean[f'g{m}_corr'], '.', label='Corrected')
        ax.axhline(g_true[m], ls='--', c='k', lw=1, label='No noise')
        ax.set_xlabel('SNR [dB]')
        ax.set_ylabel(f'$g^{{({m})}}(0)$')
        ax.legend()
    save(fig, 'correction_efficacy.png')

    np.savez(os.path.join(OUT, 'detector_imperfections.npz'),
             snr_db=SNR_DB, orders=np.array(ORDERS),
             g_true=np.array([g_true[m] for m in ORDERS]),
             analytic=analytic, tol=TOL,
             excess_bias=np.array([excess_bias[m] for m in ORDERS]),
             corr_bias=np.array([corr_bias[m][0] for m in ORDERS]),
             corr_bias_lo=np.array([corr_bias[m][1] for m in ORDERS]),
             corr_bias_hi=np.array([corr_bias[m][2] for m in ORDERS]),
             thr_raw=np.array([thr_raw[m] for m in ORDERS]),
             thr_corr=np.array([thr_corr[m] for m in ORDERS]),
             vs_mean=vs_mean, vs_lo=vs_lo, vs_hi=vs_hi,
             mean_I=mean_sig, var_I=V_sig, delta=DELTA,
             **{k: mean[k] for k in mean})

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)
