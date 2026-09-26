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
    intensity_distributions.png  noise, offset, and both, as panels (a)-(c) (Fig. 6.4)
    SNRmeas_variance.png      true / estimated noise and measured variance
    estimatedSNR.png          true against inferred SNR
    noise_recovery.png        noise estimate and inferred signal variance (Fig. 6.5)
    noise_recovery_orders.png the same, one row per order m = 2, 3, 4
    gm_noise.png              g2, g3, g4 bias from noise            (TODO 2)
    gm_noise_shift.png        g2, g3, g4 bias from noise + offset   (Fig. 6.6)
    correction_efficacy.png   raw against noise-corrected estimates (Fig. 6.7)

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
from gsdfb import save_fig

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


# ── Figures ──────────────────────────────────────────────────────────────────

# Style of the Chapter 6 figures, shared with studies/plots.ipynb (its style cell
# at the top): inward and minor ticks, grid, 18 pt text.  Applied only inside the
# Fig. 6.4 to 6.7 functions, so the other diagnostic figures keep their layout.
NB_STYLE = {
    'font.size': 18, 'axes.labelsize': 18, 'axes.titlesize': 18, 'figure.labelsize': 18,
    'xtick.labelsize': 18, 'ytick.labelsize': 18, 'legend.fontsize': 16,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.size': 6, 'ytick.major.size': 6,
    'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
    'xtick.minor.size': 3, 'ytick.minor.size': 3,
    'xtick.minor.width': 1.0, 'ytick.minor.width': 1.0,
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
    'axes.grid': True,
    'axes.formatter.use_mathtext': True,
    'figure.dpi': 150, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
}

# ── Main ─────────────────────────────────────────────────────────────────────


def plot_distributions(I_sig, offset, save, snr_demo=5.0, seed=7, bins=200,
                       zbins=150, headroom=1.35, xscale=1e23):
    """Fig. 6.4: intensity histograms with noise, baseline shift, and both.

    One figure of three panels sharing a single pair of axis labels.  The demo
    noise is set at 5 dB, which widens the distribution by ~15 %; panel (a) is
    zoomed and drawn with translucent fills plus step outlines so the two
    distributions stay distinguishable.  The same figure is produced by the
    Chapter 6 cell of studies/plots.ipynb.
    """
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Patch
    V_e_demo = np.var(I_sig) / 10.0 ** (snr_demo / 10.0)
    noise_d = np.random.default_rng(seed).normal(0.0, np.sqrt(V_e_demo), I_sig.size)
    # Intensities are divided by xscale, whose power goes in the x label.
    I_noisy = (I_sig + noise_d) / xscale
    I_shifted = (I_sig - offset) / xscale
    I_both = (I_sig - offset + noise_d) / xscale
    I_ref = I_sig / xscale
    shared = np.linspace(min(I_both.min(), I_ref.min()) * 0.98,
                         max(I_noisy.max(), I_ref.max()) * 1.02, bins)

    with plt.rc_context(NB_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.0), constrained_layout=True)

        # (a) noise: zoomed onto the data, light fills plus full-opacity outlines
        ax = axes[0]
        lo, hi = np.percentile(np.concatenate([I_noisy, I_ref]), [0.005, 99.995])
        pad = 0.05 * (hi - lo)
        zb = np.linspace(lo - pad, hi + pad, zbins)
        styles = ((I_ref, 'Noiseless', 'tab:orange', 0.35, '-', 2),
                  (I_noisy, 'Noisy', 'tab:blue', 0.15, '--', 3))
        for data, lab, col, fa, ls, z in styles:
            ax.hist(data, bins=zb, color=col, alpha=fa, zorder=z)
            ax.hist(data, bins=zb, histtype='step', color=col, lw=1.6, ls=ls,
                    zorder=z + 2)
        ax.set_xlim(zb[0], zb[-1])
        ax.legend(handles=[Patch(facecolor=to_rgba(col, fa), edgecolor=col, ls=ls,
                                 lw=1.6, label=lab)
                           for _, lab, col, fa, ls, _ in styles[::-1]])

        # (b) offset and (c) both: two filled histograms on the shared range
        for ax, (a, la), (b, lb) in (
                (axes[1], (I_shifted, 'Shifted'), (I_ref, 'Unshifted')),
                (axes[2], (I_both, 'Noisy and shifted'), (I_ref, 'Ideal'))):
            ax.hist(a, bins=shared, alpha=0.85, label=la)
            ax.hist(b, bins=shared, alpha=0.55, label=lb)
            ax.legend(loc='upper left')

        for ax, tag in zip(axes, 'abc'):
            ax.set_ylim(0, headroom * ax.get_ylim()[1])
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            ax.set_title(f'({tag})', loc='left', fontweight='bold')
        fig.supxlabel(rf'Integrated intensity $I$ ($10^{{{np.log10(xscale):.0f}}}$ arb. units)')
        fig.supylabel('Counts')
        save(fig, 'intensity_distributions.png')


def plot_noise_recovery(snr, V_e_true, V_e_meas, V_meas, V_sig, vs_mean, vs_lo, vs_hi,
                        save, linthresh=1e-4):
    """Fig. 6.5: (a) noise estimate and measured variance, (b) inferred variance.

    Panel (a) is normalised to Var[I_signal] and drawn on a log scale, since the
    noise falls a decade per 10 dB.  Panel (b) is the relative error of the
    signal variance inferred as Var[I] - Var[I_e], with its 95 % interval.  The
    same figure is produced by the Chapter 6 cell of studies/plots.ipynb.
    """
    with plt.rc_context(NB_STYLE):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14.0, 5.5), constrained_layout=True)

        ax_a.plot(snr, V_e_true / V_sig, '-', c='tab:blue', lw=1.4, label='Noise (true)')
        ax_a.plot(snr, V_e_meas / V_sig, 'o', ms=5, c='tab:orange', label='Noise (estimated)')
        ax_a.plot(snr, V_meas / V_sig, 'o', ms=5, c='tab:green', label='Measured')
        ax_a.axhline(1.0, ls='--', c='k', lw=1, label='Signal (noise-free)')
        ax_a.set_yscale('log')
        ax_a.set_xlabel('SNR [dB]')
        ax_a.set_ylabel(r'Variance / $\mathrm{Var}[I_{\rm signal}]$')
        ax_a.legend()

        ax_b.plot(snr, vs_mean, 'o', ms=5, color='tab:green')
        ax_b.fill_between(snr, vs_lo, vs_hi, alpha=0.25, color='tab:green',
                          label='95% interval')
        ax_b.axhline(0.0, ls='--', c='k', lw=1)
        ax_b.set_yscale('symlog', linthresh=linthresh)
        ax_b.set_xlabel('SNR [dB]')
        ax_b.set_ylabel(r'Inferred $\mathrm{Var}[I_{\rm signal}]$, relative error')
        ax_b.legend()

        for ax, tag in ((ax_a, '(a)'), (ax_b, '(b)')):     # as titles, above the axes
            ax.set_title(tag, loc='left', fontweight='bold')
        save(fig, 'noise_recovery.png')



def _plot_gm_pairs(snr, orders, g_true, first, second, labels, save, name):
    """One panel per order: two g^(m)(0) curves against SNR, noise-free value dashed.

    Used for Fig. 6.6, in the notebook style; the x label is common to all panels.
    The same figure is produced by the Chapter 6 cell of studies/plots.ipynb.
    """
    with plt.rc_context(NB_STYLE):
        fig, axes = plt.subplots(1, len(orders), figsize=(16.0, 5.0),
                                 constrained_layout=True, squeeze=False)
        for ax, tag, m in zip(axes[0], 'abcdefgh', orders):
            ax.plot(snr, first[m], 'o', ms=5, c='tab:blue', label=labels[0])
            ax.plot(snr, second[m], 'o', ms=5, c='tab:orange', label=labels[1])
            ax.axhline(g_true[m], ls='--', c='k', lw=1, label=labels[2])
            ax.set_ylabel(f'$g^{{({m})}}(0)$')
            ax.set_title(f'({tag})', loc='left', fontweight='bold')
        axes[0, 0].legend()
        fig.supxlabel('SNR [dB]')
        save(fig, name)


def plot_noise_shift(snr, orders, g_true, g_noise, g_shift, save):
    """Fig. 6.6: uncorrected g^(m)(0) with noise alone and with noise plus offset."""
    _plot_gm_pairs(snr, orders, g_true, g_noise, g_shift,
                   ('Noise', 'Noise and offset', 'Neither'), save, 'gm_noise_shift.png')


def plot_correction_efficacy(snr, orders, excess_bias, corr_mean, corr_lo, corr_hi, save,
                             linthresh=1e-4, tol=0.05):
    """Fig. 6.7: bias of g^(m)(0) as a fraction of the excess, before and after correction.

    excess_bias, corr_mean, corr_lo, corr_hi map order to arrays over the SNR sweep:
    the uncorrected bias (mean over realisations), the corrected bias (mean) and the
    2.5th / 97.5th percentiles of single corrected realisations.  Symlog axis, shared
    across the panels, with one legend above them.  The same figure is produced by the
    Chapter 6 cell of studies/plots.ipynb.
    """
    with plt.rc_context(NB_STYLE):
        fig, axes = plt.subplots(1, len(orders), figsize=(16.0, 5.0),
                                 constrained_layout=True, squeeze=False, sharey=True)
        for ax, tag, m in zip(axes[0], 'abcdefgh', orders):
            ax.plot(snr, excess_bias[m], 'o', ms=5, c='tab:blue', label='Uncorrected')
            ax.plot(snr, corr_mean[m], 'o', ms=5, c='tab:orange', label='Corrected')
            ax.fill_between(snr, corr_lo[m], corr_hi[m], color='tab:orange',
                            alpha=0.25, lw=0, label='95% interval')
            ax.axhline(0.0, ls='--', c='k', lw=1)
            if tol is not None:
                ax.axhline(tol, ls=':', c='k', lw=1.2, label=f'{tol:.0%} of excess')
            ax.set_yscale('symlog', linthresh=linthresh)
            ax.set_title(f'({tag})  $m = {m}$', loc='left', fontweight='bold')
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='outside upper center',
                   ncol=4, frameon=False)
        fig.supxlabel('SNR [dB]')
        fig.supylabel(r'$\Delta g^{(m)} / (g^{(m)}-1)$')
        save(fig, 'correction_efficacy.png')


def gaussian_noise_excess(m, V_e, mean_I, g_sig):
    """Contribution of zero-mean Gaussian noise of variance V_e to g^(m) - 1.

    Expanding <(I + e)^m> with e independent of I and Gaussian,

        <(I+e)^m> - <I^m> = sum_{k even >= 2} C(m,k) <I^(m-k)> (k-1)!! V_e^(k/2),

    and dividing by <I>^m gives the shift in the estimator.  g_sig maps each
    order j < m to the noise-free g^(j)(0) (g^(0) = g^(1) = 1 are filled in).
    For m = 2 this is V_e / <I>^2, i.e. V_e / Var[I_signal] times (g2 - 1).
    """
    from math import comb
    g = {0: 1.0, 1: 1.0, **g_sig}
    r = np.asarray(V_e, dtype=float) / mean_I ** 2
    out = np.zeros_like(r)
    for k in range(2, m + 1, 2):
        out = out + comb(m, k) * g[m - k] * np.prod(np.arange(k - 1, 0, -2)) * r ** (k // 2)
    return out


def plot_noise_recovery_orders(snr, orders, g_true, g_noise, g_corr, V_e_true, mean_I,
                               corr_mean, corr_lo, corr_hi, save, linthresh=1e-4,
                               row_height=5.0, name='noise_recovery_orders.png'):
    """Fig. 6.5 generalised to higher orders: one row of (a), (b) per order m.

    Every quantity is expressed relative to the noise-free excess g^(m) - 1,
    which for m = 2 is Var[I_signal] / <I>^2, so the m = 2 row reproduces
    Fig. 6.5 in the excess form.
      (a) Measured: g^(m)_meas - 1.  Noise (true): the shift that the injected
          Gaussian noise produces, gaussian_noise_excess().  Noise (estimated):
          the shift the pre-pulse correction removes, g^(m)_meas - g^(m)_corr.
      (b) Relative error of the corrected excess, with its 95 % interval.
    g_true maps order to the noise-free value; g_noise, g_corr, corr_* map
    order to arrays over the SNR sweep.
    """
    with plt.rc_context(NB_STYLE):
        n = len(orders)
        fig, axes = plt.subplots(n, 2, figsize=(14.0, row_height * n), sharex=True,
                                 constrained_layout=True, squeeze=False)
        tags = iter('abcdefghijklmnop')
        for (ax_a, ax_b), m in zip(axes, orders):
            ex = g_true[m] - 1.0
            ax_a.plot(snr, gaussian_noise_excess(m, V_e_true, mean_I, g_true) / ex, '-',
                      c='tab:blue', lw=1.4, label='Noise (true)')
            ax_a.plot(snr, (g_noise[m] - g_corr[m]) / ex, 'o', ms=5, c='tab:orange',
                      label='Noise (estimated)')
            ax_a.plot(snr, (g_noise[m] - 1.0) / ex, 'o', ms=5, c='tab:green', label='Measured')
            ax_a.axhline(1.0, ls='--', c='k', lw=1, label='Signal (noise-free)')
            ax_a.set_yscale('log')
            ax_a.set_ylabel(rf'Excess of $g^{{({m})}}$ / $(g^{{({m})}}_{{\rm signal}}-1)$')

            ax_b.plot(snr, corr_mean[m], 'o', ms=5, color='tab:green')
            ax_b.fill_between(snr, corr_lo[m], corr_hi[m], alpha=0.25, color='tab:green',
                              label='95% interval')
            ax_b.axhline(0.0, ls='--', c='k', lw=1)
            ax_b.set_yscale('symlog', linthresh=linthresh)
            ax_b.set_ylabel(rf'Corrected $g^{{({m})}}-1$, relative error')

            for ax in (ax_a, ax_b):     # as titles, clear of the legend and the data
                ax.set_title(f'({next(tags)})', loc='left', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 1].legend()
        for ax in axes[-1]:
            ax.set_xlabel('SNR [dB]')
        save(fig, name)

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
        save_fig(fig, p, bbox_inches=None, pad_inches=None)
        plt.close(fig)
        print(f"  Saved: {p}")

    plot_distributions(I_sig, offset, save)

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

    plot_noise_recovery(SNR_DB, mean['V_e_true'], mean['V_e_meas'], mean['V_meas'],
                        V_sig, vs_mean, vs_lo, vs_hi, save)
    plot_noise_recovery_orders(
        SNR_DB, ORDERS, g_true,
        {m: mean[f'g{m}_noise'] for m in ORDERS}, {m: mean[f'g{m}_corr'] for m in ORDERS},
        mean['V_e_true'], mean_sig,
        {m: corr_bias[m][0] for m in ORDERS}, {m: corr_bias[m][1] for m in ORDERS},
        {m: corr_bias[m][2] for m in ORDERS}, save)

    # g^(m) bias, noise only
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), constrained_layout=True)
    for ax, m in zip(axes, ORDERS):
        ax.plot(SNR_DB, mean[f'g{m}_noise'], '.', label=f'$g^{{({m})}}$')
        ax.axhline(g_true[m], ls='--', c='k', lw=1, label='No noise')
        ax.set_xlabel('SNR [dB]')
        ax.set_ylabel(f'$g^{{({m})}}(0)$')
        ax.legend()
    save(fig, 'gm_noise.png')

    # g^(m) bias, noise and offset (Fig. 6.6)
    plot_noise_shift(SNR_DB, ORDERS, g_true, {m: mean[f'g{m}_noise'] for m in ORDERS},
                     {m: mean[f'g{m}_shift'] for m in ORDERS}, save)

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

    # Efficacy of the noise correction (Fig. 6.7)
    plot_correction_efficacy(SNR_DB, ORDERS, excess_bias,
                             {m: corr_bias[m][0] for m in ORDERS},
                             {m: corr_bias[m][1] for m in ORDERS},
                             {m: corr_bias[m][2] for m in ORDERS}, save)

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
