"""
Shared analysis functions for gain-switched DFB laser simulations.

Consolidates metrics that were previously duplicated across 13+ scripts:
r1, phase quality, timing jitter, AMZI outputs, the Chapter 5 intensity
autocorrelation with its Bartlett bound, and the arcsine comparison for the
AMZI splitting ratio.
"""
import numpy as np


# ── Phase correlation ────────────────────────────────────────────────────────

def compute_r1(phi):
    """First-order inter-pulse phase correlation.

    Parameters
    ----------
    phi : (N,) array — phase at peak of each pulse

    Returns
    -------
    r1 : float — |<exp(i * dphi)>|, in [0, 1].
        0 = fully random (secure for QKD), 1 = fully correlated.
    dphi : (N-1,) array — wrapped inter-pulse phase differences.
    """
    dphi = np.angle(np.exp(1j * np.diff(phi)))
    r1 = float(np.abs(np.mean(np.exp(1j * dphi))))
    return r1, dphi


def phase_randomisation_quality(phi, n_bins=128):
    """Assess how uniform the inter-pulse phase distribution is.

    Returns
    -------
    dict with keys:
        r1 : float — first-order phase correlation
        kl : float — KL divergence from uniform (bits), 0 = perfect
        ks_stat : float — Kolmogorov-Smirnov statistic vs uniform
        sig_phi : float — std of wrapped phase differences
        dphi : (N-1,) array — wrapped phase differences
    """
    from scipy.stats import kstest

    dphi = np.angle(np.exp(1j * np.diff(phi)))
    r1 = float(np.abs(np.mean(np.exp(1j * dphi))))
    sig_phi = float(np.std(dphi))

    # KL divergence from uniform on [-pi, pi]
    hist, _ = np.histogram(dphi, bins=n_bins, range=(-np.pi, np.pi))
    p = (hist.astype(np.float64) + 1e-10)
    p /= p.sum()
    kl = float(np.sum(p * np.log2(p * n_bins)))

    # KS test vs uniform
    dphi_normalised = (dphi + np.pi) / (2 * np.pi)
    ks_stat, _ = kstest(dphi_normalised[:10000], 'uniform')

    return dict(r1=r1, kl=kl, ks_stat=ks_stat, sig_phi=sig_phi, dphi=dphi)


def compute_metrics(phi, pk_S, pk_k, dt, laser):
    """Compute all QKD-relevant pulse statistics.

    Parameters
    ----------
    phi : (N,) array — peak phase per pulse
    pk_S : (N,) array — peak photon density per pulse
    pk_k : (N,) array — peak time index per pulse (integer)
    dt : float — time step (s)
    laser : DFBLaserParams — for output power conversion

    Returns
    -------
    dict with: r1, sig_phi, sig_t, mean_P, std_P, cv_P, kl, fano
    """
    dphi = np.angle(np.exp(1j * np.diff(phi)))
    r1 = float(np.abs(np.mean(np.exp(1j * dphi))))
    sig_phi = float(np.std(dphi))

    # Timing jitter
    t_peak = pk_k.astype(np.float64) * dt
    sig_t = float(np.std(t_peak))

    # Power statistics
    pk_P = laser.output_power(np.maximum(pk_S, 0))
    mean_P = float(np.mean(pk_P))
    std_P = float(np.std(pk_P))
    cv_P = std_P / mean_P if mean_P > 0 else 0.0

    # Fano factor (intensity)
    fano = float(np.var(pk_S) / np.mean(pk_S)) if np.mean(pk_S) > 0 else 0.0

    # KL divergence from uniform phase
    hist, _ = np.histogram(dphi, bins=128, range=(-np.pi, np.pi))
    p = (hist.astype(np.float64) + 1e-10)
    p /= p.sum()
    kl = float(np.sum(p * np.log2(p * 128)))

    return dict(
        r1=r1, sig_phi=sig_phi, sig_t=sig_t,
        mean_P=mean_P, std_P=std_P, cv_P=cv_P, kl=kl, fano=fano,
    )


# ── Timing jitter ────────────────────────────────────────────────────────────

def absolute_jitter(peak_k, dt):
    """RMS timing jitter: std of peak arrival time within period.

    Returns (sigma_t, t_peak_array).
    """
    t_peak = peak_k.astype(np.float64) * dt
    return float(np.std(t_peak)), t_peak


def period_jitter(peak_k, dt, T_rep):
    """Period jitter: std of peak-to-peak interval.

    Returns (sigma_T, dt_peak_array).
    """
    t_peak = peak_k.astype(np.float64) * dt
    dt_peak = np.diff(t_peak)
    return float(np.std(dt_peak)), dt_peak


def allan_deviation(peak_k, dt, T_rep, max_m=1000):
    """Overlapping Allan deviation of the fractional period, from the arrival times x_n:

        sigma_y^2(m T) = < (x_{n+2m} - 2 x_{n+m} + x_n)^2 > / (2 (m T)^2)

    (Riley, NIST SP 1065, Eq. 12). This equals differencing averages of y = diff(x)/T over
    windows SEPARATED by tau. The previous implementation differenced adjacent, overlapping
    averages (lag 1), which reads sqrt(2/3) low for independent arrival times and turns white
    frequency noise into a tau^-1 slope -- 32x low at m = 1000 -- so noise types could not be
    identified from it.

    Pulses slaved to an RF drive have white arrival-time noise (white PM), for which
    sigma_y = sqrt(3) sigma_t / tau, slope -1. Independent period-to-period fluctuations
    (white FM) give slope -1/2, and a period that random-walks gives slope +1/2.

    Returns (tau_array, adev_array).
    """
    x = peak_k.astype(np.float64) * dt
    n = len(x) - 1
    ms = np.unique(np.geomspace(1, min(n // 4, max_m), 40).astype(int))
    ms = ms[ms >= 1]

    adev = np.zeros(len(ms))
    for i, m in enumerate(ms):
        d2 = x[2 * m:] - 2 * x[m:-m] + x[:-2 * m]
        adev[i] = np.sqrt(0.5 * np.mean(d2 ** 2)) / (m * T_rep)

    tau = ms * T_rep
    return tau, adev


# ── AMZI ─────────────────────────────────────────────────────────────────────

def amzi_outputs(peak_P, peak_phi, psi=0.0):
    """Compute AMZI port intensities for consecutive pulse pairs.

    Parameters
    ----------
    peak_P : (N,) array — peak power of each pulse
    peak_phi : (N,) array — phase at peak of each pulse
    psi : float — additional phase offset in delayed arm

    Returns
    -------
    I_A, I_B : (N-1,) arrays — output port intensities
    eta : (N-1,) array — splitting ratio I_A / (I_A + I_B)
    """
    E = np.sqrt(np.maximum(peak_P, 0)) * np.exp(1j * peak_phi)
    E_n = E[1:]
    E_prev = E[:-1]

    E_A = (E_n + np.exp(1j * psi) * E_prev) / np.sqrt(2)
    E_B = (E_n - np.exp(1j * psi) * E_prev) / np.sqrt(2)

    I_A = np.abs(E_A)**2
    I_B = np.abs(E_B)**2

    total = I_A + I_B
    eta = np.where(total > 0, I_A / total, 0.5)

    return I_A, I_B, eta


def amzi_splitting_ratio(peak_P, peak_phi, n_psi=64):
    """Fringe visibility from sweeping AMZI phase offset.

    Returns (visibility, psi_opt, psi_values, I_A_mean_values).
    """
    E = np.sqrt(np.maximum(peak_P, 0)) * np.exp(1j * peak_phi)
    E_n = E[1:]
    E_prev = E[:-1]

    psi_vals = np.linspace(0, 2 * np.pi, n_psi, endpoint=False)
    I_A_mean = np.zeros(n_psi)

    for i, psi in enumerate(psi_vals):
        E_A = (E_n + np.exp(1j * psi) * E_prev) / np.sqrt(2)
        I_A_mean[i] = np.mean(np.abs(E_A)**2)

    I_max = np.max(I_A_mean)
    I_min = np.min(I_A_mean)
    V = (I_max - I_min) / (I_max + I_min) if (I_max + I_min) > 0 else 0.0
    psi_opt = psi_vals[np.argmax(I_A_mean)]

    return V, psi_opt, psi_vals, I_A_mean


# ── Intensity autocorrelation (Chapter 5 definition) ─────────────────────────

def intensity_autocorrelation(x, max_lag=100):
    """Pearson autocorrelation of an intensity series, as defined in Chapter 5:

        rho(k) = sum_{n=1}^{N-k} (x_n - xbar)(x_{n+k} - xbar)  /  sum_{n=1}^{N} (x_n - xbar)^2

    One denominator over all N samples (the biased estimator), which is what the Bartlett
    bound assumes. This differs from mean-over-(N-k) / variance, used in
    studies/paper_10ghz_simulation.py, by a factor N/(N-k): 1% at k = 100 for N = 10^4.
    Computed through the FFT, identical to the explicit sum.

    Returns (lags, rho) for lags 0..max_lag, with rho[0] = 1.
    """
    c = np.asarray(x, dtype=np.float64)
    c = c - c.mean()
    n = len(c)
    max_lag = int(max(0, min(max_lag, n - 1)))
    f = np.fft.fft(c, n=2 * n)                   # zero padding makes the sum non-circular
    acf = np.fft.ifft(np.abs(f) ** 2).real[:max_lag + 1]
    rho = acf / acf[0] if acf[0] > 0 else np.zeros(max_lag + 1)
    return np.arange(max_lag + 1), rho


def surrogate_acf_envelope(peak_P, max_lag=100, series='I_A', n_surr=8, seed=0, psi=0.0):
    """Per-lag envelope of |rho(k)| that phase randomness alone produces, given these peak powers.

    The port intensity I_A = (P_n + P_{n-1})/2 + sqrt(P_n P_{n-1}) cos(dphi - psi) contains a
    two-term moving average of the pulse energies, so its lag-1 autocorrelation is about CV_P^2/2
    even when the phases are perfectly random -- measured +0.0096 at the Chapter 5 point
    (CV_P = 0.186) against a white-noise Bartlett bound of 0.0058, which would call a random
    source correlated. The white-noise bound is simply the wrong null for I_A; this one redraws
    the phases uniformly with the measured powers held fixed and returns the largest |rho(k)|
    over n_surr surrogates.

    The envelope is |mean_k| + z * sd, where mean_k is the per-lag mean over the surrogates (this
    is what carries the energy leak, concentrated at lag 1) and sd is pooled over lags >= 2, where
    the null is flat. Taking the per-lag MAXIMUM of a handful of surrogates instead is only an
    ~89% bound -- a fresh draw exceeds the max of 8 with probability 1/9 -- and flagged 11 of 100
    lags on a run whose phases were merely partly randomised.

    Returns (lags, envelope) for lags 1..max_lag.
    """
    from scipy.special import erfinv
    P = np.maximum(np.asarray(peak_P, dtype=np.float64), 0.0)
    rng = np.random.default_rng(seed)
    draws = np.empty((n_surr, max_lag))
    for i in range(n_surr):
        I_A, _, eta = amzi_outputs(P, rng.uniform(-np.pi, np.pi, len(P)), psi)
        _, rho = intensity_autocorrelation(eta if series == 'eta' else I_A, max_lag)
        draws[i] = rho[1:max_lag + 1]
    z = np.sqrt(2) * erfinv(0.99)
    spread = draws[:, 1:].std() if max_lag > 1 else draws.std()
    return np.arange(1, max_lag + 1), np.abs(draws.mean(axis=0)) + z * spread


def bartlett_bound(n, confidence=0.99):
    """Chapter 5's bound on an autocorrelation coefficient of n uncorrelated samples:
    sqrt(2) erfinv(confidence) / sqrt(n), i.e. 2.576 / sqrt(n) at 99%."""
    from scipy.special import erfinv
    return float(np.sqrt(2.0) * erfinv(confidence) / np.sqrt(n))


# ── Arcsine comparison for the AMZI splitting ratio ──────────────────────────────────

def arcsine_cdf(x, a=0.0, b=1.0):
    """CDF of the arcsine (Beta(1/2, 1/2)) distribution on [a, b]."""
    u = np.clip((np.asarray(x, dtype=np.float64) - a) / (b - a), 0.0, 1.0)
    return (2.0 / np.pi) * np.arcsin(np.sqrt(u))


def arcsine_bin_density(edges, a=0.0, b=1.0):
    """Arcsine density averaged over each histogram bin. The density itself diverges at
    both ends of the support, so a histogram must be compared with bin averages."""
    edges = np.asarray(edges, dtype=np.float64)
    return np.diff(arcsine_cdf(edges, a, b)) / np.diff(edges)


def amplitude_contrast(peak_P):
    """RMS fringe contrast allowed by pulse-energy imbalance alone:
    sqrt(<v^2>), v = 2 sqrt(P_n P_{n-1}) / (P_n + P_{n-1}) for consecutive pulses."""
    P = np.maximum(np.asarray(peak_P, dtype=np.float64), 0.0)
    p1, p2 = P[:-1], P[1:]
    tot = p1 + p2
    v = np.where(tot > 0, 2.0 * np.sqrt(p1 * p2) / np.where(tot > 0, tot, 1.0), 0.0)
    return float(np.sqrt(np.mean(v ** 2)))


def eta_null_cdf(peak_P, n_grid=2001, n_quantiles=2000, n_edge=32):
    """CDF of the AMZI splitting ratio expected if consecutive phases were independent and
    uniform, GIVEN the simulated pulse energies.

    For consecutive pulses with powers P1, P2 and phase difference dphi,

        eta = 1/2 + (v/2) cos(dphi - psi),    v = 2 sqrt(P1 P2) / (P1 + P2) <= 1,

    so under uniform phase each pair contributes an arcsine on [(1-v)/2, (1+v)/2] and the
    distribution is their mixture -- the Beta(1/2, 1/2) ideal of Chapter 5 only when every
    v = 1. Comparing a histogram with this mixture tests the phase alone: pulse-energy
    imbalance is already in the null hypothesis.

    A single scaled arcsine fitted by moments was tried and rejected: the density diverges at
    the support edges, so a support error d costs (2/pi) sqrt(d) in KS distance, and on 400k
    ideal samples the fit's own statistical error gave KS 0.0185 against 0.0024 for the true
    curve.

    The mixture uses n_quantiles quantiles of v (error at most 1/n_quantiles in F, since each
    component CDF is monotone in v; exactly zero when every pair has the same v). The grid it is
    tabulated on carries the rest of the error: each component's CDF has a square-root
    singularity at its own edges (1 +/- v)/2, which linear interpolation cannot follow, so the
    grid clusters points around the edges of a sample of components as well as at 0 and 1. With
    a cosine grid alone, a train of alternating pulse energies giving every pair v = 0.6 was off
    by 0.0056 -- larger than the 0.0033 the KS test calls significant at 250k samples, i.e.
    enough to invent a phase departure. It is now below 5e-4 for that case.

    The result is independent of the AMZI phase psi. Returns (x, F) for np.interp.
    """
    P = np.maximum(np.asarray(peak_P, dtype=np.float64), 0.0)
    p1, p2 = P[:-1], P[1:]
    tot = p1 + p2
    v = np.where(tot > 0, 2.0 * np.sqrt(p1 * p2) / np.where(tot > 0, tot, 1.0), 0.0)
    vq = np.quantile(v, (np.arange(n_quantiles) + 0.5) / n_quantiles)
    x = 0.5 * (1.0 - np.cos(np.pi * np.arange(n_grid) / (n_grid - 1)))
    if n_edge:
        edges = np.quantile(v, np.linspace(0, 1, 64))
        edges = np.unique(np.concatenate([(1 - edges) / 2, (1 + edges) / 2]))
        off = np.concatenate([[0.0], np.geomspace(1e-9, 0.05, n_edge)])
        x = np.unique(np.clip(np.concatenate(
            [x, (edges[:, None] + np.concatenate([-off[::-1], off])[None, :]).ravel()]), 0.0, 1.0))
    F = np.zeros(len(x))
    for chunk in np.array_split(vq, max(1, n_quantiles // 250)):
        c = chunk[:, None]
        live = c > 1e-12
        u = np.where(live, 0.5 + (x[None, :] - 0.5) / np.where(live, c, 1.0),
                     (x[None, :] >= 0.5).astype(np.float64))        # v = 0: eta is exactly 1/2
        F += ((2.0 / np.pi) * np.arcsin(np.sqrt(np.clip(u, 0.0, 1.0)))).sum(axis=0)
    F /= n_quantiles
    F[0], F[-1] = 0.0, 1.0
    return x, np.maximum.accumulate(F)


def ks_distance(samples, cdf, max_samples=200_000):
    """One-sample Kolmogorov-Smirnov distance between samples and a CDF (a callable), on an
    evenly spaced subsample taken with stride len(s) // max_samples, so between max_samples and
    2 * max_samples - 1 samples are used. Returns (D, n used); the 99% critical value for n iid
    samples is about 1.628 / sqrt(n)."""
    s = np.asarray(samples, dtype=np.float64)
    x = np.sort(s[::max(1, len(s) // max_samples)])
    n = len(x)
    F = cdf(x)
    return float(max(np.max(np.arange(1, n + 1) / n - F), np.max(F - np.arange(n) / n))), n
