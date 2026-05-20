"""
Shared analysis functions for gain-switched DFB laser simulations.

Consolidates metrics that were previously duplicated across 13+ scripts:
r1, phase quality, timing jitter, AMZI outputs.
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
    """Overlapping Allan deviation of fractional timing.

    Returns (tau_array, adev_array).
    """
    t_peak = peak_k.astype(np.float64) * dt
    y = np.diff(t_peak) / T_rep

    n = len(y)
    ms = np.unique(np.geomspace(1, min(n // 4, max_m), 40).astype(int))
    ms = ms[ms >= 1]

    adev = np.zeros(len(ms))
    for i, m in enumerate(ms):
        ybar = np.convolve(y, np.ones(m) / m, mode='valid')
        diff = np.diff(ybar)
        adev[i] = np.sqrt(0.5 * np.mean(diff**2))

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
