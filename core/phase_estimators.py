"""Pulse-to-pulse phase-correlation estimators, r1 = |<exp(i*dphi)>| (0 random, 1 locked).

All three agree for a single-mode source.  For a multimode source use
r1_power_weighted; the other two are kept only to reproduce earlier numbers.
"""
import numpy as np


def _as_pulses_by_modes(E_peak):
    E = np.asarray(E_peak)
    return E[:, None] if E.ndim == 1 else E


def _per_mode_r1(E):
    d = np.diff(np.angle(E), axis=0)
    return np.abs(np.mean(np.exp(1j * d), axis=0))


def r1_summed_field(E_peak):
    """Phase of the mode-summed field; scores mode-partition noise as randomisation."""
    E = _as_pulses_by_modes(E_peak).sum(axis=1)
    return float(np.abs(np.mean(np.exp(1j * np.diff(np.angle(E))))))


def r1_per_mode_mean(E_peak):
    """Unweighted mean of per-mode r1; dominated by dark wing modes on real output."""
    return float(_per_mode_r1(_as_pulses_by_modes(E_peak)).mean())


def r1_power_weighted(E_peak):
    """Per-mode r1 weighted by mean mode power; valid for multimode sources."""
    E = _as_pulses_by_modes(E_peak)
    w = np.mean(np.abs(E)**2, axis=0)
    return float(np.sum(_per_mode_r1(E) * w) / np.sum(w))
