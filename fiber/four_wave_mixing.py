"""
Four-wave mixing (FWM) between discrete WDM channels: the standard
undepleted-pump analytical treatment, giving the power generated at a
new "idler" frequency omega_i + omega_j - omega_k from three pump
channels (two of which may be the same channel -- degenerate FWM).

Why analytical, not a dynamical propagator
--------------------------------------------
Everything else co-propagating in fiber.wdm_propagator.WDMPropagator
only carries INTENSITY-driven coupling (XPM, Raman crosstalk) -- genuine
FWM needs the coherent four-field term A_i*A_j*conj(A_k), which requires
tracking phase-matching between channel triplets and (for channels not
already in the launch set) generating power at NEW idler frequencies not
present in the input. A fully general, dynamically-coupled N-channel FWM
propagator would need to dynamically grow the channel set and track
phase mismatch between every triplet at every step -- a substantially
bigger undertaking than anything else in fiber/. The engineering
question this is usually asked to answer -- "how much FWM power lands on
my quantum channel's wavelength from my classical WDM comb" -- doesn't
need the dynamics, just the generated power at the end of the fiber, so
this module gives that directly via the standard closed-form (CW/quasi-
CW, undepleted-pump) formula instead, the same steady-state-over-
dynamics tradeoff already made for fiber.brillouin.

Physics
-------
For pump channels at angular frequencies omega_i, omega_j, omega_k
(possibly omega_i=omega_j, the degenerate two-pump case), FWM generates
a new field at omega_ijk = omega_i + omega_j - omega_k with power:

    P_ijk(L) = D^2 * gamma^2 * P_i*P_j*P_k * L_eff^2 * exp(-alpha*L) * eta_ijk

D is a degeneracy factor (D=1 for the degenerate case omega_i=omega_j,
D=2 for non-degenerate distinct pumps -- see the Approximation note
below), L_eff = (1-exp(-alpha*L))/alpha, and eta_ijk in [0,1] is the
phase-matching efficiency:

    eta_ijk = alpha^2/(alpha^2+Delta_beta^2) *
              [1 + 4*exp(-alpha*L)*sin^2(Delta_beta*L/2)/(1-exp(-alpha*L))^2]

with Delta_beta the phase mismatch, computed here to leading order in
the channel separations (consistent with how delta_beta1/delta_beta0
are computed elsewhere in fiber/, reusing the same Taylor-expansion
approach rather than a full dispersion curve):

    Delta_beta = beta2*(omega_i-omega_k)*(omega_j-omega_k)

eta_ijk=1 exactly at perfect phase matching (Delta_beta=0) in both the
lossy and lossless limits -- a directly checkable sanity condition,
verified in tests/test_four_wave_mixing.py.

Approximation
-------------
The phase-matching efficiency formula and its Delta_beta=0 limit are
standard, textbook results independently checkable this way. The
degeneracy prefactor D used here (D=1 degenerate, D=2 non-degenerate) is
a representative, commonly-used convention, not independently verified
against a reference the way other quantities in this codebase were
cross-checked -- treat fwm_power()'s ABSOLUTE magnitude as order-of-
magnitude, and calibrate D against a reference for your specific
convention if you need precise quantitative predictions. eta_ijk's
SHAPE (which triplets are phase-matched, and how efficiency falls off
with channel spacing/dispersion) is on much firmer ground than the
absolute prefactor.

    from fiber.four_wave_mixing import fwm_power, fwm_ghost_tone_power
"""
import numpy as np
from itertools import combinations_with_replacement

from fiber.constants import hbar


def fwm_phase_mismatch(fiber, omega_i, omega_j, omega_k):
    """Leading-order FWM phase mismatch Delta_beta (rad/m) for the triplet
    (i,j,k), all as angular-frequency OFFSETS (rad/s) from the fiber's
    reference wavelength -- consistent with WDMPropagator's channel_offsets
    convention (2*pi*channel_offsets_Hz)."""
    return fiber.beta2 * (omega_i - omega_k) * (omega_j - omega_k)


def fwm_efficiency(fiber, omega_i, omega_j, omega_k, L):
    """Phase-matching efficiency eta_ijk in [0,1] (see module docstring)."""
    alpha = fiber.alpha
    dbeta = fwm_phase_mismatch(fiber, omega_i, omega_j, omega_k)

    if alpha <= 0:
        if dbeta == 0:
            return 1.0
        x = dbeta * L / 2
        return float((np.sin(x) / x) ** 2)

    denom = alpha ** 2 + dbeta ** 2
    base = alpha ** 2 / denom if denom > 0 else 1.0
    expL = np.exp(-alpha * L)
    if expL >= 1.0 - 1e-15:
        bracket = 1.0
    else:
        bracket = 1 + 4 * expL * np.sin(dbeta * L / 2) ** 2 / (1 - expL) ** 2
    return float(base * bracket)


def fwm_power(fiber, P_i, P_j, P_k, omega_i, omega_j, omega_k, L):
    """Generated FWM idler power (W) at omega_i+omega_j-omega_k from pump
    powers P_i, P_j, P_k (W) at their respective channel offsets (rad/s).

    Degenerate (omega_i == omega_j, i.e. a single pump self-mixing with a
    second channel) uses D=1; non-degenerate (three distinct frequencies,
    or omega_i != omega_j in general) uses D=2 -- see module docstring's
    Approximation note.
    """
    alpha = fiber.alpha
    L_eff = (1 - np.exp(-alpha * L)) / alpha if alpha > 0 else L
    D = 1.0 if omega_i == omega_j else 2.0
    eta = fwm_efficiency(fiber, omega_i, omega_j, omega_k, L)
    return (D ** 2 * fiber.gamma ** 2 * P_i * P_j * P_k * L_eff ** 2
            * np.exp(-alpha * L) * eta)


def fwm_ghost_tone_power(fiber, channel_powers_W, channel_offsets_Hz, target_offset_Hz, L,
                          tolerance_Hz=1e9):
    """Total FWM power (W) landing within tolerance_Hz of target_offset_Hz
    (e.g. a quantum channel's wavelength) from every (i,j,k) triplet drawn
    from the given classical channels whose product omega_i+omega_j-omega_k
    falls in that window.

    Different triplets are generally not mutually phase-coherent, so their
    powers are summed incoherently (the standard, conservative engineering
    approach) rather than added as fields.

    Parameters
    ----------
    fiber : fiber.fiber_params.FiberParams
    channel_powers_W : array-like -- launch power (W) of each classical channel
    channel_offsets_Hz : array-like -- carrier offset (Hz) of each classical channel
    target_offset_Hz : float -- the quantum/victim channel's offset (Hz)
    L : float -- fiber length (m)
    tolerance_Hz : float -- how close an FWM product must land to
        target_offset_Hz to count (default 1 GHz)

    Returns
    -------
    dict with 'total_power_W' and 'contributions' (list of
    (i, j, k, idler_offset_Hz, power_W) for every triplet that contributed)
    """
    P = np.asarray(channel_powers_W, dtype=float)
    W = 2 * np.pi * np.asarray(channel_offsets_Hz, dtype=float)
    n = len(W)
    target_w = 2 * np.pi * target_offset_Hz
    tol_w = 2 * np.pi * tolerance_Hz

    total = 0.0
    contributions = []
    for i, j in combinations_with_replacement(range(n), 2):
        for k in range(n):
            if k == i or k == j:
                continue
            omega_idler = W[i] + W[j] - W[k]
            if abs(omega_idler - target_w) > tol_w:
                continue
            p_ijk = fwm_power(fiber, P[i], P[j], P[k], W[i], W[j], W[k], L)
            total += p_ijk
            contributions.append((i, j, k, omega_idler / (2 * np.pi), p_ijk))

    return dict(total_power_W=total, contributions=contributions)
