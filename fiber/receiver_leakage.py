"""
Receiver-side spectral filter leakage: direct, un-shifted classical-
channel carrier power reaching the single-photon detector because the
receive filter's ACTUAL rejection floor is finite -- a fundamentally
different (linear, post-fiber, wavelength-domain) mechanism from the
in-fiber Raman/XPM crosstalk modeled in fiber.hybrid_crosstalk.

Motivation
----------
A bandpass filter's headline "roll-off" spec (e.g. a dB/nm slope quoted
near the passband edge) does NOT continue indefinitely out to large
detunings -- every real filter (thin-film interference, FBG, etc.) has a
REJECTION FLOOR set by back-reflections, coating imperfections, and
secondary leakage paths inside the package. Naively extrapolating a
roll-off slope out several nm gives numbers in the thousands of dB,
which is not physical; the actual floor is typically 30-60 dB (up to
~80-100 dB for premium multi-cavity designs), and is usually specified
separately (often as "isolation" or "channel rejection" at a stated
offset) rather than implied by the roll-off slope.

This module converts a classical channel's power AT THE RECEIVER (after
ordinary fiber attenuation over the actual span) into an estimated
photon count landing in the QKD detection window, given an assumed or
MEASURED floor isolation -- letting a real measured value (see the
diagnostic in fiber.hybrid_crosstalk's "Launch-side mode crosstalk"
section for the analogous launch-side mechanism) be plugged in directly.
Because this is a linear, power-domain leakage mechanism independent of
spatial mode, it does not care whether the classical channel shared a
spatial mode with the QKD signal or not -- consistent with an
experimental observation that mode diversity alone gave no improvement.

    from fiber.receiver_leakage import filter_leakage_photons
"""
import numpy as np

hbar = 1.0545718e-34  # J.s


def filter_leakage_photons(P_bright_launch_W, L, fiber, bright_mode,
                            filter_floor_dB, omega_qkd, gate_window_s):
    """Mean photon count landing in the QKD detection window per gate,
    from direct (un-shifted) classical-carrier leakage through a
    finite-rejection receive filter.

    Parameters
    ----------
    P_bright_launch_W : float -- classical channel launch power (W)
    L : float -- fiber length (m)
    fiber : fiber.multimode_fiber.MultimodeFiberParams -- supplies the
        classical channel's own per-mode attenuation (fiber.alpha)
    bright_mode : int -- mode group the classical channel occupies
    filter_floor_dB : float -- the filter's ACTUAL rejection floor at the
        classical channel's wavelength offset (NOT the roll-off slope
        quoted near the passband edge -- see module docstring). Positive.
    omega_qkd : float -- QKD channel's carrier angular frequency (rad/s),
        sets the photon energy
    gate_window_s : float -- detector gate/integration window (s)

    Returns
    -------
    float -- mean photons per QKD detection gate from this mechanism alone
    """
    P_at_receiver_W = P_bright_launch_W * np.exp(-fiber.alpha[bright_mode] * L)
    P_leaked_W = P_at_receiver_W / 10 ** (filter_floor_dB / 10.0)
    E_photon = hbar * omega_qkd
    return P_leaked_W * gate_window_s / E_photon


def required_floor_dB(P_bright_launch_W, L, fiber, bright_mode,
                       omega_qkd, gate_window_s, target_photons_per_gate):
    """Inverse of filter_leakage_photons: the filter floor (dB) needed to
    keep leaked photons per gate at or below target_photons_per_gate.
    Useful for specifying how much MORE isolation (e.g. from a cascaded
    filter stage) a system needs to reach a given noise budget."""
    P_at_receiver_W = P_bright_launch_W * np.exp(-fiber.alpha[bright_mode] * L)
    E_photon = hbar * omega_qkd
    P_target_W = target_photons_per_gate * E_photon / gate_window_s
    return 10 * np.log10(P_at_receiver_W / P_target_W)
