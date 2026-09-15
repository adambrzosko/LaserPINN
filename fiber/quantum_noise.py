"""
Semiclassical quantum-noise extension of FiberPropagator: adds spontaneous
Raman scattering as a Langevin noise term at every propagation step, with
strength set by the Bose-Einstein phonon occupation (fluctuation-dissipation
relation between the classical Raman gain and its spontaneous noise).

Physics
-------
Per frequency bin Omega, using fiber.raman_response.raman_gain_spectrum
(g_R) and fiber.materials.FiberMaterial.phonon_occupation (n_th):

    gain sub-band (g_R > 0):  noise ~ hbar*omega*g_R*(n_th + 1)
        -> nonzero even as T -> 0 (vacuum-driven spontaneous scattering
           persists on the gain side)
    loss sub-band (g_R < 0):  noise ~ hbar*omega*|g_R|*n_th
        -> vanishes as T -> 0 (this side needs a real phonon to be
           absorbed from the bath, so there is nothing to scatter with at
           zero temperature)

This is the standard noise-figure recipe for a phase-insensitive
gain/loss channel coupled to a thermal bath (the same fluctuation-
dissipation structure as ASE in a Raman amplifier), applied locally at
each z-step of the classical GNLSE integration.

Approximation
-------------
The noise generated at each step is driven by the *local instantaneous
peak power* of the field (a quasi-CW / undepleted-pump approximation),
not a full convolution with the pump's own time-varying spectrum. This
keeps the model tractable and is correct for a CW or quasi-CW pump
(temperature dependence, gain/loss-band asymmetry, scaling with power and
with the Raman gain spectrum); the absolute noise PSD is checked against
the analytic forward spontaneous-Raman result in
tests/test_noise_calibration.py. For pulsed pumps the peak-power driver
spreads peak-level noise across the whole window and overstates the total;
use fiber.gmmnlse with noise='stochastic' (time-local driver) instead.

    from fiber.quantum_noise import QuantumRamanPropagator, ensemble_propagate
"""
import numpy as np

from fiber.propagator import FiberPropagator
from fiber.raman_response import raman_gain_spectrum

hbar = 1.0545718e-34  # J.s


class QuantumRamanPropagator(FiberPropagator):
    """FiberPropagator with spontaneous-Raman Langevin noise added per step.

    Parameters
    ----------
    fiber : fiber.fiber_params.FiberParams
    seed : int or None -- RNG seed for a reproducible noise realization
    """

    def __init__(self, fiber, seed=None):
        super().__init__(fiber, include_raman=True)
        self.rng = np.random.default_rng(seed)
        self._noise_gain = None
        self._omega_abs = None

    def _prepare(self, Omega, dt):
        g_R = raman_gain_spectrum(self.fiber.material, Omega, self.fiber.gamma)
        n_th = self.fiber.material.phonon_occupation(Omega)

        gain_side = g_R > 0
        noise_gain = np.where(gain_side, g_R * (n_th + 1), np.abs(g_R) * n_th)
        self._noise_gain = np.nan_to_num(noise_gain, nan=0.0, posinf=0.0, neginf=0.0)
        # physical optical frequency at offset Omega is omega0 - Omega, given
        # this codebase's envelope convention (see raman_gain_spectrum)
        self._omega_abs = self.fiber.omega0 - Omega

    def _noise_step(self, A_t, dz, dt, n_pts):
        P_loc = np.max(np.abs(A_t) ** 2)
        if P_loc <= 0:
            return 0.0
        psd = hbar * np.abs(self._omega_abs) * self._noise_gain * P_loc
        # ifft divides by n_pts, so frequency-domain amplitudes carry sqrt(n_pts)
        amp = np.sqrt(np.clip(psd, 0, None) * n_pts * dz / dt)
        noise_f = amp * (self.rng.standard_normal(n_pts)
                          + 1j * self.rng.standard_normal(n_pts)) / np.sqrt(2)
        return np.fft.ifft(noise_f)


def ensemble_propagate(fiber, A0, dt, L, n_runs, seed0=0, **propagate_kwargs):
    """Run n_runs independent QuantumRamanPropagator noise realizations.

    Returns a complex array of shape (n_runs, len(A0)) with one output
    field per realization, e.g. for estimating the mean field and the
    noise-power spectrum (variance across realizations) at the output.
    """
    n_pts = len(A0)
    outs = np.empty((n_runs, n_pts), dtype=complex)
    for i in range(n_runs):
        prop = QuantumRamanPropagator(fiber, seed=seed0 + i)
        outs[i] = prop.propagate(A0, dt, L, **propagate_kwargs)
    return outs
