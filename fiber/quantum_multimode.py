"""
Semiclassical quantum-noise extension of MultimodeFiberPropagator: adds
spontaneous Raman scattering as a Langevin noise term at every
propagation step, for BOTH mechanisms MultimodeFiberPropagator carries
classically -- intramodal (a mode group's own delayed Raman response)
and intermodal (spatial-overlap-weighted coupling between mode groups).

This is the multimode counterpart of fiber.quantum_noise.QuantumRamanPropagator
(single-mode) and fiber.quantum_wdm.QuantumWDMPropagator (WDM channels);
see QuantumRamanPropagator's docstring for the underlying fluctuation-
dissipation physics and approximation caveats, which all carry over here.

Physics
-------
Unlike the DETERMINISTIC intermodal Raman term in MultimodeFiberPropagator
(which needs the driving mode to have genuine time-varying power -- a CW
mode transfers zero classical Raman effect via the delayed-response
convolution, see that module's docstring), spontaneous-noise generation
here is driven by each mode's LOCAL INSTANTANEOUS PEAK POWER directly
(the same mechanism QuantumRamanPropagator uses), which is nonzero even
for a perfectly CW/quasi-CW mode. This is the physically important
difference: a CW classical signal launched into one mode group of a
multimode fiber generates NO classical Raman distortion of itself or
others, but DOES generate a real spontaneous-Raman noise floor, both in
its own mode and (weighted by spatial overlap) in every other mode group
-- exactly the mechanism relevant to a bright classical channel sharing
a multimode fiber with a weak quantum channel in a different spatial mode.

Because both the intramodal and intermodal terms here share the SAME
local (per-mode-independent) frequency grid and the SAME material Raman
response (unlike the WDM case, where inter-channel noise uses the gain
spectrum evaluated at a large, FIXED channel separation), they combine
into one compact calculation: the fluctuation-dissipation-weighted
Raman-response SHAPE (gain-side/loss-side split via Bose-Einstein
phonon occupation) is identical for every mode PAIR, and only its
overall STRENGTH differs -- set by MultimodeFiberParams.gamma_matrix
[p,q] (already the same spatial-overlap-weighted coupling used for the
classical term; gamma_matrix's diagonal is the intramodal/self term).

    from fiber.quantum_multimode import QuantumMultimodePropagator, ensemble_propagate_multimode
"""
import numpy as np

from fiber.multimode_propagator import MultimodeFiberPropagator
from fiber.raman_response import raman_response_freq_analytic

hbar = 1.0545718e-34  # J.s


class QuantumMultimodePropagator(MultimodeFiberPropagator):
    """MultimodeFiberPropagator with intra- and inter-modal spontaneous-
    Raman Langevin noise added per step.

    Parameters
    ----------
    fiber : fiber.multimode_fiber.MultimodeFiberParams
    seed : int or None -- RNG seed for a reproducible noise realization
    """

    def __init__(self, fiber, seed=None):
        super().__init__(fiber, include_raman=True)
        self.rng = np.random.default_rng(seed)
        self._weighted_shape = None
        self._omega_abs = None

    def _prepare(self, Omega, dt):
        material = self.fiber.material
        H_R = raman_response_freq_analytic(material, Omega)
        n_th = material.phonon_occupation(Omega)

        # gamma factored OUT here (unlike raman_gain_spectrum's g_R): the
        # per-mode-pair strength is applied separately via gamma_matrix in
        # _noise_step, since it varies by (p,q) here rather than being a
        # single shared scalar as in the single-mode/WDM cases
        raw_shape = -2.0 * material.f_R * np.imag(H_R)
        gain_side = raw_shape > 0
        weighted = np.where(gain_side, raw_shape * (n_th + 1), np.abs(raw_shape) * n_th)
        self._weighted_shape = np.nan_to_num(weighted, nan=0.0, posinf=0.0, neginf=0.0)
        # all mode groups share the same nominal carrier wavelength (unlike
        # WDM channels), so a single shared photon-energy array suffices
        self._omega_abs = self.fiber.omega0 - Omega

    def _noise_step(self, A_t, P, dz, dt, n_pts):
        M = self.fiber.n_modes
        P_loc = np.max(P, axis=1)  # (M,) local peak power per mode group

        # gamma_matrix already includes the diagonal (intramodal) term, so
        # this one matrix-vector product covers both intra- and inter-modal
        # noise sources simultaneously
        coupled_power = self.fiber.gamma_matrix @ P_loc  # (M,)

        psd = (hbar * np.abs(self._omega_abs)[None, :] * self._weighted_shape[None, :]
               * coupled_power[:, None])  # (M, n_pts)
        # ifft divides by n_pts, so frequency-domain amplitudes carry sqrt(n_pts)
        amp = np.sqrt(np.clip(psd, 0, None) * n_pts * dz / dt)
        noise_f = amp * (self.rng.standard_normal((M, n_pts))
                          + 1j * self.rng.standard_normal((M, n_pts))) / np.sqrt(2)
        return np.fft.ifft(noise_f, axis=1)


def ensemble_propagate_multimode(fiber, A0, dt, L, n_runs, seed0=0, **propagate_kwargs):
    """Run n_runs independent QuantumMultimodePropagator noise realizations.

    Returns a complex array of shape (n_runs, n_modes, n_pts) with one
    output field set per realization.
    """
    A0 = np.asarray(A0)
    n_pts = A0.shape[-1]
    outs = np.empty((n_runs, fiber.n_modes, n_pts), dtype=complex)
    for i in range(n_runs):
        prop = QuantumMultimodePropagator(fiber, seed=seed0 + i)
        outs[i] = prop.propagate(A0, dt, L, **propagate_kwargs)
    return outs
