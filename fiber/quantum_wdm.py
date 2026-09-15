"""
Semiclassical quantum-noise extension of WDMPropagator: adds spontaneous
Raman scattering as a Langevin noise term at every propagation step, for
BOTH mechanisms fiber.wdm_propagator.WDMPropagator carries classically:

  Intra-channel spontaneous Raman: each channel's own local peak power
  seeds noise across its own (local, small) internal Omega grid -- the
  same physics and formula as fiber.quantum_noise.QuantumRamanPropagator,
  just applied per channel using that channel's own absolute carrier
  frequency (fiber.omega0 + channel_offset) for the photon-energy term.

  Inter-channel spontaneous Raman ("Raman crosstalk noise floor"): the
  quantum-noise counterpart of WDMPropagator's deterministic inter-
  channel Raman crosstalk term. Each channel q's local instantaneous
  power seeds spontaneous-Raman noise landing in every OTHER channel p,
  using the material's Raman gain spectrum evaluated at the FIXED
  channel-to-channel separation (matching the classical term), weighted
  by the Bose-Einstein phonon occupation at that separation. This is the
  mechanism most often cited as the dominant noise source in classical/
  quantum coexistence on shared fiber -- a bright classical channel's
  spontaneous Raman scattering landing in a co-propagating quantum
  channel's wavelength slot.

Both use the same fluctuation-dissipation recipe as QuantumRamanPropagator
(see its docstring for the physics and the approximation caveats -- in
particular, noise here is driven by each channel's LOCAL INSTANTANEOUS
power, a quasi-CW/undepleted-pump approximation, and the absolute noise-
photon calibration should be checked against a measured cross-section
before use for absolute photon-count predictions).

    from fiber.quantum_wdm import QuantumWDMPropagator, ensemble_propagate_wdm
"""
import numpy as np

from fiber.wdm_propagator import WDMPropagator
from fiber.raman_response import raman_gain_spectrum

hbar = 1.0545718e-34  # J.s


class QuantumWDMPropagator(WDMPropagator):
    """WDMPropagator with intra- and inter-channel spontaneous-Raman
    Langevin noise added per step.

    Parameters
    ----------
    fiber : fiber.fiber_params.FiberParams
    channel_offsets_Hz : array-like -- as in WDMPropagator
    seed : int or None -- RNG seed for a reproducible noise realization
    """

    def __init__(self, fiber, channel_offsets_Hz, seed=None):
        super().__init__(fiber, channel_offsets_Hz, include_raman=True)
        self.rng = np.random.default_rng(seed)
        self._intra_noise_gain = None
        self._intra_omega_abs = None
        self._inter_noise_gain = None
        self._channel_omega_abs = None

    def _prepare(self, Omega, dt):
        material = self.fiber.material
        gamma = self.fiber.gamma

        # intra-channel: same local-Omega recipe as QuantumRamanPropagator,
        # shared across channels (small internal Omega grid), but the
        # absolute photon energy differs per channel via its own carrier offset
        g_R_local = raman_gain_spectrum(material, Omega, gamma)
        n_th_local = material.phonon_occupation(Omega)
        gain_side_local = g_R_local > 0
        noise_gain_local = np.where(gain_side_local, g_R_local * (n_th_local + 1),
                                     np.abs(g_R_local) * n_th_local)
        self._intra_noise_gain = np.nan_to_num(noise_gain_local, nan=0.0, posinf=0.0, neginf=0.0)
        # physical frequency at internal offset Omega, per channel (see
        # raman_gain_spectrum: physical_freq = omega0 - Omega for the local
        # grid; each channel's own carrier is fiber.omega0 + channel_offset)
        self._intra_omega_abs = ((self.fiber.omega0 + self.channel_offsets[:, None])
                                  - Omega[None, :])  # (N, n_pts)

        # inter-channel: fluctuation-dissipation applied to the ALREADY
        # computed g_R_matrix (fixed channel-to-channel separation)
        Wp, Wq = np.meshgrid(self.channel_offsets, self.channel_offsets, indexing='ij')
        Delta_omega_pq = Wq - Wp  # matches the convention used to build g_R_matrix
        n_th_pq = material.phonon_occupation(Delta_omega_pq)
        gain_side_pq = self.g_R_matrix > 0
        noise_gain_pq = np.where(gain_side_pq, self.g_R_matrix * (n_th_pq + 1),
                                  np.abs(self.g_R_matrix) * n_th_pq)
        np.fill_diagonal(noise_gain_pq, 0.0)  # self term is the intra-channel piece above
        self._inter_noise_gain = np.nan_to_num(noise_gain_pq, nan=0.0, posinf=0.0, neginf=0.0)  # (N, N)
        self._channel_omega_abs = self.fiber.omega0 + self.channel_offsets  # (N,)

    def _noise_step(self, A_t, P, dz, dt, n_pts):
        N = self.n_channels

        # intra-channel: each channel's own local peak power drives noise
        # across its own internal Omega grid
        P_loc = np.max(P, axis=1, keepdims=True)  # (N, 1)
        psd_intra = hbar * np.abs(self._intra_omega_abs) * self._intra_noise_gain[None, :] * P_loc
        # ifft divides by n_pts, so frequency-domain amplitudes carry sqrt(n_pts)
        amp_intra = np.sqrt(np.clip(psd_intra, 0, None) * n_pts * dz / dt)
        noise_f_intra = amp_intra * (self.rng.standard_normal((N, n_pts))
                                      + 1j * self.rng.standard_normal((N, n_pts))) / np.sqrt(2)
        noise_intra = np.fft.ifft(noise_f_intra, axis=1)

        # inter-channel: channel q's LOCAL INSTANTANEOUS power (not just its
        # peak) drives white-in-time noise landing in channel p, since the
        # Raman gain spectrum is treated as flat across each channel's own
        # (narrow) bandwidth around the fixed pair separation
        variance_per_t = self._inter_noise_gain @ P  # (N, n_pts): sum_q noise_gain[p,q]*P[q,t]
        psd_inter = hbar * np.abs(self._channel_omega_abs)[:, None] * variance_per_t
        amp_inter = np.sqrt(np.clip(psd_inter, 0, None) * dz / dt)
        noise_inter = amp_inter * (self.rng.standard_normal((N, n_pts))
                                    + 1j * self.rng.standard_normal((N, n_pts))) / np.sqrt(2)

        return noise_intra + noise_inter


def ensemble_propagate_wdm(fiber, channel_offsets_Hz, A0, dt, L, n_runs, seed0=0, **propagate_kwargs):
    """Run n_runs independent QuantumWDMPropagator noise realizations.

    Returns a complex array of shape (n_runs, n_channels, len(A0's own
    time axis)) with one output field set per realization.
    """
    A0 = np.asarray(A0)
    n_channels = len(np.atleast_1d(channel_offsets_Hz))
    n_pts = A0.shape[-1]
    outs = np.empty((n_runs, n_channels, n_pts), dtype=complex)
    for i in range(n_runs):
        prop = QuantumWDMPropagator(fiber, channel_offsets_Hz, seed=seed0 + i)
        outs[i] = prop.propagate(A0, dt, L, **propagate_kwargs)
    return outs
