"""
Combined mode + wavelength crosstalk: a bright classical signal launched
into ONE spatial mode group at ONE DWDM channel, and a weak (QKD-level)
signal launched into a DIFFERENT mode group at a DIFFERENT DWDM channel
of the same multimode fiber -- a scenario neither
fiber.multimode_propagator (single wavelength, multiple modes) nor
fiber.wdm_propagator (single mode, multiple wavelengths) covers alone.

Physics
-------
The QKD field is weak enough (single/few-photon level) that its own
back-action on the bright field is negligible -- this is a one-way
problem: the bright field propagates according to its own mode's
classical dynamics (fiber.multimode_propagator's physics, restricted to
its own mode), and the QKD field responds to it via cross-phase
modulation, Raman crosstalk, and spontaneous Raman noise, while its own
self-phase modulation is negligible.

The cross-coupling strength combines two already-validated, separately-
built pieces multiplicatively, rather than introducing new physics:

  Spatial part: MultimodeFiberParams.gamma_matrix[qkd_mode, bright_mode]
  -- the same spatial-overlap-weighted nonlinear coefficient used for
  intermodal coupling in fiber.multimode_propagator.

  Spectral part: fiber.raman_response.raman_gain_spectrum evaluated at
  the (typically large, fixed) DWDM channel separation -- the same
  mechanism used for inter-channel Raman crosstalk in
  fiber.wdm_propagator, but called with gamma_matrix[qkd,bright] in
  place of a single shared gamma, so the returned coefficient already
  carries the spatial overlap too.

Cross-phase modulation (XPM) on the QKD field is INSTANTANEOUS (Kerr,
not delayed), so it depends only on the spatial part (gamma_matrix) and
the bright field's own instantaneous power -- not on the channel
separation, which is far too small (~hundreds of GHz) to matter for the
electronic Kerr response.

Raman crosstalk (deterministic) and spontaneous Raman noise on the QKD
field BOTH depend on the channel separation too, via the material's
Raman gain spectrum shape -- unlike XPM, and unlike the purely
intramodal quantum-noise case, these use the bright field's LOCAL
INSTANTANEOUS (Raman) or PEAK (noise) power directly, exactly as in
fiber.wdm_propagator / fiber.quantum_wdm, so both work for a CW/quasi-CW
bright signal as well as a pulsed one.

Launch-side mode crosstalk (optional, launch_extinction_dB)
-------------------------------------------------------------
The spatial-overlap picture above assumes the bright field is launched
with perfect modal purity into bright_mode. A real mode-selective launch
device (e.g. a photonic lantern) has finite mode extinction between its
ports -- typically ~15-20 dB, not infinite -- so a fraction of the
nominal bright_mode launch power actually appears DIRECTLY in qkd_mode
at z=0, co-propagating with the QKD signal in the SAME spatial mode from
the very start. This is a qualitatively different (and typically much
stronger) coupling pathway than the fiber-propagation spatial-overlap
term above: it uses the FULL same-mode nonlinear coefficient
gamma_matrix[qkd_mode, qkd_mode] (not the weaker gamma_cross), exactly
like a same-mode WDMPropagator channel pair, because by the time this
leaked light is in qkd_mode, it propagates as an ordinary qkd_mode field
at the bright channel's wavelength -- indistinguishable, downstream of
the launch device, from having been launched there directly. Modeled as
a third field, with launch amplitude bright's own envelope scaled by
10^(-launch_extinction_dB/20), propagating with qkd_mode's own dispersion
(it now occupies that waveguide) at the bright channel's fixed carrier
offset, and contributing XPM + deterministic Raman crosstalk + spontaneous
Raman noise onto the QKD field via the same mechanisms as the bright
field above, but with gamma_matrix[qkd_mode,qkd_mode] in place of
gamma_cross. Off (launch_extinction_dB=None) by default, reproducing the
exact behavior validated before this feature existed.

    from fiber.hybrid_crosstalk import HybridCrosstalkPropagator
"""
import numpy as np

from fiber.constants import c
from fiber.raman_response import raman_response_freq_analytic, raman_gain_spectrum

from fiber.constants import hbar


class HybridCrosstalkPropagator:
    """One-way bright-mode -> QKD-mode/channel crosstalk propagator.

    Row 0 of the output is the QKD field; row 1 is the bright field.

    Parameters
    ----------
    fiber : fiber.multimode_fiber.MultimodeFiberParams
    qkd_mode : int -- mode group index the QKD field occupies (row 0)
    bright_mode : int -- mode group index the bright field occupies (row 1)
    channel_separation_Hz : float
        Bright channel's carrier frequency MINUS the QKD channel's
        (positive = bright is higher/bluer, per the direct-physical-
        offset convention used throughout fiber.wdm_propagator).
    include_raman : bool -- include the delayed Raman response (both
        intramodal on the QKD field and the deterministic Raman
        crosstalk term)
    noise : bool -- add spontaneous-Raman Langevin noise to the QKD
        field (both intramodal and from the bright field); requires seed
    seed : int or None -- RNG seed, only used if noise=True
    launch_extinction_dB : float or None -- mode extinction ratio (dB,
        positive) between the bright_mode and qkd_mode ports of the
        launch device (e.g. a photonic lantern). None (default) models a
        perfectly pure launch, i.e. no launch-side crosstalk term. See
        module docstring "Launch-side mode crosstalk".
    """

    def __init__(self, fiber, qkd_mode, bright_mode, channel_separation_Hz,
                 include_raman=True, noise=False, seed=None, launch_extinction_dB=None):
        self.fiber = fiber
        self.qkd_mode = qkd_mode
        self.bright_mode = bright_mode
        self.include_raman = include_raman
        self.noise = noise
        self.launch_extinction_dB = launch_extinction_dB
        self.channel_offset = 2 * np.pi * channel_separation_Hz  # rad/s, bright - qkd

        self.gamma_cross = fiber.gamma_matrix[qkd_mode, bright_mode]

        # deterministic Raman crosstalk coefficient (gain/loss seen BY the
        # QKD channel FROM the bright channel), combining spatial (via
        # gamma_cross in place of a bare gamma) and spectral (via the fixed
        # channel separation) parts -- see module docstring. Following the
        # convention fixed in fiber.wdm_propagator (g_R_matrix built from
        # Delta_omega_pq = Wq - Wp, i.e. "receiver minus sender" is NOT
        # negated because raman_gain_spectrum's own Omega already runs
        # opposite to the direct physical-offset convention used for
        # channel_separation_Hz): the argument here is +channel_offset
        # (= W_bright - W_qkd, receiver=qkd is implicit at offset 0).
        self.g_R_cross = raman_gain_spectrum(
            fiber.material, self.channel_offset, self.gamma_cross)

        # interchannel walk-off contribution (bright channel's carrier vs
        # the QKD channel's), using the bright mode's own beta2/beta3
        self.interchannel_beta1 = (fiber.beta2[bright_mode] * self.channel_offset
                                    + 0.5 * fiber.beta3[bright_mode] * self.channel_offset ** 2)

        # Absolute phase of the bright field RELATIVE TO the QKD field
        # (which is the z-independent phase reference, offset 0), combining
        # the intermodal absolute-phase offset already carried by
        # MultimodeFiberParams.delta_beta0 (see fiber.multimode_fiber's
        # "Absolute inter-mode phase" docstring section) with the
        # interchannel absolute-phase offset from being at a different
        # carrier too (same construction as fiber.wdm_propagator's
        # delta_beta0, antiderivative of interchannel_beta1 above, using
        # beta1_ref = material.n_g/c and the bright mode's own beta2/beta3).
        # This is THE piece of physics this module exists to provide: an
        # instantaneous-power-only comparison (XPM/Raman terms) says
        # nothing about the QKD signal's own phase, which is what a
        # phase-encoded protocol actually measures.
        beta1_ref = fiber.material.n_g / c
        self.delta_beta0_bright = (
            (fiber.delta_beta0[bright_mode] - fiber.delta_beta0[qkd_mode])
            + beta1_ref * self.channel_offset
            + 0.5 * fiber.beta2[bright_mode] * self.channel_offset ** 2
            + (1.0 / 6.0) * fiber.beta3[bright_mode] * self.channel_offset ** 3
        )

        if noise:
            n_th = fiber.material.phonon_occupation(self.channel_offset)
            gain_side = self.g_R_cross > 0
            self._cross_noise_gain = (self.g_R_cross * (n_th + 1) if gain_side
                                       else abs(self.g_R_cross) * n_th)
            self.rng = np.random.default_rng(seed)

        self.leak_active = launch_extinction_dB is not None
        if self.leak_active:
            self.leak_amplitude_scale = 10 ** (-launch_extinction_dB / 20.0)

            # once leaked into qkd_mode, this light propagates as an
            # ordinary qkd_mode field at the bright channel's wavelength --
            # so it uses the FULL same-mode gamma (not gamma_cross) both
            # for its own SPM and for its XPM/Raman crosstalk onto QKD
            self.gamma_leak = fiber.gamma_matrix[qkd_mode, qkd_mode]
            self.g_R_leak = raman_gain_spectrum(
                fiber.material, self.channel_offset, self.gamma_leak)

            # dispersion/walk-off: qkd_mode's own waveguide, at the bright
            # channel's carrier offset (mirrors interchannel_beta1 above,
            # but with qkd_mode's beta2/beta3 instead of bright_mode's)
            self.interchannel_beta1_leak = (fiber.beta2[qkd_mode] * self.channel_offset
                                             + 0.5 * fiber.beta3[qkd_mode] * self.channel_offset ** 2)

            if noise:
                gain_side_leak = self.g_R_leak > 0
                self._leak_noise_gain = (self.g_R_leak * (n_th + 1) if gain_side_leak
                                          else abs(self.g_R_leak) * n_th)

    def propagate(self, A_qkd_0, A_bright_0, dt, L, step_size=20.0, n_steps=None):
        """Propagate the QKD + bright fields through fiber length L (m).

        Parameters
        ----------
        A_qkd_0, A_bright_0 : complex arrays, shape (n_pts,)
            Launch fields (same time grid/window for both).
        dt, L, step_size, n_steps : as in fiber.propagator.FiberPropagator

        Returns
        -------
        (A_qkd_out, A_bright_out) : complex arrays, shape (n_pts,) each
        """
        n_pts = len(A_qkd_0)
        if n_steps is None:
            n_steps = max(int(np.ceil(L / step_size)), 1)
        dz = L / n_steps

        Omega = 2 * np.pi * np.fft.fftfreq(n_pts, d=dt)
        f_R = self.fiber.material.f_R
        H_R = (raman_response_freq_analytic(self.fiber.material, Omega)
               if self.include_raman else None)

        fiber = self.fiber
        q, b = self.qkd_mode, self.bright_mode

        base_qkd = (-fiber.alpha[q] / 2 + 1j * fiber.beta2[q] / 2 * Omega ** 2
                    - 1j * fiber.beta3[q] / 6 * Omega ** 3
                    - 1j * fiber.delta_beta1[q] * Omega)
        base_bright = (-fiber.alpha[b] / 2 + 1j * fiber.beta2[b] / 2 * Omega ** 2
                        - 1j * fiber.beta3[b] / 6 * Omega ** 3
                        - 1j * (fiber.delta_beta1[b] + self.interchannel_beta1) * Omega)
        D_half_qkd = np.exp(base_qkd * dz / 2)
        D_half_bright = np.exp(base_bright * dz / 2)

        if self.leak_active:
            base_leak = (-fiber.alpha[q] / 2 + 1j * fiber.beta2[q] / 2 * Omega ** 2
                         - 1j * fiber.beta3[q] / 6 * Omega ** 3
                         - 1j * (fiber.delta_beta1[q] + self.interchannel_beta1_leak) * Omega)
            D_half_leak = np.exp(base_leak * dz / 2)
            A_leak_f = np.fft.fft((A_bright_0 * self.leak_amplitude_scale).astype(complex))

        if self.noise:
            g_R_self = raman_gain_spectrum(fiber.material, Omega, fiber.gamma_matrix[q, q])
            n_th_self = fiber.material.phonon_occupation(Omega)
            gain_side_self = g_R_self > 0
            self_noise_gain = np.where(gain_side_self, g_R_self * (n_th_self + 1),
                                        np.abs(g_R_self) * n_th_self)
            self_noise_gain = np.nan_to_num(self_noise_gain, nan=0.0, posinf=0.0, neginf=0.0)
            omega_abs_self = fiber.omega0 - Omega
            # noise photon energy is set by the RECEIVING (QKD) channel's own
            # absolute carrier, fiber.omega0 -- matching fiber.quantum_wdm's
            # _channel_omega_abs[p] (receiver's own carrier), not the source's

        A_qkd_f = np.fft.fft(A_qkd_0.astype(complex))
        A_bright_f = np.fft.fft(A_bright_0.astype(complex))

        for _ in range(n_steps):
            A_qkd_f *= D_half_qkd
            A_bright_f *= D_half_bright
            A_qkd_t = np.fft.ifft(A_qkd_f)
            A_bright_t = np.fft.ifft(A_bright_f)
            if self.leak_active:
                A_leak_f *= D_half_leak
                A_leak_t = np.fft.ifft(A_leak_f)

            P_qkd = np.abs(A_qkd_t) ** 2
            P_bright = np.abs(A_bright_t) ** 2

            gamma_self_qkd = fiber.gamma_matrix[q, q]
            gamma_self_bright = fiber.gamma_matrix[b, b]

            if self.include_raman and f_R > 0:
                conv_qkd = np.real(np.fft.ifft(np.fft.fft(P_qkd) * H_R))
                conv_bright = np.real(np.fft.ifft(np.fft.fft(P_bright) * H_R))
                spm_qkd = (1 - f_R) * P_qkd + f_R * conv_qkd
                spm_bright = (1 - f_R) * P_bright + f_R * conv_bright
            else:
                spm_qkd = P_qkd
                spm_bright = P_bright

            # QKD field: self-SPM (negligible at QKD power, kept for
            # correctness) + XPM from bright (Kerr, instantaneous) +
            # deterministic Raman crosstalk from bright (uses g_R_cross)
            xpm_from_bright = 2 * (1 - f_R) * P_bright if self.include_raman else 2 * P_bright
            kerr_phase_qkd = gamma_self_qkd * spm_qkd + self.gamma_cross * xpm_from_bright
            raman_crosstalk_qkd = 0.5 * self.g_R_cross * P_bright if self.include_raman else 0.0

            if self.leak_active:
                P_leak = np.abs(A_leak_t) ** 2
                if self.include_raman and f_R > 0:
                    conv_leak = np.real(np.fft.ifft(np.fft.fft(P_leak) * H_R))
                    spm_leak = (1 - f_R) * P_leak + f_R * conv_leak
                    xpm_from_leak = 2 * (1 - f_R) * P_leak
                else:
                    spm_leak = P_leak
                    xpm_from_leak = 2 * P_leak
                # same-mode coupling: full gamma_leak (=gamma_matrix[q,q]),
                # not the weaker gamma_cross, since this light now occupies
                # qkd_mode's own waveguide
                kerr_phase_qkd = kerr_phase_qkd + self.gamma_leak * xpm_from_leak
                raman_crosstalk_qkd = raman_crosstalk_qkd + (
                    0.5 * self.g_R_leak * P_leak if self.include_raman else 0.0)

            A_qkd_t = A_qkd_t * np.exp((1j * kerr_phase_qkd + raman_crosstalk_qkd) * dz)

            if self.noise:
                P_loc_self = np.max(P_qkd)
                psd_self = hbar * np.abs(omega_abs_self) * self_noise_gain * P_loc_self
                # ifft divides by n_pts, so frequency-domain amplitudes carry sqrt(n_pts)
                amp_self = np.sqrt(np.clip(psd_self, 0, None) * n_pts * dz / dt)
                noise_self_f = amp_self * (self.rng.standard_normal(n_pts)
                                            + 1j * self.rng.standard_normal(n_pts)) / np.sqrt(2)
                noise_self = np.fft.ifft(noise_self_f)

                # cross-channel (matches fiber.quantum_wdm's inter-channel
                # term): driven by the bright field's LOCAL INSTANTANEOUS
                # power P_bright(t), not just its peak, since the Raman gain
                # is flat across the bright channel's own narrow bandwidth
                # around the fixed pair separation -- generated directly in
                # the time domain (white-in-time), unlike the intramodal
                # term above which needs the frequency-dependent shape
                psd_cross = hbar * abs(fiber.omega0) * self._cross_noise_gain * P_bright
                amp_cross = np.sqrt(np.clip(psd_cross, 0, None) * dz / dt)
                noise_cross = amp_cross * (self.rng.standard_normal(n_pts)
                                            + 1j * self.rng.standard_normal(n_pts)) / np.sqrt(2)

                A_qkd_t = A_qkd_t + noise_self + noise_cross

                if self.leak_active:
                    psd_leak = hbar * abs(fiber.omega0) * self._leak_noise_gain * P_leak
                    amp_leak = np.sqrt(np.clip(psd_leak, 0, None) * dz / dt)
                    noise_leak = amp_leak * (self.rng.standard_normal(n_pts)
                                              + 1j * self.rng.standard_normal(n_pts)) / np.sqrt(2)
                    A_qkd_t = A_qkd_t + noise_leak

            # bright field: self-dynamics only (QKD back-action negligible)
            kerr_phase_bright = gamma_self_bright * spm_bright
            A_bright_t = A_bright_t * np.exp(1j * kerr_phase_bright * dz)

            A_qkd_f = np.fft.fft(A_qkd_t) * D_half_qkd
            A_bright_f = np.fft.fft(A_bright_t) * D_half_bright
            if self.leak_active:
                # leak field: self-dynamics only (same one-way approximation)
                kerr_phase_leak = self.gamma_leak * spm_leak
                A_leak_t = A_leak_t * np.exp(1j * kerr_phase_leak * dz)
                A_leak_f = np.fft.fft(A_leak_t) * D_half_leak

        A_qkd_out = np.fft.ifft(A_qkd_f)
        A_bright_out = np.fft.ifft(A_bright_f) * np.exp(1j * self.delta_beta0_bright * L)
        return A_qkd_out, A_bright_out
