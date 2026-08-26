"""
WDM (multi-channel) single-mode fiber propagator: symmetric split-step
solver for N co-propagating channels A_p(z,t), p=0..N-1, sharing ONE
spatial mode of a fiber.fiber_params.FiberParams fiber, at channel
carrier offsets from the fiber's reference wavelength.

Implements:

  Self-phase modulation (SPM): each channel's own intensity modulates its
  own phase via the fiber's Kerr coefficient gamma, plus the intra-channel
  delayed Raman response -- identical physics to a single-channel
  fiber.propagator.FiberPropagator (a 1-channel WDMPropagator reduces to
  it exactly).

  Cross-phase modulation (XPM): every OTHER channel's INSTANTANEOUS
  intensity also phase-modulates channel p, with the standard non-
  degenerate factor of 2 relative to SPM. Purely Kerr (not delayed) --
  XPM does not need any special-casing for CW channels, since it acts
  multiplicatively on each channel's own instantaneous power directly.

  Inter-channel Raman scattering ("Raman crosstalk" / "Raman tilt"):
  a mechanism distinct from XPM. Each channel's power creates real GAIN
  or LOSS in every other channel via the material's Raman gain spectrum
  evaluated AT THE FIXED CHANNEL-TO-CHANNEL CARRIER SEPARATION (not at
  each channel's own internal bandwidth, unlike the intra-channel
  conv(h_R, |A_p|^2) term) -- shorter-wavelength ("bluer") channels pump
  longer-wavelength ("redder") channels, exactly as in a Raman fiber
  amplifier. This works for CW/unmodulated channels (see the contrasting
  caveat in fiber.multimode_propagator's docstring, where the analogous
  intermodal term uses a co-located baseband convolution that gives
  zero for a CW driving channel -- that limitation does not apply here
  because this term uses the fixed carrier separation directly).

Per channel p (Delta_omega_pq = channel_offset_p - channel_offset_q):

    dA_p/dz = -alpha/2*A_p - i*Delta_beta1_p*dA_p/dt
              - i*(beta2/2)*d2A_p/dt2 + (beta3/6)*d3A_p/dt3
              + i*gamma*A_p*[(1-f_R)*(|A_p|^2 + 2*sum_{q!=p}|A_q|^2)
                             + f_R*(h_R (x) |A_p|^2)]
              + (1/2)*sum_{q!=p} g_R(Delta_omega_pq)*P_q(t)*A_p

Delta_beta1_p is the WDM channel walk-off from chromatic dispersion,
Delta_beta1(Delta_omega) = beta2*Delta_omega + 0.5*beta3*Delta_omega^2,
evaluated at each channel's own (large) carrier offset -- distinct from
the (small) internal Omega grid used for each channel's own dispersion
and delayed-Raman response.

Absolute inter-channel phase (beta0)
--------------------------------------
Everything above governs each channel's own envelope dynamics (power,
pulse shape, walk-off) and is correct without needing an absolute phase
reference. But if two channels will later be coherently combined --
e.g. a phase-encoded quantum channel interfered with a local oscillator
or a "twin" pulse that travelled a different wavelength -- the ABSOLUTE
relative phase between their carriers matters too, and that requires
going one order beyond delta_beta1: since delta_beta1(Delta_omega) IS
d(beta0)/d(Delta_omega) by definition, the accumulated relative phase
after length L is

    delta_beta0(Delta_omega)*L, with
    delta_beta0(Delta_omega) = beta1_ref*Delta_omega + 0.5*beta2*Delta_omega^2
                                + (1/6)*beta3*Delta_omega^3

beta1_ref = fiber.material.n_g/c is the fiber's absolute group index at
the reference wavelength (a new, explicit parameter -- everything else
in this codebase only ever needed RELATIVE dispersion, so no absolute
index was tracked before). This is a closed-form, exactly-linear-in-L
quantity (no numerical integration needed), applied once to the
propagated output rather than inside the per-step split-step loop --
see propagate()'s final phase multiply. It is a large, rapidly-wrapping
quantity for widely-spaced channels (physically correct: it is exactly
the optical beat phase between two different-wavelength carriers) and
is referenced to channel_offsets=0 (delta_beta0=0 there).

    from fiber.wdm_propagator import WDMPropagator
"""
import numpy as np

from fiber.raman_response import raman_response_freq_analytic, raman_gain_spectrum


class WDMPropagator:
    """Symmetric split-step multi-channel GNLSE solver for a FiberParams fiber.

    Parameters
    ----------
    fiber : fiber.fiber_params.FiberParams
    channel_offsets_Hz : array-like
        Carrier frequency offset of each channel (Hz) from the fiber's
        reference wavelength (fiber.omega0). E.g. a 100 GHz WDM grid:
        [-200e9, -100e9, 0, 100e9, 200e9].
    include_raman : bool
        Include both the intra-channel delayed Raman response and the
        inter-channel Raman crosstalk term (default True).
    """

    def __init__(self, fiber, channel_offsets_Hz, include_raman=True):
        self.fiber = fiber
        self.include_raman = include_raman
        self.channel_offsets = 2 * np.pi * np.asarray(channel_offsets_Hz, dtype=float)
        self.n_channels = len(self.channel_offsets)

        self.delta_beta1 = (fiber.beta2 * self.channel_offsets
                             + 0.5 * fiber.beta3 * self.channel_offsets ** 2)

        # absolute inter-channel phase (see module docstring): delta_beta0 is
        # the antiderivative of delta_beta1 with respect to channel_offsets,
        # plus the leading beta1_ref*Delta_omega term that delta_beta1 itself
        # doesn't carry (delta_beta1 is defined relative to the reference
        # channel's OWN beta1, not the absolute one)
        self.delta_beta0 = (fiber.beta1_ref * self.channel_offsets
                             + 0.5 * fiber.beta2 * self.channel_offsets ** 2
                             + (1.0 / 6.0) * fiber.beta3 * self.channel_offsets ** 3)

        Wp, Wq = np.meshgrid(self.channel_offsets, self.channel_offsets, indexing='ij')
        # channel_offsets_Hz is a direct physical-frequency offset (positive =
        # higher/"bluer"), but raman_gain_spectrum's Omega argument follows the
        # opposite convention (physical frequency = omega0 - Omega, established
        # in fiber.raman_response / fiber.propagator) -- so the argument here is
        # Wq - Wp, not Wp - Wq, to correctly evaluate the gain seen by channel p
        # from channel q at their true physical separation.
        Delta_omega_pq = Wq - Wp
        g_R_matrix = raman_gain_spectrum(fiber.material, Delta_omega_pq, fiber.gamma)
        np.fill_diagonal(g_R_matrix, 0.0)
        self.g_R_matrix = g_R_matrix if include_raman else np.zeros_like(g_R_matrix)

    def _prepare(self, Omega, dt):
        """Hook for subclasses to precompute per-frequency/per-channel arrays
        before the step loop starts. No-op in the base (purely classical)
        propagator -- see fiber.quantum_wdm.QuantumWDMPropagator."""
        pass

    def _noise_step(self, A_t, P, dz, dt, n_pts):
        """Hook returning a (N, n_pts) noise increment added after the
        nonlinear step each iteration. Zero in the base propagator."""
        return 0.0

    def propagate(self, A0, dt, L, step_size=50.0, n_steps=None):
        """Propagate channel fields through fiber length L (m).

        Parameters
        ----------
        A0 : complex array, shape (n_pts,) or (n_channels, n_pts)
            Launch field. A 1-D array is launched entirely into channel 0.
        dt, L, step_size, n_steps : as in fiber.propagator.FiberPropagator

        Returns
        -------
        complex array, shape (n_channels, n_pts)
        """
        N = self.n_channels
        A0 = np.asarray(A0)
        if A0.ndim == 1:
            A = np.zeros((N, len(A0)), dtype=complex)
            A[0] = A0
        else:
            if A0.shape[0] != N:
                raise ValueError(f"A0 has {A0.shape[0]} channel rows, expected n_channels={N}")
            A = A0.astype(complex).copy()

        n_pts = A.shape[1]
        if n_steps is None:
            n_steps = max(int(np.ceil(L / step_size)), 1)
        dz = L / n_steps

        Omega = 2 * np.pi * np.fft.fftfreq(n_pts, d=dt)
        f_R = self.fiber.material.f_R
        H_R = (raman_response_freq_analytic(self.fiber.material, Omega)
               if self.include_raman else None)

        base = (-self.fiber.alpha / 2
                + 1j * self.fiber.beta2 / 2 * Omega ** 2
                - 1j * self.fiber.beta3 / 6 * Omega ** 3)
        D_half = np.exp(
            (base[None, :] - 1j * self.delta_beta1[:, None] * Omega[None, :]) * dz / 2
        )  # (N, n_pts)

        gamma = self.fiber.gamma
        self._prepare(Omega, dt)
        A_f = np.fft.fft(A, axis=1)

        for _ in range(n_steps):
            A_f *= D_half
            A_t = np.fft.ifft(A_f, axis=1)

            P = np.abs(A_t) ** 2  # (N, n_pts)
            P_total = np.sum(P, axis=0, keepdims=True)  # (1, n_pts)

            if self.include_raman and f_R > 0:
                conv = np.real(np.fft.ifft(np.fft.fft(P, axis=1) * H_R[None, :], axis=1))
                spm = (1 - f_R) * P + f_R * conv
                xpm = (1 - f_R) * 2 * (P_total - P)
            else:
                spm = P
                xpm = 2 * (P_total - P)

            kerr_phase = gamma * (spm + xpm)                  # imaginary (phase) part
            raman_crosstalk = 0.5 * (self.g_R_matrix @ P)      # real (gain/loss) part

            A_t = A_t * np.exp((1j * kerr_phase + raman_crosstalk) * dz)
            A_t = A_t + self._noise_step(A_t, P, dz, dt, n_pts)

            A_f = np.fft.fft(A_t, axis=1) * D_half

        A_out = np.fft.ifft(A_f, axis=1)
        # absolute inter-channel phase, applied once as a closed-form final
        # multiply (see module docstring) -- needed only if channels will be
        # coherently combined later; harmless otherwise (pure phase, doesn't
        # touch power/pulse shape, so it never changes any power-based result)
        A_out = A_out * np.exp(1j * self.delta_beta0[:, None] * L)
        return A_out
