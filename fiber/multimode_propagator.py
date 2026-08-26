"""
Multimode GNLSE propagator: symmetric split-step solver for a set of
coupled principal-mode-group envelopes A_p(z,t), p=0..M-1, through a
MultimodeFiberParams fiber (see fiber.multimode_fiber).

Per mode group p:

    dA_p/dz = -alpha/2*A_p - i*Delta_beta1_p*dA_p/dt
              - i*(beta2/2)*d2A_p/dt2 + (beta3/6)*d3A_p/dt3
              + i*gamma_pp*A_p*P_self,p
              + i*sum_{q!=p} gamma_pq*A_p*[2*(1-f_R)*|A_q|^2 + f_R*(h_R (x) |A_q|^2)]

Delta_beta1_p is the intermodal (differential-mode) group-delay term --
a per-mode linear-in-Omega phase ramp that walks each mode group's pulse
relative to the reference (index 0) mode group, the mechanism behind
graded-index fiber's DMD-limited bandwidth. alpha, beta2, and beta3 are
ALSO per-mode (see fiber.multimode_fiber.MultimodeFiberParams): alpha
carries differential mode attenuation (higher-order groups lossier),
and beta2/beta3 carry a mode-dependent waveguide-dispersion correction
on top of the shared material dispersion -- all three, together with
Delta_beta1_p, are referenced to mode 0 so a single-mode-only launch
reproduces fiber.propagator.FiberPropagator exactly.

P_self,p = (1-f_R)*|A_p|^2 + f_R*(h_R (x) |A_p|^2) is the same intramodal
Kerr+Raman nonlinear polarization as fiber.propagator.FiberPropagator.
The sum over q!=p is the intermodal coupling: mode group q's own POWER
ENVELOPE |A_q(t)|^2 creates a nonlinear (instantaneous Kerr + delayed
Raman) polarization that phase-modulates mode group p, weighted by the
gamma_pq cross-coupling coefficient (see MultimodeFiberParams.gamma_matrix
-- an intensity-overlap model, not a full 4-index coherent overlap
integral, so genuine phase-matched four-wave-mixing between mode groups
is out of scope here).

Important: this term only transfers *time-varying* power. A perfectly
CW/constant-power pump in mode q has zero spectral content away from
Omega=0, so conv(h_R, |A_q|^2) is exactly constant regardless of h_R's
shape -- convolving a delta-function spectrum with any response can only
ever produce more delta-function-at-DC output. Intermodal Raman transfer
here requires the driving mode to be a genuine pulse (or otherwise have
real temporal/spectral structure): a strong pulse in mode q spectrally
broadens and red-shifts a weaker, temporally-overlapped pulse in mode p
through exactly this mechanism (see tests/test_multimode_fiber.py, check
4) -- but a monochromatic CW pump in mode q will NOT Raman-amplify an
already-CW probe sitting in mode p at the Stokes offset the way a
same-mode two-tone pump/probe does in fiber.propagator.FiberPropagator
(there, the pump and probe share one field, so their coherent beat note
in |A|^2 is what seeds the delayed response at the probe's own
frequency -- that beat has no counterpart here since |A_q|^2 and |A_p|^2
are computed independently per mode).

Launching a field into a single mode group with gamma_matrix's off-
diagonal terms unused (no other mode populated) reduces exactly to
fiber.propagator.FiberPropagator's physics -- verified in
tests/test_multimode_fiber.py.

The output also carries each mode group's ABSOLUTE propagation-constant
offset (delta_beta0*L, see fiber.multimode_fiber) -- needed to correctly
combine/interfere two mode groups later (e.g. a phase-encoded signal in
one mode with a reference field in another); it does not affect any
power-based quantity (pulse shape, spectrum, energy).

    from fiber.multimode_propagator import MultimodeFiberPropagator
"""
import numpy as np

from fiber.raman_response import raman_response_freq_analytic


class MultimodeFiberPropagator:
    """Symmetric split-step multimode GNLSE solver for a MultimodeFiberParams fiber.

    Parameters
    ----------
    fiber : fiber.multimode_fiber.MultimodeFiberParams
    include_raman : bool -- include the delayed Raman response (default True)
    """

    def __init__(self, fiber, include_raman=True):
        self.fiber = fiber
        self.include_raman = include_raman

    def _linear_coupling_step(self, A_t, dz):
        """Hook for subclasses to apply a per-step LINEAR mode-mixing
        operator (e.g. random coupling from bends/splices -- a mechanism
        distinct from the nonlinear coupling above). No-op here -- see
        fiber.mode_coupling.RandomModeCouplingPropagator."""
        return A_t

    def _prepare(self, Omega, dt):
        """Hook for subclasses to precompute per-frequency/per-mode arrays
        before the step loop starts. No-op in the base (purely classical)
        propagator -- see fiber.quantum_multimode.QuantumMultimodePropagator."""
        pass

    def _noise_step(self, A_t, P, dz, dt, n_pts):
        """Hook returning a (M, n_pts) noise increment added after the
        nonlinear step each iteration. Zero in the base propagator."""
        return 0.0

    def propagate(self, A0, dt, L, step_size=20.0, n_steps=None):
        """Propagate mode-group fields through fiber length L (m).

        Parameters
        ----------
        A0 : complex array, shape (n_pts,) or (n_modes, n_pts)
            Launch field. A 1-D array is launched entirely into mode
            group 0 (the fundamental-like group); a 2-D array gives each
            mode group's own launch field directly.
        dt : float -- time step (s)
        L : float -- propagation length (m)
        step_size : float -- spatial step size (m), default 20 m (finer
            than the single-mode default since intermodal walk-off needs
            a shorter step to stay resolved)
        n_steps : int, optional -- overrides step_size if given

        Returns
        -------
        complex array, shape (n_modes, n_pts) -- output field per mode group
        """
        M = self.fiber.n_modes
        A0 = np.asarray(A0)
        if A0.ndim == 1:
            A = np.zeros((M, len(A0)), dtype=complex)
            A[0] = A0
        else:
            if A0.shape[0] != M:
                raise ValueError(f"A0 has {A0.shape[0]} mode rows, fiber has n_modes={M}")
            A = A0.astype(complex).copy()

        n_pts = A.shape[1]
        if n_steps is None:
            n_steps = max(int(np.ceil(L / step_size)), 1)
        dz = L / n_steps

        Omega = 2 * np.pi * np.fft.fftfreq(n_pts, d=dt)
        f_R = self.fiber.material.f_R
        H_R = (raman_response_freq_analytic(self.fiber.material, Omega)
               if self.include_raman else None)

        # every term here is per-mode (shape (M,1) broadcasting against Omega's
        # (1,n_pts)): loss, GVD, TOD, and group-delay walk-off all vary by
        # mode group (see fiber.multimode_fiber.MultimodeFiberParams)
        base = (-self.fiber.alpha[:, None] / 2
                + 1j * self.fiber.beta2[:, None] / 2 * Omega[None, :] ** 2
                - 1j * self.fiber.beta3[:, None] / 6 * Omega[None, :] ** 3
                - 1j * self.fiber.delta_beta1[:, None] * Omega[None, :])
        D_half = np.exp(base * dz / 2)  # (M, n_pts)

        gamma_diag = np.diag(self.fiber.gamma_matrix)[:, None]      # (M, 1)
        gamma_off = self.fiber.gamma_matrix - np.diag(np.diag(self.fiber.gamma_matrix))  # (M, M)

        self._prepare(Omega, dt)
        A_f = np.fft.fft(A, axis=1)

        for _ in range(n_steps):
            A_f *= D_half
            A_t = np.fft.ifft(A_f, axis=1)

            P = np.abs(A_t) ** 2  # (M, n_pts)
            if self.include_raman and f_R > 0:
                conv = np.real(np.fft.ifft(np.fft.fft(P, axis=1) * H_R[None, :], axis=1))
                P_self = (1 - f_R) * P + f_R * conv
                cross_source = 2 * (1 - f_R) * P + f_R * conv
            else:
                P_self = P
                cross_source = 2 * P

            phase = gamma_diag * P_self + gamma_off @ cross_source
            A_t = A_t * np.exp(1j * phase * dz)
            A_t = A_t + self._noise_step(A_t, P, dz, dt, n_pts)
            A_t = self._linear_coupling_step(A_t, dz)

            A_f = np.fft.fft(A_t, axis=1) * D_half

        A_out = np.fft.ifft(A_f, axis=1)
        # absolute inter-mode phase, applied once as a closed-form final
        # multiply (see fiber.multimode_fiber's "Absolute inter-mode phase"
        # docstring section) -- needed only if mode groups will be
        # coherently combined later; pure phase, so it never touches any
        # power-based result
        A_out = A_out * np.exp(1j * self.fiber.delta_beta0[:, None] * L)
        return A_out
