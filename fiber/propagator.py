"""
Classical GNLSE fiber propagator (symmetric split-step Fourier method).

Solves the generalized nonlinear Schrodinger equation for the field
envelope A(z,t), with |A|^2 in Watts:

    dA/dz = -alpha/2*A - i*(beta2/2)*d2A/dt2 + (beta3/6)*d3A/dt3
            + i*gamma*A*[(1-f_R)*|A|^2 + f_R*(h_R (x) |A|^2)]

The delayed Raman convolution h_R (x) |A|^2 is applied in the frequency
domain via the fiber material's analytic Raman response H_R(Omega) (see
fiber.raman_response) -- this adds stimulated Raman scattering and the
Raman-induced spectral self-frequency shift on top of plain Kerr SPM.
Setting include_raman=False (or f_R=0 on the material) recovers the
pure-Kerr NLSE used by the original studies/fiber_propagation.py script.

Each spatial step is: half-step linear operator (loss + dispersion) ->
full nonlinear step -> half-step linear operator. Subclasses (see
fiber.quantum_noise) can hook in an additional per-step noise term without
touching the dispersion/nonlinear code, via _prepare()/_noise_step().

    from fiber.propagator import FiberPropagator
"""
import numpy as np

from fiber.raman_response import raman_response_freq_analytic


class FiberPropagator:
    """Symmetric split-step GNLSE solver for a given FiberParams.

    Parameters
    ----------
    fiber : fiber.fiber_params.FiberParams
    include_raman : bool
        Include the delayed Raman response (default True). Has no effect
        if fiber.material.f_R == 0.
    """

    def __init__(self, fiber, include_raman=True):
        self.fiber = fiber
        self.include_raman = include_raman

    def _prepare(self, Omega, dt):
        """Hook for subclasses to precompute per-frequency arrays before the
        step loop starts. No-op in the base (purely classical) propagator."""
        pass

    def _noise_step(self, A_t, dz, dt, n_pts):
        """Hook returning a time-domain field increment added after the
        nonlinear step each iteration. Zero in the base propagator."""
        return 0.0

    def _raman_power(self, P, H_R):
        f_R = self.fiber.material.f_R
        if not (self.include_raman and f_R > 0):
            return P
        P_raman = np.real(np.fft.ifft(np.fft.fft(P) * H_R))
        return (1 - f_R) * P + f_R * P_raman

    def propagate(self, A0, dt, L, step_size=50.0, n_steps=None):
        """Propagate A0(t) through fiber length L (m).

        Parameters
        ----------
        A0 : complex array -- input field envelope, |A0|^2 in Watts
        dt : float -- time step (s)
        L : float -- propagation length (m)
        step_size : float -- spatial step size (m), default 50 m
        n_steps : int, optional -- overrides step_size if given

        Returns
        -------
        complex array -- output field envelope
        """
        n_pts = len(A0)
        if n_steps is None:
            n_steps = max(int(np.ceil(L / step_size)), 1)
        dz = L / n_steps

        Omega = 2 * np.pi * np.fft.fftfreq(n_pts, d=dt)
        H_R = (raman_response_freq_analytic(self.fiber.material, Omega)
               if self.include_raman else None)
        self._prepare(Omega, dt)

        D_half = np.exp((-self.fiber.alpha / 2
                          + 1j * self.fiber.beta2 / 2 * Omega ** 2
                          - 1j * self.fiber.beta3 / 6 * Omega ** 3) * dz / 2)

        A_f = np.fft.fft(A0.astype(complex))

        for _ in range(n_steps):
            A_f *= D_half
            A_t = np.fft.ifft(A_f)

            P_nl = self._raman_power(np.abs(A_t) ** 2, H_R)
            A_t = A_t * np.exp(1j * self.fiber.gamma * P_nl * dz)
            A_t = A_t + self._noise_step(A_t, dz, dt, n_pts)

            A_f = np.fft.fft(A_t) * D_half

        return np.fft.ifft(A_f)
