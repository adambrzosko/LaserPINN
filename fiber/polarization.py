"""
Polarization effects for single-mode fiber: a 2-component (Jones vector)
GNLSE propagator carrying polarization mode dispersion (PMD) and
polarization-dependent cross-phase modulation / Raman gain -- all absent
from fiber.propagator.FiberPropagator and fiber.wdm_propagator.WDMPropagator,
which are scalar (implicitly single-, co-polarized-field) models.

Per polarization component p in {x, y} (q the other component):

    dA_p/dz = -alpha/2*A_p - i*(beta2/2)*d2A_p/dt2 + (beta3/6)*d3A_p/dt3
              + i*gamma*A_p*[(1-f_R)*(|A_p|^2 + (2/3)*|A_q|^2)
                             + f_R*(h_R (x) |A_p|^2)
                             + raman_copol_ratio*f_R*(h_R (x) |A_q|^2)]
              + PMD (see below)

Cross-phase modulation
-----------------------
The 2/3 factor for orthogonally-polarized XPM (vs. the factor of 2 for
co-polarized/different-wavelength XPM elsewhere in this codebase) is a
standard, well-established result from the symmetry of the silica
chi^(3) tensor for an isotropic medium (see e.g. Agrawal, Nonlinear
Fiber Optics, "vector NLSE" treatment) -- unlike several other
parameters in this codebase, this one is not a representative/
approximate placeholder.

Cross-polarization Raman
--------------------------
Raman gain between orthogonally-polarized pump and probe is measurably
SMALLER than co-polarized Raman gain in silica; raman_copol_ratio (0-1)
scales the cross-polarization Raman convolution relative to the co-
polarized one. Unlike the XPM factor, this ratio does not have one
universally quoted value (it depends on detuning and measurement
conditions) -- the default here is a representative order-of-magnitude
placeholder; calibrate against a measurement if you need precision.

Polarization mode dispersion (PMD)
-------------------------------------
Modelled with the standard "coarse-step" method used throughout the PMD-
emulation literature: each propagation step is treated as a short
birefringent segment with a random axis orientation (a random SU(2)
rotation of the Jones vector) and a FIXED local differential group delay
(DGD) between its own eigen-axes. Because consecutive segments'
orientations are independent random draws, the accumulated DGD after
distance L executes an isotropic random walk, giving the standard
RMS(DGD) ~ D_PMD*sqrt(L) scaling (D_PMD in ps/sqrt(km), the usual
telecom PMD specification unit) rather than growing linearly with L --
calibrated here by setting the per-step local DGD to D_PMD*sqrt(dz)
(so that summing N independent random-orientation steps of that local
DGD, each contributing an uncorrelated random 3-vector of fixed length
to the accumulated PMD vector, reproduces D_PMD^2*L after distance
L=N*dz -- verified statistically, across many realizations, in
tests/test_polarization.py). Both the rotation and the DGD phase are
unitary/pure-phase operations, so total power (summed over both
components, integrated over time) is conserved exactly at every step,
also verified there.

    from fiber.polarization import PolarizationPropagator
"""
import numpy as np

from fiber.raman_response import raman_response_freq_analytic

from fiber.constants import hbar


class PolarizationPropagator:
    """2-component (Jones vector) split-step GNLSE solver for a FiberParams
    fiber, with PMD and polarization-dependent XPM/Raman.

    Parameters
    ----------
    fiber : fiber.fiber_params.FiberParams
    D_PMD : float
        PMD parameter (ps/sqrt(km)), the standard telecom specification
        unit. Typical modern fiber ~0.01-0.1; older/legacy fiber can be
        ~0.5-1 or worse. Default 0 (no PMD).
    raman_copol_ratio : float
        Cross-polarization Raman gain relative to co-polarized (0-1);
        see module docstring. Default 0.5, a representative placeholder.
    seed : int or None -- RNG seed for the random PMD axis rotations
    include_raman : bool
    """

    def __init__(self, fiber, D_PMD=0.0, raman_copol_ratio=0.5, seed=None, include_raman=True):
        self.fiber = fiber
        self.include_raman = include_raman
        self.raman_copol_ratio = raman_copol_ratio
        self.D_PMD_SI = D_PMD * 1e-12 / np.sqrt(1e3)  # ps/sqrt(km) -> s/sqrt(m)
        self.rng = np.random.default_rng(seed)

    def _random_su2(self):
        """A Haar-random 2x2 unitary matrix (random polarization-axis
        orientation for this segment)."""
        a, b, c, d = self.rng.standard_normal(4)
        alpha_c = a + 1j * b
        beta_c = c + 1j * d
        norm = np.sqrt(abs(alpha_c) ** 2 + abs(beta_c) ** 2)
        alpha_c, beta_c = alpha_c / norm, beta_c / norm
        return np.array([[alpha_c, -np.conj(beta_c)],
                          [beta_c, np.conj(alpha_c)]])

    def _pmd_step(self, A_t, Omega, dz):
        if self.D_PMD_SI <= 0:
            return A_t
        R = self._random_su2()
        A_rot = R @ A_t  # (2, n_pts), random local axis orientation

        dgd = self.D_PMD_SI * np.sqrt(dz)
        A_rot_f = np.fft.fft(A_rot, axis=1)
        A_rot_f[0] *= np.exp(-1j * (dgd / 2) * Omega)
        A_rot_f[1] *= np.exp(1j * (dgd / 2) * Omega)
        return np.fft.ifft(A_rot_f, axis=1)

    def propagate(self, A0, dt, L, step_size=50.0, n_steps=None):
        """Propagate the Jones vector field through fiber length L (m).

        Parameters
        ----------
        A0 : complex array, shape (n_pts,) or (2, n_pts)
            Launch field. A 1-D array is launched entirely into the x
            component (A0, [0,0,...]).
        dt, L, step_size, n_steps : as in fiber.propagator.FiberPropagator

        Returns
        -------
        complex array, shape (2, n_pts) -- [A_x, A_y]
        """
        A0 = np.asarray(A0)
        if A0.ndim == 1:
            A = np.zeros((2, len(A0)), dtype=complex)
            A[0] = A0
        else:
            if A0.shape[0] != 2:
                raise ValueError(f"A0 must have shape (n_pts,) or (2, n_pts), got {A0.shape}")
            A = A0.astype(complex).copy()

        n_pts = A.shape[1]
        if n_steps is None:
            n_steps = max(int(np.ceil(L / step_size)), 1)
        dz = L / n_steps

        Omega = 2 * np.pi * np.fft.fftfreq(n_pts, d=dt)
        f_R = self.fiber.material.f_R
        H_R = (raman_response_freq_analytic(self.fiber.material, Omega)
               if self.include_raman else None)

        base = (-self.fiber.alpha_at(self.fiber.omega0 - Omega) / 2
                + 1j * self.fiber.beta2 / 2 * Omega ** 2
                - 1j * self.fiber.beta3 / 6 * Omega ** 3)
        D_half = np.exp(base * dz / 2)  # (n_pts,), shared by both components

        gamma = self.fiber.gamma
        A_f = np.fft.fft(A, axis=1)

        for _ in range(n_steps):
            A_f *= D_half[None, :]
            A_t = np.fft.ifft(A_f, axis=1)

            P = np.abs(A_t) ** 2  # (2, n_pts)
            P_cross = P[::-1]      # (2, n_pts): P[1] paired with component 0, P[0] with 1

            if self.include_raman and f_R > 0:
                conv = np.real(np.fft.ifft(np.fft.fft(P, axis=1) * H_R[None, :], axis=1))
                conv_cross = conv[::-1]
                spm = (1 - f_R) * P + f_R * conv
                xpm = (1 - f_R) * (2.0 / 3.0) * P_cross + self.raman_copol_ratio * f_R * conv_cross
            else:
                spm = P
                xpm = (2.0 / 3.0) * P_cross

            phase = gamma * (spm + xpm)
            A_t = A_t * np.exp(1j * phase * dz)

            A_t = self._pmd_step(A_t, Omega, dz)

            A_f = np.fft.fft(A_t, axis=1) * D_half[None, :]

        return np.fft.ifft(A_f, axis=1)
