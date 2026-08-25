"""
Stimulated Brillouin scattering (SBS): steady-state (CW/quasi-CW) coupled
pump/Stokes power equations, solved as a two-point boundary value problem
by shooting, plus the standard analytic threshold formula.

Physics
-------
SBS couples a forward-propagating pump to a backward-propagating Stokes
wave through electrostriction-driven ACOUSTIC phonons -- a mechanism
distinct from Raman scattering (which involves optical phonons):

    dP_p/dz = -(g_B/A_eff)*P_p*P_s - alpha*P_p
    dP_s/dz = -(g_B/A_eff)*P_p*P_s + alpha*P_s

The standard tabulated g_B (units m/W) is an *intensity*-gain coefficient
(dI_p/dz = -g_B*I_p*I_s); converting to the *power* equations above (P
rather than I) needs the extra 1/A_eff, exactly as fiber.gamma already
divides by A_eff to turn n2 into a power-based Kerr coefficient.

g_B is the (on-resonance) Brillouin gain coefficient; off resonance it
follows the material's Lorentzian gain spectrum (see
FiberMaterial.brillouin_gain), with a linewidth delta_nu_B of order tens
of MHz -- roughly 1000x narrower than the Raman gain spectrum, since
acoustic phonons live far longer (~ns) than optical phonons (~ps).
Correspondingly the Brillouin frequency shift (~10-11 GHz at 1550 nm in
silica) is roughly 1000x smaller than the Raman shift (~13 THz), and SBS
is predominantly BACKWARD-scattering (whereas Raman gain in a fiber is
predominantly forward) with a much LOWER threshold power for CW/narrow-
linewidth sources -- typically a few mW over tens of km, versus ~1 W for
Raman.

The Brillouin gain linewidth is far too narrow to resolve on the fs-ps
time grids fiber.propagator/fiber.wdm_propagator use (resolving 30 MHz
would need a >30 ns simulation window) -- SBS is modelled here instead
as a separate, standalone steady-state power-domain problem, the
standard treatment for CW/quasi-CW SBS threshold and gain analysis.

Because the Stokes wave counter-propagates, its boundary condition is
set at z=L (not z=0): P_s(L) = P_noise, a small effective input power
representing spontaneous phonon-scattering noise seeding backscatter
from every point along the fiber (see spontaneous_brillouin_noise_power).
This makes the problem a two-point boundary value problem, solved here
by shooting on the unknown P_s(0): guess a value, integrate both
equations forward from z=0 to z=L, and adjust the guess (bisection in
log-space, since Stokes power typically spans tens of orders of
magnitude between the noise floor and threshold) until the computed
P_s(L) matches P_noise.

    from fiber.brillouin import BrillouinPropagator, sbs_threshold_power, spontaneous_brillouin_noise_power
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

hbar = 1.0545718e-34  # J.s


def spontaneous_brillouin_noise_power(fiber):
    """Effective input power (W) seeding the backward Stokes wave from
    spontaneous phonon scattering: ~one photon per mode over the
    Brillouin gain bandwidth, P_noise = hbar*omega0*delta_nu_B."""
    return hbar * fiber.omega0 * fiber.material.delta_nu_B


def sbs_threshold_power(fiber, L, K=21.0):
    """Analytic CW SBS threshold power (W): P_th = K*A_eff/(g_B*L_eff).

    Standard engineering formula (Smith, Applied Optics, 1972); K~21 is
    the commonly used value accounting for polarization/spectral
    averaging in a typical fiber. L_eff = (1-exp(-alpha*L))/alpha is the
    effective (loss-limited) interaction length.
    """
    alpha = fiber.alpha
    L_eff = (1 - np.exp(-alpha * L)) / alpha if alpha > 0 else L
    g_B = fiber.material.g_B
    A_eff = fiber.geometry.A_eff
    return K * A_eff / (g_B * L_eff)


class BrillouinPropagator:
    """Steady-state (CW/quasi-CW) coupled pump/Stokes SBS solver.

    Parameters
    ----------
    fiber : fiber.fiber_params.FiberParams
    detuning_Hz : float
        Offset (Hz) of the pump-Stokes frequency separation from the
        exact Brillouin resonance (0 = on-resonance, i.e. separation =
        material.nu_B). Sets the effective gain via
        FiberMaterial.brillouin_gain.
    """

    def __init__(self, fiber, detuning_Hz=0.0):
        self.fiber = fiber
        self.g_B_eff = fiber.material.brillouin_gain(detuning_Hz)
        self.g_over_Aeff = self.g_B_eff / fiber.geometry.A_eff

    def _rhs(self, z, y):
        Pp, Ps = max(y[0], 0.0), max(y[1], 0.0)
        alpha = self.fiber.alpha
        dPp = -self.g_over_Aeff * Pp * Ps - alpha * Pp
        dPs = -self.g_over_Aeff * Pp * Ps + alpha * Ps
        return [dPp, dPs]

    def _integrate(self, P_p0, P_s0, L, n_eval=400):
        z_eval = np.linspace(0, L, n_eval)
        sol = solve_ivp(self._rhs, [0, L], [P_p0, P_s0], t_eval=z_eval,
                         method='LSODA', rtol=1e-8, atol=1e-30)
        return sol.t, np.clip(sol.y[0], 0, None), np.clip(sol.y[1], 0, None)

    def solve(self, P_pump_in, L, P_noise=None, n_eval=400):
        """Solve the SBS boundary value problem.

        Parameters
        ----------
        P_pump_in : float -- pump power (W) launched at z=0
        L : float -- fiber length (m)
        P_noise : float, optional -- Stokes boundary value at z=L
            (default: spontaneous_brillouin_noise_power(fiber))

        Returns
        -------
        z, P_pump(z), P_stokes(z) : arrays
        """
        if P_noise is None:
            P_noise = spontaneous_brillouin_noise_power(self.fiber)
        if P_pump_in <= 0:
            z = np.linspace(0, L, n_eval)
            return z, np.zeros(n_eval), np.zeros(n_eval)

        def mismatch(log10_Ps0):
            _, _, Ps = self._integrate(P_pump_in, 10 ** log10_Ps0, L, n_eval=50)
            return Ps[-1] - P_noise

        lo, hi = -30.0, np.log10(P_pump_in)
        f_lo, f_hi = mismatch(lo), mismatch(hi)
        tries = 0
        while f_lo * f_hi > 0 and tries < 30:
            hi += 3.0
            f_hi = mismatch(hi)
            tries += 1
        if f_lo * f_hi > 0:
            raise RuntimeError("Could not bracket the SBS shooting solution; "
                                "check fiber parameters (gain too weak/strong?).")

        log10_Ps0 = brentq(mismatch, lo, hi, xtol=1e-4)
        return self._integrate(P_pump_in, 10 ** log10_Ps0, L, n_eval=n_eval)

    def reflectivity(self, P_pump_in, L, **kwargs):
        """Backscattered fraction P_stokes(0) / P_pump_in for the given launch."""
        _, Pp, Ps = self.solve(P_pump_in, L, **kwargs)
        return Ps[0] / P_pump_in
