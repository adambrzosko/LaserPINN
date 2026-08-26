"""
Elastic Rayleigh backscattering: linear, non-stimulated, always-present
scattering from refractive-index microinhomogeneities frozen into the
glass during fabrication -- distinct from stimulated Brillouin
scattering (fiber.brillouin), which is a threshold-driven, narrow-
linewidth, acoustic-phonon-mediated process. Rayleigh scattering has NO
threshold and NO frequency shift: it simply redirects a small, constant
fraction of the local forward power into the backward-guided mode at
every point along the fiber, proportional to that local power. This is
exactly the physics behind OTDR (optical time-domain reflectometry).

Physics
-------
alpha_R (1/m) is the part of the fiber's total attenuation alpha due to
Rayleigh scattering specifically (it typically dominates alpha in modern
telecom-grade silica fiber). S (dimensionless) is the fraction of
Rayleigh-scattered light captured into the backward-guided mode (vs.
scattered out of the waveguide entirely). At position z, forward power
P_in*exp(-alpha*z) generates backscattered power at rate
alpha_R*S*P_fwd(z) per unit length; that backscattered light then
travels back to z=0, picking up a further exp(-alpha*z) of attenuation.
Integrating over the fiber:

    P_back_total(L) = alpha_R*S*P_in * (1 - exp(-2*alpha*L)) / (2*alpha)

-- saturating (not growing without bound) as L increases, since light
scattered far down the fiber is itself heavily attenuated on the way
back. For a pulsed source, the time-resolved backscatter signal (the
classic OTDR trace) is

    P_back(t) = alpha_R*S*P_in*(v_g/2)*exp(-2*alpha*v_g*t/2),  z=v_g*t/2

both derived directly (not independently referenced) from the same
local-generation-plus-double-pass-attenuation picture, and cross-checked
against each other and against brute-force numerical integration in
tests/test_rayleigh_backscatter.py.

Approximation
-------------
alpha_R and S are new FiberMaterial/FiberGeometry-level parameters with
representative, order-of-magnitude default values (alpha_R as a fraction
of total alpha; S from a simple weakly-guiding-fiber capture estimate) --
neither has the kind of standardised reference figure OFL/EMB bandwidth
has for DMD. Calibrate against a measured OTDR backscatter coefficient
for your fiber if you need precise absolute predictions; the FORMULAS
(saturation with L, double-pass attenuation, the OTDR trace shape) are
on much firmer ground than these two specific numbers.

    from fiber.rayleigh_backscatter import rayleigh_backscatter_power, rayleigh_otdr_trace
"""
import numpy as np

# Representative defaults, not manufacturer/measured values (see module docstring).
DEFAULT_ALPHA_R_FRACTION = 0.9   # Rayleigh scattering's share of total attenuation in modern SMF
DEFAULT_S_CAPTURE = 2e-3         # backscatter capture fraction, typical order of magnitude for SMF


def rayleigh_backscatter_power(fiber, P_in, L, alpha_R_fraction=DEFAULT_ALPHA_R_FRACTION,
                                S_capture=DEFAULT_S_CAPTURE):
    """Total Rayleigh-backscattered power (W) arriving at the input (z=0)
    from a CW/quasi-CW forward launch power P_in (W) over fiber length L (m).

    Parameters
    ----------
    fiber : fiber.fiber_params.FiberParams
    P_in : float -- forward launch power (W)
    L : float -- fiber length (m)
    alpha_R_fraction : float -- Rayleigh's share of fiber.alpha (0-1)
    S_capture : float -- backscatter capture fraction (0-1)
    """
    alpha = fiber.alpha
    alpha_R = alpha_R_fraction * alpha
    if alpha <= 0:
        return alpha_R * S_capture * P_in * L
    return alpha_R * S_capture * P_in * (1 - np.exp(-2 * alpha * L)) / (2 * alpha)


def rayleigh_otdr_trace(fiber, P_in, L, t, v_g, alpha_R_fraction=DEFAULT_ALPHA_R_FRACTION,
                         S_capture=DEFAULT_S_CAPTURE):
    """Time-resolved backscattered power (W) arriving at the input at
    round-trip return time t (s) -- the classic OTDR trace -- for a
    launch pulse effectively CW/flat-top over the relevant window.

    Parameters
    ----------
    fiber : fiber.fiber_params.FiberParams
    P_in : float -- forward launch power (W)
    L : float -- fiber length (m); trace is zero for t beyond the round trip to L
    t : float or array -- round-trip return time(s) (s)
    v_g : float -- group velocity (m/s)
    alpha_R_fraction, S_capture : as in rayleigh_backscatter_power

    Returns
    -------
    float or array -- backscattered power (W) at each t
    """
    alpha = fiber.alpha
    alpha_R = alpha_R_fraction * alpha
    t = np.asarray(t, dtype=float)
    z = v_g * t / 2
    P_back = alpha_R * S_capture * P_in * (v_g / 2) * np.exp(-2 * alpha * z)
    return np.where((z >= 0) & (z <= L), P_back, 0.0)
