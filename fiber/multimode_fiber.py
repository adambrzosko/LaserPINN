"""
Multimode graded-index fiber support (OM1-OM5): geometry, principal
mode-group decomposition, and the per-mode-group / cross-mode-group
parameters MultimodeFiberPropagator needs for intermodal dispersion and
intermodal Raman scattering.

    from fiber.multimode_fiber import MultimodeFiberGeometry, MultimodeFiberParams, make_multimode_fiber

Model
-----
Full per-LP-mode simulation of a graded-index fiber is impractical for a
general tool -- an OM3/OM4/OM5-class 50/125 fiber guides on the order of
a few hundred scalar modes. This module uses the standard reduced-order
description instead: modes are grouped into "principal mode groups"
p=1..M (the same near-degenerate grouping used in industry differential-
mode-delay/EMB characterisation), each treated as one effective spatial
channel with its own group delay and effective area.

Intermodal (differential mode) delay
-------------------------------------
Uses a graded-index (alpha-profile) group-delay model built from the
classic Olshansky & Keck (1976) leading-order term -- which correctly
predicts zero delay spread at the ideal parabolic profile (alpha=2) and
growing spread as alpha departs from 2, the mechanism OM-grade bandwidth
differences (OM1 -> OM5) are built on -- plus a quadratic-in-Delta
"floor" term so that even a perfectly-made alpha=2 fiber has the finite,
nonzero bandwidth real graded-index fiber does (in reality this floor
comes from the profile's material-dispersion-corrected optimum alpha,
which needs Sellmeier data this module doesn't have; here it's a single
calibrated constant instead -- see KAPPA_FLOOR below):

    tau(p)/L = (n1/c) * [ Delta*(alpha-2)/(alpha+2)*x^(2a/(a+2))
                           + Delta^2*KAPPA_FLOOR*x^(4a/(a+2)) ],   x = p/M

This is a simplified, order-of-magnitude-correct model, NOT a
reproduction of manufacturer DMD measurements: alpha_profile per OM grade
here is chosen so this model's own estimated_bandwidth() lands near the
nominal datasheet EMB/OFL figure (stored as ofl_bandwidth_MHz_km, for
reference), not because real OM1-OM5 fiber is manufactured at exactly
that alpha.

Spatial overlap between mode groups
------------------------------------
Needed for intermodal Kerr XPM and intermodal Raman scattering; true
LP-mode overlap integrals require a full vector mode solve, so this uses
a simplified, monotonically-decaying-with-separation model instead (see
MultimodeFiberParams.overlap) -- also documented there.

Per-mode loss (differential mode attenuation, DMA)
----------------------------------------------------
Higher-order mode groups are less tightly confined and generally see
more bend/confinement loss than the fundamental -- a real, measured
effect (alongside DMD) in fiber characterisation. Modelled here with the
same reduced-order philosophy as the delay spread: a power-law growth
of loss with normalised mode-group index x=p/M, referenced to the
fundamental (mode index 0) so alpha_dB_km remains exactly the
fundamental's loss:

    alpha(p) = alpha_0 * [1 + dma_fraction*(x_p^dma_exponent - x_1^dma_exponent)]

dma_fraction/dma_exponent are geometry fields, calibrated per OM grade
only in the sense that better (more bend-insensitive, laser-optimized)
grades get a smaller dma_fraction -- these are representative,
order-of-magnitude values, not manufacturer DMA specs; there is no
widely standardised reference figure for DMA the way OFL/EMB bandwidth
is standardised for DMD.

Per-mode waveguide dispersion (beta2, beta3)
-----------------------------------------------
The bulk MATERIAL dispersion (from the D field, as in FiberParams) is
common to all mode groups -- they all sample essentially the same core
glass. What varies by mode group is the WAVEGUIDE dispersion: since
beta1(p) [[the group delay above]] is itself a function of frequency
through the mode count M(omega) (the number of guided principal mode
groups grows with omega), and beta2 = d(beta1)/d(omega), beta3 =
d^2(beta1)/d(omega^2) BY DEFINITION, the mode-dependent part of beta2/
beta3 is obtained here by finite-differencing the SAME delay formula in
omega around the design wavelength -- reusing the already-validated DMD
physics instead of introducing new hand-tuned constants. This is added
on top of the shared material beta2/beta3 (referenced to mode 0, so
mode 0 reproduces the scalar single-mode behaviour exactly).

Absolute inter-mode phase (beta0)
------------------------------------
delta_beta1(p) above governs each mode group's own envelope dynamics and
needs no absolute reference. But coherently combining two mode groups
(e.g. interfering a phase-encoded signal in one mode with a local
oscillator or reference field in another) needs their ABSOLUTE relative
phase too. Since beta1(p,omega) = tau_over_L(p,omega) is beta0(p,omega)'s
frequency DERIVATIVE by definition, beta0 itself follows directly from
the SAME leading-order WKB relation used for delay (Olshansky & Keck):

    beta0(p,omega) = n1*(omega/c) * [1 - Delta*x_p^(2*alpha/(alpha+2))],   x_p = p/M(omega)

(this is an exact antiderivative check, not just dimensional analysis:
differentiating this beta0(p,omega) with respect to omega reproduces
_tau_over_L's leading term exactly -- verified in
tests/test_multimode_fiber.py). delta_beta0(p) = beta0(p,omega0) -
beta0(1,omega0) is a per-mode, z-INDEPENDENT propagation-constant offset;
the accumulated phase after length L is delta_beta0(p)*L, applied once
as a closed-form final multiply in MultimodeFiberPropagator rather than
inside the per-step split-step loop.
"""
import numpy as np
from dataclasses import dataclass, field

from core.dfb_laser import c
from fiber.materials import FiberMaterial, make_material

KAPPA_FLOOR = 1.0   # calibration constant for the alpha=2 "floor" delay term (see module docstring)


@dataclass
class MultimodeFiberGeometry:
    """Graded-index multimode fiber cross-section."""
    name: str = 'om3'
    core_radius: float = 25e-6      # m
    clad_radius: float = 62.5e-6    # m
    NA: float = 0.20
    n1: float = 1.4682              # core index (silica, ~850 nm)
    alpha_profile: float = 2.051    # graded-index profile exponent (alpha=2 -> parabolic)
    ofl_bandwidth_MHz_km: float = 1500.0   # nominal datasheet bandwidth, reference only
    dma_fraction: float = 0.4        # differential mode attenuation: highest-order mode group's
                                      # EXTRA loss as a fraction of the fundamental's loss
    dma_exponent: float = 2.0        # power-law shape of DMA growth with mode-group index


# Representative OM1-OM5 geometries. core_radius/NA are standard values
# (62.5/125 for OM1, 50/125 for OM2-OM5); alpha_profile is calibrated (see
# module docstring) so this module's own estimated_bandwidth() lands near
# each grade's nominal 850 nm datasheet EMB/OFL bandwidth; dma_fraction
# decreases with grade (better bend-insensitive, laser-optimized fibers
# have tighter mode confinement) -- treat all of these as representative,
# not manufacturing specs.
MULTIMODE_GEOMETRIES = {
    'om1': MultimodeFiberGeometry(name='om1', core_radius=31.25e-6, clad_radius=62.5e-6,
                                   NA=0.275, alpha_profile=2.301, ofl_bandwidth_MHz_km=200.0,
                                   dma_fraction=1.0),
    'om2': MultimodeFiberGeometry(name='om2', core_radius=25e-6, clad_radius=62.5e-6,
                                   NA=0.20, alpha_profile=2.241, ofl_bandwidth_MHz_km=500.0,
                                   dma_fraction=0.7),
    'om3': MultimodeFiberGeometry(name='om3', core_radius=25e-6, clad_radius=62.5e-6,
                                   NA=0.20, alpha_profile=2.051, ofl_bandwidth_MHz_km=1500.0,
                                   dma_fraction=0.4),
    'om4': MultimodeFiberGeometry(name='om4', core_radius=25e-6, clad_radius=62.5e-6,
                                   NA=0.20, alpha_profile=1.988, ofl_bandwidth_MHz_km=4700.0,
                                   dma_fraction=0.2),
    'om5': MultimodeFiberGeometry(name='om5', core_radius=25e-6, clad_radius=62.5e-6,
                                   NA=0.20, alpha_profile=1.988, ofl_bandwidth_MHz_km=4700.0,
                                   dma_fraction=0.2),
}


def make_multimode_geometry(name='om3', **overrides):
    """Create a MultimodeFiberGeometry from a named OM1-OM5 preset."""
    if name not in MULTIMODE_GEOMETRIES:
        raise ValueError(f"Unknown multimode geometry {name!r}. Choose from {list(MULTIMODE_GEOMETRIES)}.")
    base = MULTIMODE_GEOMETRIES[name]
    return MultimodeFiberGeometry(**{**base.__dict__, **overrides})


def _mode_group_count(omega, geo):
    """Continuous (non-integer) number of guided principal mode groups at
    angular frequency omega -- used both to set the actual (rounded)
    n_modes at the design frequency, and, evaluated off-design, to derive
    the mode-dependent part of beta2/beta3 by finite difference."""
    k0 = omega / c
    V = k0 * geo.core_radius * geo.NA
    alpha_p = geo.alpha_profile
    n_modes_scalar = (alpha_p / (alpha_p + 2)) * V ** 2 / 2
    return np.sqrt(2 * max(n_modes_scalar, 1.0))


def _tau_over_L(p, omega, geo):
    """Group delay per unit length (s/m), i.e. beta1, for mode group p at
    angular frequency omega (see module docstring)."""
    M_continuous = _mode_group_count(omega, geo)
    Delta = geo.NA ** 2 / (2 * geo.n1 ** 2)
    alpha_p = geo.alpha_profile
    x = p / M_continuous
    exp1 = 2 * alpha_p / (alpha_p + 2)
    exp2 = 4 * alpha_p / (alpha_p + 2)
    return (geo.n1 / c) * (Delta * (alpha_p - 2) / (alpha_p + 2) * x ** exp1
                            + Delta ** 2 * KAPPA_FLOOR * x ** exp2)


def _beta0(p, omega, geo):
    """Absolute propagation constant (rad/m) for mode group p at angular
    frequency omega, leading-order WKB alpha-profile relation (see module
    docstring "Absolute inter-mode phase"). d(_beta0)/d(omega) reproduces
    _tau_over_L's leading term exactly."""
    M_continuous = _mode_group_count(omega, geo)
    Delta = geo.NA ** 2 / (2 * geo.n1 ** 2)
    alpha_p = geo.alpha_profile
    x = p / M_continuous
    exp1 = 2 * alpha_p / (alpha_p + 2)
    return geo.n1 * (omega / c) * (1 - Delta * x ** exp1)


@dataclass
class MultimodeFiberParams:
    """Derived per-mode-group parameters for multimode GNLSE propagation.

    Modes are indexed 0..n_modes-1 (principal mode group p=1..n_modes);
    index 0 is the fundamental-like group and is used as the reference
    for every per-mode quantity below (delta_beta1[0]=0, alpha[0]=the
    scalar alpha_dB_km loss exactly, beta2[0]/beta3[0]=the scalar
    material values exactly) -- so a single-mode-only launch (mode 0)
    reproduces fiber.propagator.FiberPropagator exactly, and every other
    mode group carries an explicit, physically-motivated correction on
    top of that baseline.
    """
    lambda0: float = 850e-9
    material: FiberMaterial = field(default_factory=make_material)
    geometry: MultimodeFiberGeometry = field(default_factory=make_multimode_geometry)

    alpha_dB_km: float = 3.0        # fundamental-mode attenuation (dB/km); MMF at 850 nm is lossier
                                     # than SMF at 1550 nm; other mode groups get MORE loss on top
                                     # (see geometry.dma_fraction/dma_exponent)
    D: float = -120.0               # chromatic (material) dispersion (ps/nm/km); silica is normal-
                                     # dispersion below its ~1270 nm zero-dispersion wavelength, hence
                                     # negative D at 850 nm; shared by all mode groups
    beta3_material: float = 0.0     # third-order MATERIAL dispersion (s^3/m), shared by all mode groups
    mode_coupling_length: float = 3.0   # decay length (in mode-group index units) of the overlap model

    # Derived (SI units / dimensionless), filled in __post_init__
    omega0: float = field(init=False)
    k0: float = field(init=False)
    Delta: float = field(init=False)          # core-cladding relative index difference
    V: float = field(init=False)
    n_modes: int = field(init=False)          # M, number of principal mode groups resolved
    delta_beta0: np.ndarray = field(init=False)  # (M,) absolute inter-mode propagation-constant offset (rad/m)
    delta_beta1: np.ndarray = field(init=False)  # (M,) relative group delay per unit length (s/m)
    alpha: np.ndarray = field(init=False)     # (M,) per-mode loss (1/m); alpha[0] = the scalar alpha_dB_km loss
    beta2: np.ndarray = field(init=False)     # (M,) per-mode GVD (s^2/m) = material + waveguide correction
    beta3: np.ndarray = field(init=False)     # (M,) per-mode TOD (s^3/m) = material + waveguide correction
    A_eff: np.ndarray = field(init=False)     # (M,) effective area per mode group (m^2)
    gamma_self: np.ndarray = field(init=False)  # (M,) intramodal nonlinear coefficient (1/W/m)
    overlap: np.ndarray = field(init=False)   # (M,M) spatial overlap factor, diag=1
    gamma_matrix: np.ndarray = field(init=False)  # (M,M) nonlinear coupling gamma_pq (1/W/m)

    def __post_init__(self):
        geo = self.geometry
        self.omega0 = 2 * np.pi * c / self.lambda0
        self.k0 = 2 * np.pi / self.lambda0
        self.Delta = geo.NA ** 2 / (2 * geo.n1 ** 2)
        self.V = self.k0 * geo.core_radius * geo.NA

        M_continuous0 = _mode_group_count(self.omega0, geo)
        self.n_modes = max(int(round(M_continuous0)), 2)
        p = np.arange(1, self.n_modes + 1)
        x = p / self.n_modes

        tau_over_L = _tau_over_L(p, self.omega0, geo)
        self.delta_beta1 = tau_over_L - tau_over_L[0]

        beta0_p = _beta0(p, self.omega0, geo)
        self.delta_beta0 = beta0_p - beta0_p[0]

        # Per-mode loss (differential mode attenuation): power-law growth
        # with normalised mode index, referenced to mode 0 so alpha[0] is
        # exactly the scalar alpha_dB_km loss.
        alpha0 = self.alpha_dB_km / (10 * np.log10(np.e)) / 1e3
        dma_shape = x ** geo.dma_exponent
        self.alpha = alpha0 * (1 + geo.dma_fraction * (dma_shape - dma_shape[0]))

        # Per-mode waveguide dispersion: beta1(p, omega) = _tau_over_L(p, omega)
        # is a function of omega through the mode count M(omega); beta2 =
        # d(beta1)/d(omega), beta3 = d^2(beta1)/d(omega^2), evaluated by
        # finite difference and referenced to mode 0 (so beta2[0]/beta3[0]
        # equal the scalar material values exactly).
        domega = 0.01 * self.omega0
        tau_plus = _tau_over_L(p, self.omega0 + domega, geo)
        tau_minus = _tau_over_L(p, self.omega0 - domega, geo)
        beta2_wg = (tau_plus - tau_minus) / (2 * domega)
        beta3_wg = (tau_plus - 2 * tau_over_L + tau_minus) / domega ** 2

        beta2_material = -self.D * 1e-6 * self.lambda0 ** 2 / (2 * np.pi * c)
        self.beta2 = beta2_material + (beta2_wg - beta2_wg[0])
        self.beta3 = self.beta3_material + (beta3_wg - beta3_wg[0])

        # Effective area per mode group: higher-order modes spread further
        # from the core axis. For a parabolic-index fiber the paraxial wave
        # equation maps exactly onto the 2D quantum harmonic oscillator, whose
        # eigenmode size grows as sqrt(2*(mode number)+1) -- so mode AREA
        # (size^2) grows linearly with mode-group number p.
        A_core = np.pi * geo.core_radius ** 2
        A_eff_1 = A_core / self.n_modes
        self.A_eff = A_eff_1 * p

        self.gamma_self = 2 * np.pi * self.material.n2 / (self.lambda0 * self.A_eff)

        P, Q = np.meshgrid(p, p, indexing='ij')
        self.overlap = np.exp(-np.abs(P - Q) / self.mode_coupling_length)

        gamma_geom_mean = (2 * np.pi * self.material.n2 / self.lambda0
                            / np.sqrt(np.outer(self.A_eff, self.A_eff)))
        self.gamma_matrix = gamma_geom_mean * self.overlap

    def rms_delay_spread(self):
        """RMS intermodal (differential-mode) delay spread per unit length (s/m)."""
        return float(np.std(self.delta_beta1))

    def estimated_bandwidth_MHz_km(self):
        """This model's own 3-dB bandwidth-length product estimate (MHz*km),
        from the RMS delay spread via BW ~ 0.44/sigma. Compare against
        geometry.ofl_bandwidth_MHz_km (the nominal datasheet reference)."""
        sigma = self.rms_delay_spread()
        if sigma < 1e-30:
            return float('inf')
        return 0.44 / sigma * 1e-9   # Hz*m -> MHz*km


MULTIMODE_FIBER_PRESETS = ['om1', 'om2', 'om3', 'om4', 'om5']


def make_multimode_fiber(fiber_type='om3', material_overrides=None, geometry_overrides=None,
                          **overrides):
    """Create a MultimodeFiberParams from a named OM1-OM5 preset.

    Parameters
    ----------
    fiber_type : str -- 'om1', 'om2', 'om3', 'om4', 'om5'
    material_overrides, geometry_overrides : dict, optional
    **overrides
        Override any top-level MultimodeFiberParams field (lambda0,
        alpha_dB_km, D, beta3_material, mode_coupling_length).

    Examples
    --------
    >>> om4 = make_multimode_fiber('om4')
    >>> om3_1300 = make_multimode_fiber('om3', lambda0=1300e-9, D=0.0)
    """
    if fiber_type not in MULTIMODE_FIBER_PRESETS:
        raise ValueError(f"Unknown fiber_type {fiber_type!r}. Choose from {MULTIMODE_FIBER_PRESETS}.")
    material = make_material('silica', **(material_overrides or {}))
    geometry = make_multimode_geometry(fiber_type, **(geometry_overrides or {}))
    return MultimodeFiberParams(material=material, geometry=geometry, **overrides)
