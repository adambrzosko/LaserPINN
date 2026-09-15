"""
Fiber parameter presets: combines geometry + material + published
dispersion/loss numbers into the derived quantities the propagator needs
(alpha, beta2, beta3, gamma, omega0).

    from fiber.fiber_params import FiberParams, make_fiber
"""
import numpy as np
from dataclasses import dataclass, field

from fiber.constants import c
from fiber.materials import FiberMaterial, make_material
from fiber.geometry import FiberGeometry, make_geometry


@dataclass
class FiberParams:
    """Derived fiber parameters for GNLSE propagation.

    beta3 is taken as a direct physical constant (s^3/m) rather than
    derived from a dispersion slope S, since D-to-S conversions are
    fiber-specific and error-prone; supply beta3 straight from a datasheet
    or measurement, as the original fiber_propagation.py study did.
    """
    lambda0: float = 1550e-9          # carrier wavelength (m)
    material: FiberMaterial = field(default_factory=make_material)
    geometry: FiberGeometry = field(default_factory=make_geometry)

    alpha_dB_km: float = 0.2          # attenuation (dB/km)
    D: float = 17.0                   # dispersion (ps/nm/km) -> beta2
    beta3: float = 0.07e-39           # third-order dispersion (s^3/m)

    # Derived (SI), filled in __post_init__
    alpha: float = field(init=False)   # power loss coefficient (1/m)
    beta2: float = field(init=False)   # GVD (s^2/m)
    gamma: float = field(init=False)   # nonlinear coefficient (1/W/m)
    omega0: float = field(init=False)  # carrier angular frequency (rad/s)
    beta1_ref: float = field(init=False)  # absolute reference group delay per length (s/m) = n_g/c;
                                           # used for inter-channel ABSOLUTE phase tracking, see
                                           # fiber.wdm_propagator.WDMPropagator's delta_beta0

    def __post_init__(self):
        self.alpha = self.alpha_dB_km / (10 * np.log10(np.e)) / 1e3
        self.beta2 = -self.D * 1e-6 * self.lambda0 ** 2 / (2 * np.pi * c)
        self.gamma = 2 * np.pi * self.material.n2 / (self.lambda0 * self.geometry.A_eff)
        self.omega0 = 2 * np.pi * c / self.lambda0
        self.beta1_ref = self.material.n_g / c


# Approximate, order-of-magnitude presets -- adjust D/alpha_dB_km/beta3/A_eff
# to match your fiber's actual datasheet before drawing quantitative
# conclusions.
FIBER_PRESETS = {
    'smf28': dict(
        material='silica', geometry='smf28',
        alpha_dB_km=0.2, D=17.0, beta3=0.07e-39),
    'dcf': dict(
        material='silica', geometry='dcf',
        alpha_dB_km=0.5, D=-100.0, beta3=0.3e-39),
    'hnlf': dict(
        material='silica', geometry='hnlf',
        alpha_dB_km=0.8, D=0.5, beta3=0.03e-39),
    'pcf_supercontinuum': dict(
        material='silica', geometry='pcf_smallcore',
        alpha_dB_km=5.0, D=2.0, beta3=0.02e-39),
    'chalcogenide_waveguide': dict(
        material='chalcogenide_as2s3', geometry='chalc_ridge',
        alpha_dB_km=2000.0, D=-50.0, beta3=0.1e-39),
}


def make_fiber(fiber_type='smf28', material_overrides=None, geometry_overrides=None,
               **overrides):
    """Create a FiberParams from a named preset, with optional overrides.

    Parameters
    ----------
    fiber_type : str
        'smf28', 'dcf', 'hnlf', 'pcf_supercontinuum', 'chalcogenide_waveguide'
    material_overrides, geometry_overrides : dict, optional
        Overrides passed through to make_material()/make_geometry().
    **overrides
        Override any top-level FiberParams field (lambda0, alpha_dB_km, D, beta3).

    Examples
    --------
    >>> smf = make_fiber('smf28')
    >>> hnlf = make_fiber('hnlf', alpha_dB_km=0.6)
    >>> hot_smf = make_fiber('smf28', material_overrides=dict(T=350.0))
    """
    if fiber_type not in FIBER_PRESETS:
        raise ValueError(f"Unknown fiber_type {fiber_type!r}. Choose from {list(FIBER_PRESETS)}.")
    preset = dict(FIBER_PRESETS[fiber_type])
    mat_name = preset.pop('material')
    geo_name = preset.pop('geometry')
    material = make_material(mat_name, **(material_overrides or {}))
    geometry = make_geometry(geo_name, **(geometry_overrides or {}))
    preset.update(overrides)
    return FiberParams(material=material, geometry=geometry, **preset)
