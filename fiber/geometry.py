"""
Fiber geometry models: core/cladding structure -> effective mode area.

    from fiber.geometry import FiberGeometry, make_geometry
"""
from dataclasses import dataclass


@dataclass
class FiberGeometry:
    """Waveguide cross-section, reduced to what the propagator needs.

    A_eff (effective mode area) is what actually drives the nonlinear
    coefficient gamma = 2*pi*n2/(lambda*A_eff); core_radius/NA are kept for
    reference and for approximate mode-field estimates, but A_eff itself is
    the authoritative field -- for real fibers, prefer the datasheet or
    mode-solver value over deriving it from core_radius/NA.
    """
    profile: str = 'step_index'      # 'step_index' | 'pcf' | 'planar'
    core_radius: float = 4.1e-6      # m
    NA: float = 0.14                 # numerical aperture
    A_eff: float = 80e-12            # effective mode area (m^2)

    def mode_field_diameter(self):
        """Rough step-index MFD estimate (Marcuse formula region); prefer
        A_eff directly for anything but order-of-magnitude sanity checks."""
        return 2.0 * self.core_radius * 1.1


GEOMETRIES = {
    'smf28':         FiberGeometry(profile='step_index', core_radius=4.1e-6, NA=0.14, A_eff=80e-12),
    'dcf':           FiberGeometry(profile='step_index', core_radius=2.3e-6, NA=0.20, A_eff=22e-12),
    'hnlf':          FiberGeometry(profile='step_index', core_radius=1.6e-6, NA=0.28, A_eff=11e-12),
    'pcf_smallcore': FiberGeometry(profile='pcf', core_radius=1.0e-6, NA=0.40, A_eff=3e-12),
    'chalc_ridge':   FiberGeometry(profile='planar', core_radius=0.5e-6, NA=0.80, A_eff=1.5e-12),
}


def make_geometry(name='smf28', **overrides):
    """Create a FiberGeometry from a named preset, with optional overrides.

    Presets: 'smf28', 'dcf' (dispersion-compensating fiber), 'hnlf'
    (highly nonlinear fiber), 'pcf_smallcore' (small-core photonic-crystal
    fiber, e.g. for supercontinuum), 'chalc_ridge' (chalcogenide ridge
    waveguide).
    """
    if name not in GEOMETRIES:
        raise ValueError(f"Unknown geometry {name!r}. Choose from {list(GEOMETRIES)}.")
    base = GEOMETRIES[name]
    return FiberGeometry(**{**base.__dict__, **overrides})
