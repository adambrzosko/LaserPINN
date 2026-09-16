"""
Fiber material models: nonlinear index and Raman response.

    from fiber.materials import FiberMaterial, make_material
"""
import numpy as np
from dataclasses import dataclass

from fiber.constants import hbar, kB


@dataclass
class FiberMaterial:
    """Optical material for nonlinear fiber propagation.

    The Raman response uses the damped-oscillator (Blow & Wood 1989) model:

        h_R(t) = (tau1^2+tau2^2)/(tau1*tau2^2) * exp(-t/tau2) * sin(t/tau1),  t>=0

    normalised so that integral(h_R dt) = 1. For silica, tau1=12.2 fs and
    tau2=32 fs reproduce the ~13.2 THz Raman gain peak (Agrawal, Nonlinear
    Fiber Optics). f_R is the fractional Raman contribution to the total
    third-order nonlinear response (~0.18 for silica; the remainder is the
    (near-)instantaneous electronic Kerr response).
    """
    name: str = 'silica'
    n2: float = 2.6e-20        # nonlinear refractive index (m^2/W)
    f_R: float = 0.18          # fractional Raman contribution to n2
    tau1: float = 12.2e-15     # Raman oscillation period (s)
    tau2: float = 32e-15       # Raman damping time (s)
    T: float = 300.0           # temperature (K), sets phonon occupation
    n_g: float = 1.4682        # group index (dimensionless) at the reference wavelength; sets the
                                # ABSOLUTE reference propagation constant beta1_ref=n_g/c used for
                                # inter-channel/inter-mode coherent phase tracking (see
                                # fiber.wdm_propagator / fiber.multimode_fiber delta_beta0)

    # Brillouin scattering (acoustic phonons -- distinct from the Raman
    # optical-phonon response above): peak gain coefficient, frequency
    # shift, and gain linewidth. Representative silica values at ~1550 nm;
    # nu_B scales roughly as 1/lambda0, so rescale for other wavelengths.
    g_B: float = 5e-11         # peak Brillouin gain coefficient (m/W)
    nu_B: float = 10.8e9       # Brillouin frequency shift (Hz)
    delta_nu_B: float = 30e6   # Brillouin gain linewidth, FWHM (Hz)

    def brillouin_gain(self, delta_nu):
        """Lorentzian Brillouin gain spectrum g_B(delta_nu) [m/W].

        delta_nu is the frequency offset (Hz) of the pump-Stokes
        separation from the exact resonance (delta_nu = 0 at
        separation = nu_B, i.e. delta_nu = separation - nu_B).
        """
        half_width = self.delta_nu_B / 2.0
        return self.g_B / (1.0 + (np.asarray(delta_nu) / half_width) ** 2)

    def acoustic_phonon_lifetime(self):
        """Acoustic phonon (Brillouin) lifetime tau_B = 1/(pi*delta_nu_B) (s)."""
        return 1.0 / (np.pi * self.delta_nu_B)

    def raman_response(self, t):
        """Time-domain Raman response h_R(t), causal (h_R(t<0) = 0)."""
        t = np.asarray(t, dtype=float)
        h = np.zeros_like(t)
        mask = t >= 0
        tm = t[mask]
        h[mask] = ((self.tau1 ** 2 + self.tau2 ** 2) / (self.tau1 * self.tau2 ** 2)
                   * np.exp(-tm / self.tau2) * np.sin(tm / self.tau1))
        return h

    def phonon_occupation(self, Omega):
        """Bose-Einstein phonon occupation number n_th at frequency offset
        Omega (rad/s) and the material temperature T.

        n_th -> 0 as T -> 0 or |Omega| -> infinity (no thermal phonons to
        supply anti-Stokes scattering); n_th -> kT/(hbar*|Omega|) at high T.
        The Omega=0 bin is set to 0 as a safe placeholder -- it is always
        multiplied by a vanishing Raman gain there, so its value is moot.
        """
        Omega = np.asarray(Omega, dtype=float)
        x = hbar * np.abs(Omega) / (kB * self.T)
        n = np.zeros_like(x)
        nz = (x > 1e-12) & (x < 700)   # expm1 overflows above ~700; n_th -> 0 there anyway
        n[nz] = 1.0 / np.expm1(x[nz])
        return n


MATERIALS = {
    'silica': FiberMaterial(
        name='silica', n2=2.6e-20, f_R=0.18, tau1=12.2e-15, tau2=32e-15,
        g_B=5e-11, nu_B=10.8e9, delta_nu_B=30e6, n_g=1.4682),
    # Brillouin parameters for the non-silica presets below are rough,
    # order-of-magnitude placeholders (chalcogenides generally have a
    # LOWER acoustic velocity, giving smaller nu_B, and often a LARGER
    # g_B from bigger photoelastic coefficients) -- calibrate against a
    # measurement for quantitative work. n_g values are similarly
    # representative, not measured.
    'chalcogenide_as2s3': FiberMaterial(
        name='chalcogenide_as2s3', n2=3.0e-18, f_R=0.10, tau1=15e-15, tau2=230e-15,
        g_B=3e-10, nu_B=7.5e9, delta_nu_B=50e6, n_g=2.45),
    'silicon': FiberMaterial(
        name='silicon', n2=4.5e-18, f_R=0.0, tau1=12.2e-15, tau2=32e-15,
        g_B=1e-9, nu_B=16e9, delta_nu_B=10e6, n_g=4.2),
}


def make_material(name='silica', **overrides):
    """Create a FiberMaterial from a named preset, with optional overrides.

    Presets: 'silica' (SMF/HNLF/PCF host glass), 'chalcogenide_as2s3'
    (high-index-contrast nonlinear waveguides), 'silicon' (f_R=0: no
    first-order Raman response in the envelope sense used here -- silicon's
    Raman response is a narrow-band optical-phonon line better modelled as
    a separate resonance if needed; treated as Kerr-only by default).

    Examples
    --------
    >>> silica = make_material('silica')
    >>> hot_silica = make_material('silica', T=350.0)
    """
    if name not in MATERIALS:
        raise ValueError(f"Unknown material {name!r}. Choose from {list(MATERIALS)}.")
    base = MATERIALS[name]
    return FiberMaterial(**{**base.__dict__, **overrides})
