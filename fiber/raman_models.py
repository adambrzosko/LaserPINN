"""
Silica Raman response models in the frequency domain.

Kernel convention: H(Omega) = integral h(t) exp(-i Omega t) dt, the same as
fiber.raman_response, where the physical optical frequency is omega0 - Omega, so
Omega > 0 is the Stokes side.

    BlowWood    single damped oscillator (Blow & Wood, IEEE JQE 25, 2665, 1989).
    LinAgrawal  adds the boson-peak term (Lin & Agrawal, Opt. Lett. 31, 3086, 2006).
                Much more accurate below ~5 THz, which is where WDM channel
                spacings of a few hundred GHz sit; the single oscillator
                underestimates the gain there by ~1.4x at 400 GHz and ~2x at 3 THz.
    HollenbeckCantrell
                intermediate-broadening model: 13 silica vibrational modes, each a
                Gaussian-Lorentzian (Voigt) line (Hollenbeck & Cantrell, JOSA B 19,
                2886, 2002). The model Woodward's thesis uses (Imperial, 2015, eq. 2.3.7);
                fits the measured gain spectrum across the whole band, including the
                shoulders the single oscillator smooths over.

    from fiber.raman_models import LinAgrawal, gain_shape, spontaneous_shape
"""
from dataclasses import dataclass

import numpy as np
from scipy.special import wofz

from fiber.constants import c, hbar, kB


def phonon_occupation(Omega, T):
    """Bose-Einstein occupation at frequency offset Omega (rad/s), temperature T (K)."""
    x = hbar * np.abs(np.asarray(Omega, dtype=float)) / (kB * T)
    n = np.zeros_like(x)
    ok = (x > 1e-12) & (x < 700)
    n[ok] = 1.0 / np.expm1(x[ok])
    return n


def _damped_oscillator(Omega, tau1, tau2):
    a, b = 1.0 / tau2, 1.0 / tau1
    return (a * a + b * b) / ((a * a + b * b - Omega ** 2) + 2j * a * Omega)


@dataclass(frozen=True)
class BlowWood:
    f_R: float = 0.18
    tau1: float = 12.2e-15
    tau2: float = 32e-15
    name: str = 'blow_wood'

    def H(self, Omega):
        return _damped_oscillator(np.asarray(Omega, dtype=float), self.tau1, self.tau2)


@dataclass(frozen=True)
class LinAgrawal:
    f_R: float = 0.245
    f_b: float = 0.21
    tau1: float = 12.2e-15
    tau2: float = 32e-15
    tau_b: float = 96e-15
    name: str = 'lin_agrawal'

    def H(self, Omega):
        Omega = np.asarray(Omega, dtype=float)
        H_a = _damped_oscillator(Omega, self.tau1, self.tau2)
        s = 1.0 + 1j * Omega * self.tau_b
        H_b = (2.0 * s - 1.0) / s ** 2
        return (1.0 - self.f_b) * H_a + self.f_b * H_b


# Hollenbeck & Cantrell (2002), Table 1: component position, amplitude, Gaussian FWHM
# and Lorentzian FWHM of each vibrational mode (positions and widths in cm^-1).
_HC_POSITION = (56.25, 100.0, 231.25, 362.5, 463.0, 497.0, 611.5, 691.67, 793.67,
                835.5, 930.0, 1080.0, 1215.0)
_HC_AMPLITUDE = (1.0, 11.40, 36.67, 67.67, 74.0, 4.5, 6.8, 4.6, 4.2, 4.5, 2.7, 3.1, 3.0)
_HC_GAUSS_FWHM = (52.10, 110.42, 175.00, 162.50, 135.33, 24.5, 41.5, 155.00, 59.5, 64.3,
                  150.0, 91.0, 160.0)
_HC_LORENTZ_FWHM = (17.37, 38.81, 58.33, 54.17, 45.11, 8.17, 13.83, 51.67, 19.83, 21.43,
                    50.00, 30.33, 53.33)


@dataclass(frozen=True)
class HollenbeckCantrell:
    """Intermediate-broadening Raman response,

        h_R(t) = sum_i A_i exp(-gamma_i t) exp(-Gamma_i^2 t^2 / 4) sin(omega_i t),  t >= 0

    normalised to integral(h_R dt) = 1, with omega_i = 2 pi c x (position), gamma_i =
    pi c x (Lorentzian FWHM) and Gamma_i = pi c x (Gaussian FWHM), x converting cm^-1.
    The paper writes the prefactor as A'_i / omega_i with A'_i = A_i omega_i, so the
    tabulated amplitudes enter directly; the other reading (A_i / omega_i) moves the gain
    peak to 12.7 THz instead of the measured 13.2 THz.

    The transform is closed form: each term is a one-sided Gaussian-times-exponential
    integral, (sqrt(pi)/Gamma) erfcx(p/Gamma) with p = gamma + i(Omega -+ omega_i), and
    erfcx(z) = w(iz) is the Faddeeva function, so no time grid is sampled.

    f_R = 0.18 follows Woodward's thesis (after Stolen et al. 1989). It is not fitted to
    this line shape: the implied peak g_R at 1550 nm is 4.7e-14 m/W, against 5.8e-14 for
    LinAgrawal and 5.1e-14 for BlowWood. f_R = 0.22 matches LinAgrawal's peak, which
    isolates the difference in spectral shape when comparing the two.
    """
    f_R: float = 0.18
    name: str = 'hollenbeck_cantrell'

    def _raw(self, Omega):
        x = 100.0 * c   # cm^-1 -> Hz
        wv = 2 * np.pi * x * np.asarray(_HC_POSITION)
        gam = np.pi * x * np.asarray(_HC_LORENTZ_FWHM)
        Gam = np.pi * x * np.asarray(_HC_GAUSS_FWHM)
        amp = np.asarray(_HC_AMPLITUDE)
        W = np.asarray(Omega, dtype=float)[..., None]

        def one_sided(p):   # integral_0^inf exp(-Gam^2 t^2 / 4 - p t) dt
            return np.sqrt(np.pi) / Gam * wofz(1j * p / Gam)

        terms = (one_sided(gam + 1j * (W - wv)) - one_sided(gam + 1j * (W + wv))) / 2j
        return np.sum(amp * terms, axis=-1)

    def H(self, Omega):
        return self._raw(Omega) / self._raw(0.0).real


def gain_shape(model, Omega):
    """-2 f_R Im H(Omega). Times (n2*omega_signal/c)*S_qlql gives the Raman power
    gain per watt of pump per metre."""
    return -2.0 * model.f_R * np.imag(model.H(Omega))


def spontaneous_shape(model, Omega, T):
    """Fluctuation-dissipation weight of gain_shape: (n_th + 1) on the Stokes side,
    n_th on the anti-Stokes side (vanishes as T -> 0)."""
    g = gain_shape(model, Omega)
    n = phonon_occupation(Omega, T)
    return np.where(g > 0, g * (n + 1.0), -g * n)


def peak_gain_coefficient(model, n2, wavelength):
    """Peak bulk Raman gain coefficient g_R (m/W) implied by the model, for comparison
    with measured values (~6e-14 m/W for co-polarised silica at 1550 nm)."""
    Omega = 2 * np.pi * np.linspace(1e11, 3e13, 30000)
    omega0 = 2 * np.pi * c / wavelength
    return float(np.max(n2 * (omega0 - Omega) / c * gain_shape(model, Omega)))
