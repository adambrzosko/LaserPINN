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

    from fiber.raman_models import LinAgrawal, gain_shape, spontaneous_shape
"""
from dataclasses import dataclass

import numpy as np

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
