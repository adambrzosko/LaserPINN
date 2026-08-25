"""
Raman response in the frequency domain, and the associated classical
stimulated-Raman gain spectrum.

    from fiber.raman_response import raman_response_freq_analytic, raman_gain_spectrum
"""
import numpy as np


def raman_response_freq_analytic(material, Omega):
    """Analytic Fourier transform of the Blow-Wood h_R(t) response.

    For h_R(t) = A*exp(-a t)*sin(b t)*theta(t) with a=1/tau2, b=1/tau1,
    A=(a^2+b^2)/b (the normalisation that gives integral(h_R dt)=1):

        H_R(Omega) = (a^2+b^2) / [(a^2+b^2-Omega^2) + i*2*a*Omega]

    H_R(0) = 1 (matches the time-domain normalisation); no poles for real
    Omega since a,b>0, so this is safe on an FFT frequency grid without
    needing to sample h_R(t) directly.
    """
    a = 1.0 / material.tau2
    b = 1.0 / material.tau1
    denom = (a ** 2 + b ** 2 - Omega ** 2) + 1j * 2 * a * Omega
    return (a ** 2 + b ** 2) / denom


def raman_gain_spectrum(material, Omega, gamma):
    """Classical stimulated-Raman gain coefficient g_R(Omega) [1/W/m].

    g_R(Omega) = -2*gamma*f_R*Im{H_R(Omega)}. The minus sign (relative to
    the bare Im{H_R(Omega)} one might naively expect) comes from linearizing
    fiber.propagator's actual pump/probe nonlinear-phase update dA/dz =
    i*gamma*A*conv(R,|A|^2) around a strong CW pump: a weak probe at offset
    Omega grows as d|probe|^2/dz = g_R(Omega)*P_pump*|probe|^2. This was
    verified directly (not just derived) by propagating a pump+probe field
    through FiberPropagator and confirming the probe at Omega>0 gains while
    Omega<0 loses, matching this formula's sign.

    Given this codebase's envelope convention (E ~ Re[A(t)*exp(-i*omega0*t)]
    with A's own spectral content indexed via the standard FFT convention),
    the physical optical frequency at offset Omega is omega0 - Omega -- so
    Omega>0 is the lower-frequency (Stokes) side, where g_R(Omega)>0 (gain);
    Omega<0 is the higher-frequency (anti-Stokes) side, where g_R(Omega)<0
    (loss). This matches the textbook picture: a probe placed below the
    pump by the Raman shift experiences gain.
    """
    H = raman_response_freq_analytic(material, Omega)
    return -2.0 * gamma * material.f_R * np.imag(H)
