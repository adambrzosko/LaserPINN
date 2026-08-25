"""
Adapters that convert core laser-simulation outputs into launch fields for
fiber.propagator.FiberPropagator (and its quantum_noise subclass).

    from fiber.sources import intracavity_to_field, extract_pulse, zero_pad
"""
import numpy as np

from core.dfb_laser import h


def intracavity_to_field(Er, Ei, laser, eta_i=0.8):
    """Convert DFB/FP intracavity photon-density field (Er, Ei), as produced
    by core.million_pulse_comparison / studies.fiber_propagation's Numba
    solvers, into an output field envelope A(t) with |A|^2 in Watts,
    referenced to the front facet.

    The intracavity field has units ~ sqrt(photon density); output power
    P = eta_i * frac_front * h*nu0 * V * S / tau_p, with S = Er^2+Ei^2.
    The phase structure of the complex field is preserved -- only the
    magnitude is rescaled.
    """
    frac_front = (1 - laser.R1) / ((1 - laser.R1) + (1 - laser.R2))
    scale = np.sqrt(eta_i * frac_front * h * laser.nu0 * laser.V / laser.tau_p)
    return scale * (Er + 1j * Ei)


def extract_pulse(A, pts_period, pulse_idx):
    """Slice a single pulse out of a multi-pulse waveform of period pts_period."""
    start = pulse_idx * pts_period
    end = (pulse_idx + 1) * pts_period
    return A[start:end]


def zero_pad(A, pad_factor=4):
    """Zero-pad a pulse envelope (centered) for finer spectral resolution
    before FFT-based propagation/analysis."""
    n = len(A)
    n_padded = n * pad_factor
    out = np.zeros(n_padded, dtype=complex)
    offset = (n_padded - n) // 2
    out[offset:offset + n] = A
    return out
