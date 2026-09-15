"""Absolute calibration of the single-mode QuantumRamanPropagator noise (regression test
for the missing sqrt(N) in its frequency-domain Langevin amplitudes, which made the
noise N times too weak).

    python tests/test_noise_calibration.py
"""
import numpy as np

from fiber.fiber_params import make_fiber
from fiber.quantum_noise import QuantumRamanPropagator, hbar
from fiber.raman_response import raman_gain_spectrum


def check_quantum_raman_propagator_psd():
    """CW pump, equal loss for pump and noise: forward PSD = hbar omega g P0 exp(-alpha L) L."""
    fiber = make_fiber('smf28', alpha_dB_km=0.2)
    N, dt = 2 ** 11, 20e-15
    P0, L, runs = 1e-3, 5000.0, 8
    A0 = np.full(N, np.sqrt(P0), dtype=complex)
    Omega = 2 * np.pi * np.fft.fftfreq(N, d=dt)
    psd = np.zeros(N)
    for seed in range(runs):
        out = QuantumRamanPropagator(fiber, seed=seed).propagate(A0, dt, L, step_size=50.0)
        psd += np.abs(np.fft.fft(out)) ** 2 * dt / N / runs
    g = raman_gain_spectrum(fiber.material, Omega, fiber.gamma)
    n_th = fiber.material.phonon_occupation(Omega)
    shape = np.where(g > 0, g * (n_th + 1), -g * n_th)
    analytic = hbar * (fiber.omega0 - Omega) * shape * P0 * np.exp(-fiber.alpha * L) * L
    band = (Omega > 2 * np.pi * 8e12) & (Omega < 2 * np.pi * 18e12)
    ratio = psd[band].mean() / analytic[band].mean()
    assert abs(ratio - 1) < 0.06, ratio
    print(f'QuantumRamanPropagator absolute calibration OK: ensemble PSD / analytic = {ratio:.3f}')


if __name__ == '__main__':
    check_quantum_raman_propagator_psd()
    print('\nNoise calibration checks passed.')
