"""Validation checks for QuantumMultimodePropagator (spontaneous Raman
noise, intramodal and intermodal, in multimode fiber).

Checks, in order:
  1. Mode-0 consistency: the intramodal noise formula for mode 0 (using
     gamma_matrix's diagonal) exactly matches QuantumRamanPropagator's
     single-mode formula for a fiber with matching gamma.
  2. Intermodal spontaneous noise from a CW/quasi-CW driving mode: unlike
     MultimodeFiberPropagator's DETERMINISTIC intermodal Raman term
     (which needs genuine time-varying power to transfer anything -- see
     that module's docstring), spontaneous noise here is driven by local
     PEAK power directly, so a quasi-CW pulse in mode 0 with NOTHING
     launched elsewhere still seeds measurable noise in other, initially
     empty mode groups -- with locality (nearby mode groups get far more
     than distant ones) matching the overlap-decay weighting already
     used for the classical coupling.
  3. Stokes/anti-Stokes asymmetry at T~0: the same physics already
     validated for the single-mode and WDM cases, now for the
     intramodal band of a multimode fiber's mode 0.
"""
import numpy as np

from fiber.multimode_fiber import make_multimode_fiber
from fiber.quantum_multimode import QuantumMultimodePropagator
from fiber.quantum_noise import QuantumRamanPropagator
from fiber.fiber_params import FiberParams
from fiber.materials import make_material
from fiber.analysis import band_power

print("=" * 70)
print("  QuantumMultimodePropagator validation")
print("=" * 70)

# ── 1. Mode-0 consistency ───────────────────────────────────────────

print("\n1. Mode-0 intramodal noise vs single-mode QuantumRamanPropagator:")

fiber_mm = make_multimode_fiber('om3', lambda0=1550e-9, D=17.0, alpha_dB_km=0.2)
N = 2 ** 10
dt = 20e-15
Omega = 2 * np.pi * np.fft.fftfreq(N, d=dt)

qmm = QuantumMultimodePropagator(fiber_mm, seed=0)
qmm._prepare(Omega, dt)

sm_fiber = FiberParams(material=make_material('silica'), alpha_dB_km=0.2, D=17.0, beta3=0.0)
sm_fiber.gamma = fiber_mm.gamma_self[0]
qsm = QuantumRamanPropagator(sm_fiber, seed=0)
qsm._prepare(Omega, dt)

mode0_gain = qmm._weighted_shape * fiber_mm.gamma_matrix[0, 0]
gain_match = np.allclose(mode0_gain, qsm._noise_gain, rtol=1e-6)
omega_match = np.allclose(qmm._omega_abs, qsm._omega_abs)
print(f"   mode-0 noise_gain matches single-mode QuantumRamanPropagator: {gain_match}")
print(f"   omega_abs matches: {omega_match}")
assert gain_match and omega_match, \
    "Mode 0's intramodal spontaneous-Raman noise formula should exactly match the single-mode case"
print("   PASS")

# ── 2. Intermodal spontaneous noise from a CW/quasi-CW driving mode ──

print("\n2. Intermodal spontaneous noise from a quasi-CW driving mode (T~0):")

fiber_cold = make_multimode_fiber('om3', lambda0=1550e-9, D=17.0, alpha_dB_km=0.0,
                                   material_overrides=dict(T=1e-3))
N2 = 2 ** 10
dt2 = 20e-15
t2 = (np.arange(N2) - N2 // 2) * dt2
T0 = 2e-12
P_pump = 0.05
pump = (np.sqrt(P_pump) / np.cosh(t2 / T0)).astype(complex)  # quasi-CW-like

M = fiber_cold.n_modes
A0 = np.zeros((M, N2), dtype=complex)
A0[0] = pump  # ONLY mode 0 populated

L_test = 500.0
n_runs = 15


def run(fiber, A0, seed):
    prop = QuantumMultimodePropagator(fiber, seed=seed)
    return prop.propagate(A0, dt2, L_test, step_size=20.0)


outs = [run(fiber_cold, A0, s) for s in range(n_runs)]
energy_adjacent = np.mean([np.sum(np.abs(o[1]) ** 2) for o in outs])
energy_mid = np.mean([np.sum(np.abs(o[min(5, M - 1)]) ** 2) for o in outs])
energy_far = np.mean([np.sum(np.abs(o[-1]) ** 2) for o in outs])

print(f"   spontaneous noise energy: mode 1 (adjacent, empty) = {energy_adjacent:.3e}")
print(f"   spontaneous noise energy: mid mode                 = {energy_mid:.3e}")
print(f"   spontaneous noise energy: farthest mode             = {energy_far:.3e}")
print(f"   adjacent/farthest ratio (locality): {energy_adjacent/energy_far:.1f}")

assert energy_adjacent > 0, \
    "A quasi-CW driving mode should seed measurable spontaneous noise in an initially-empty adjacent mode"
assert energy_adjacent > 10 * energy_far, \
    "Nearby mode groups should pick up far more spontaneous noise than distant ones (overlap-decay locality)"
print("   PASS (spontaneous noise leaks into other modes even from a CW driver, with correct locality)")

# ── 3. Stokes/anti-Stokes asymmetry at T~0 ──────────────────────────

print("\n3. Stokes/anti-Stokes asymmetry at T~0 (mode 0's own intramodal band):")

T0_cw = 2e-12
P_pump_cw = 0.5
A0_cw = (np.sqrt(P_pump_cw) / np.cosh(t2 / T0_cw)).astype(complex)
A0_asym = np.zeros((M, N2), dtype=complex)
A0_asym[0] = A0_cw

Omega_grid = 2 * np.pi * np.fft.fftfreq(N2, d=dt2)
Omega_max = Omega_grid.max()
gain_band = (0.3 * Omega_max, 0.9 * Omega_max)
loss_band = (-0.9 * Omega_max, -0.3 * Omega_max)


def band_energy(fiber, A0, band, n_runs):
    vals = []
    for seed in range(n_runs):
        prop = QuantumMultimodePropagator(fiber, seed=seed)
        A_out = prop.propagate(A0, dt2, L_test, step_size=20.0)
        vals.append(band_power(A_out[0], dt2, *band))
    return float(np.mean(vals))


gain_e = band_energy(fiber_cold, A0_asym, gain_band, n_runs)
loss_e = band_energy(fiber_cold, A0_asym, loss_band, n_runs)
print(f"   T~0 gain-side band power: {gain_e:.3e}")
print(f"   T~0 loss-side band power: {loss_e:.3e}")
assert gain_e > 0, "Spontaneous noise should persist on the gain side even at T->0"
assert loss_e < 0.2 * gain_e, "Loss-side noise should be strongly suppressed relative to gain-side at T->0"
print("   PASS")

print("\n" + "=" * 70)
print("  All QuantumMultimodePropagator checks passed.")
print("=" * 70)
