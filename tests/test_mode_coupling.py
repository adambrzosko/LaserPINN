"""Validation checks for RandomModeCouplingPropagator (linear random mode
coupling in multimode fiber).

Checks, in order:
  1. Exact power conservation: with no loss/nonlinearity, the total
     energy summed across ALL mode groups is conserved to floating-point
     precision -- the coupling operator is constructed as exp(i*Hermitian),
     which is exactly unitary regardless of coupling strength, so this
     should hold far tighter than any physics-approximation tolerance.
  2. On/off: kappa=0 gives EXACTLY zero coupling (a pulse launched purely
     into mode 0 stays purely in mode 0); kappa>0 measurably couples
     power into other mode groups.
  3. Growth with distance: the coupled-away fraction from mode 0
     increases monotonically with propagation length.
  4. Locality: mode groups closer in index (which the overlap-decay
     model treats as more strongly coupled, the same model already used
     for nonlinear coupling) pick up far more power than distant ones.
"""
import numpy as np

from fiber.multimode_fiber import make_multimode_fiber
from fiber.mode_coupling import RandomModeCouplingPropagator

print("=" * 70)
print("  Random mode coupling validation")
print("=" * 70)

fiber = make_multimode_fiber('om3', material_overrides=dict(n2=0.0, f_R=0.0), alpha_dB_km=0.0)
M = fiber.n_modes
N = 2 ** 9
dt = 5e-12
T0 = 15e-12
t = (np.arange(N) - N // 2) * dt
pulse = np.exp(-t ** 2 / (2 * T0 ** 2)).astype(complex)

A0 = np.zeros((M, N), dtype=complex)
A0[0] = pulse
energy_in = np.sum(np.abs(A0) ** 2) * dt

L = 100.0

# ── 1. Exact power conservation ─────────────────────────────────────

print("\n1. Exact power conservation (no loss/nonlinearity):")

prop = RandomModeCouplingPropagator(fiber, kappa=0.05, seed=0)
A_out = prop.propagate(A0, dt, L, step_size=5.0)
energy_out = np.sum(np.abs(A_out) ** 2) * dt
rel_diff = abs(energy_out - energy_in) / energy_in
print(f"   energy in={energy_in:.6e}, out={energy_out:.6e}, relative difference={rel_diff:.2e}")
assert rel_diff < 1e-10, "Random mode coupling should conserve total energy across mode groups exactly"
print("   PASS")

# ── 2. On/off ────────────────────────────────────────────────────────

print("\n2. On/off (kappa=0 vs kappa>0):")

prop0 = RandomModeCouplingPropagator(fiber, kappa=0.0, seed=0)
A_out0 = prop0.propagate(A0, dt, L, step_size=5.0)
power_other_0 = np.sum(np.abs(A_out0[1:]) ** 2)
power_other = np.sum(np.abs(A_out[1:]) ** 2)
print(f"   power in other modes: kappa=0 -> {power_other_0:.3e}, kappa=0.05 -> {power_other:.3e}")
assert power_other_0 == 0.0, "kappa=0 should give exactly zero mode coupling"
assert power_other > 0, "kappa>0 should measurably couple power out of mode 0"
print("   PASS")

# ── 3. Growth with distance ──────────────────────────────────────────

print("\n3. Coupled-away fraction growth with distance:")

lengths = [10, 50, 200, 500]
fractions = []
for Ltest in lengths:
    p = RandomModeCouplingPropagator(fiber, kappa=0.05, seed=1)
    A_out_L = p.propagate(A0, dt, Ltest, step_size=5.0)
    frac = np.sum(np.abs(A_out_L[1:]) ** 2) / np.sum(np.abs(A_out_L) ** 2)
    fractions.append(frac)
    print(f"   L={Ltest:4d} m: coupled-away fraction = {frac:.4f}")

assert np.all(np.diff(fractions) > 0), \
    "The coupled-away fraction should grow monotonically with propagation distance"
print("   PASS")

# ── 4. Locality ──────────────────────────────────────────────────────

print("\n4. Locality (nearby mode groups couple more than distant ones):")

p_loc = RandomModeCouplingPropagator(fiber, kappa=0.05, seed=2)
A_out_loc = p_loc.propagate(A0, dt, 20.0, step_size=5.0)
power_adjacent = np.sum(np.abs(A_out_loc[1]) ** 2)
power_distant = np.sum(np.abs(A_out_loc[-1]) ** 2)
print(f"   power in mode 1 (adjacent to launch mode 0):      {power_adjacent:.3e}")
print(f"   power in mode {M-1} (farthest from launch mode 0): {power_distant:.3e}")
assert power_adjacent > 100 * power_distant, \
    "Mode coupling should strongly favor nearby mode groups over distant ones (overlap-decay weighting)"
print("   PASS")

print("\n" + "=" * 70)
print("  All random mode coupling checks passed.")
print("=" * 70)
