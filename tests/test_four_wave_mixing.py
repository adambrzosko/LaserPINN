"""Validation checks for four-wave mixing (fiber/four_wave_mixing.py).

Checks, in order:
  1. Phase-matching efficiency: eta=1 exactly at perfect phase match
     (Delta_beta=0), in both the lossy and lossless limits, and
     decreases as the triplet's frequency spacing (and hence dispersion-
     driven phase mismatch) grows.
  2. Degeneracy scaling: for otherwise identical pump powers/frequencies,
     the non-degenerate (D=2) case generates exactly 4x the power of the
     degenerate (D=1) case, i.e. exactly D^2 -- a direct check of the
     formula's own internal consistency, not an external reference.
  3. Ghost-tone sweep on an equally-spaced grid: an FWM product from a
     3-channel equal-spacing comb lands EXACTLY on the middle channel's
     own slot -- the classic, well-documented reason equally-spaced WDM
     grids are avoided in real high-power systems -- and
     fwm_ghost_tone_power correctly finds and sums that contribution.
"""
import numpy as np

from fiber.fiber_params import make_fiber
from fiber.four_wave_mixing import fwm_efficiency, fwm_power, fwm_ghost_tone_power

print("=" * 70)
print("  Four-wave mixing validation")
print("=" * 70)

fiber = make_fiber('smf28', D=0.5)  # near-zero dispersion, favorable for FWM
L = 10e3

# ── 1. Phase-matching efficiency ────────────────────────────────────

print("\n1. Phase-matching efficiency (eta):")

eta_matched = fwm_efficiency(fiber, 0.0, 0.0, 0.0, L)
print(f"   eta at exact phase match: {eta_matched:.6f}")
assert abs(eta_matched - 1.0) < 1e-12, "eta should be exactly 1 at perfect phase matching"

fiber_lossless = make_fiber('smf28', D=0.5, alpha_dB_km=0.0)
eta_matched_lossless = fwm_efficiency(fiber_lossless, 0.0, 0.0, 0.0, L)
print(f"   eta at exact phase match (lossless limit): {eta_matched_lossless:.6f}")
assert abs(eta_matched_lossless - 1.0) < 1e-12, "eta should also be exactly 1 at phase match in the lossless limit"

w_close = 2 * np.pi * 100e9
w_far = 2 * np.pi * 1000e9
eta_close = fwm_efficiency(fiber, w_close, w_close, -w_close, L)
eta_far = fwm_efficiency(fiber, w_far, w_far, -w_far, L)
print(f"   eta, 100 GHz-scale triplet spacing:  {eta_close:.4e}")
print(f"   eta, 1000 GHz-scale triplet spacing: {eta_far:.4e}")
assert 0 < eta_close < 1, "Efficiency should be strictly between 0 and 1 away from exact phase match"
assert eta_far < eta_close, "Wider-spaced triplets (larger dispersion-driven mismatch) should be less efficient"
print("   PASS")

# ── 2. Degeneracy scaling ────────────────────────────────────────────

print("\n2. Degeneracy scaling (D^2):")

P = 0.01
w_i = 2 * np.pi * 100e9
w_k = -2 * np.pi * 100e9
p_degenerate = fwm_power(fiber, P, P, P, w_i, w_i, w_k, L)  # omega_i == omega_j -> D=1
p_nondegenerate = fwm_power(fiber, P, P, P, w_i, w_i + 1.0, w_k, L)  # tiny offset -> distinct -> D=2
ratio = p_nondegenerate / p_degenerate
print(f"   degenerate (D=1) power:     {p_degenerate:.4e} W")
print(f"   non-degenerate (D=2) power: {p_nondegenerate:.4e} W")
print(f"   ratio: {ratio:.4f} (want exactly 4.0)")
assert abs(ratio - 4.0) < 1e-6, "Non-degenerate FWM should generate exactly D^2=4x the degenerate power"
print("   PASS")

# ── 3. Ghost-tone sweep on an equally-spaced grid ───────────────────

print("\n3. Ghost tone landing exactly on a channel slot (equal-spacing grid):")

offsets = [-100e9, 0.0, 100e9]
powers = [0.01, 0.01, 0.01]
result = fwm_ghost_tone_power(fiber, powers, offsets, target_offset_Hz=0.0, L=L, tolerance_Hz=1e9)

print(f"   total ghost-tone power at the middle channel's slot: {result['total_power_W']:.4e} W")
for (i, j, k, idler_Hz, p) in result['contributions']:
    print(f"     triplet (i={i}, j={j}, k={k}): idler at {idler_Hz/1e9:+.1f} GHz, power={p:.4e} W")

assert result['total_power_W'] > 0, "The classic equal-spacing FWM ghost tone should be found"
assert any(i == 0 and j == 2 and k == 1 for (i, j, k, _, _) in result['contributions']), \
    "Expected the (channel0, channel2, channel1) triplet to land on channel 1's slot"

# A target far from any triplet's product should find nothing
result_empty = fwm_ghost_tone_power(fiber, powers, offsets, target_offset_Hz=5e12, L=L, tolerance_Hz=1e9)
print(f"   total power at an unrelated 5 THz target: {result_empty['total_power_W']:.4e} W "
      f"({len(result_empty['contributions'])} contributions)")
assert result_empty['total_power_W'] == 0.0, "A target far from every triplet's product should find nothing"
print("   PASS")

print("\n" + "=" * 70)
print("  All four-wave mixing checks passed.")
print("=" * 70)
