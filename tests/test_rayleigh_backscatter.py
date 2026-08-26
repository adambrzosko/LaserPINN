"""Validation checks for elastic Rayleigh backscattering
(fiber/rayleigh_backscatter.py).

Checks, in order:
  1. Closed-form vs brute-force: rayleigh_backscatter_power's closed-form
     integral matches direct numerical integration of the same local-
     generation-plus-double-pass-attenuation physics.
  2. Saturation with length: total backscattered power grows with L but
     saturates to the analytic asymptote alpha_R*S*P_in/(2*alpha) as
     L->infinity, rather than growing without bound (light scattered far
     down the fiber is itself heavily attenuated on the way back).
  3. Self-consistency: integrating the time-resolved OTDR trace over
     return time reproduces the same total power as the CW formula.
"""
import numpy as np

from fiber.fiber_params import make_fiber
from fiber.rayleigh_backscatter import rayleigh_backscatter_power, rayleigh_otdr_trace

print("=" * 70)
print("  Elastic Rayleigh backscattering validation")
print("=" * 70)

fiber = make_fiber('smf28')
P_in = 1e-3  # 1 mW
L = 20e3
alpha_R_frac = 0.9
S = 2e-3

# ── 1. Closed-form vs brute-force numerical integration ─────────────

print("\n1. Closed-form vs brute-force numerical integration:")

alpha = fiber.alpha
alpha_R = alpha_R_frac * alpha
z = np.linspace(0, L, 200_000)
dz = z[1] - z[0]
integrand = alpha_R * S * P_in * np.exp(-2 * alpha * z)
numeric = np.sum(integrand) * dz
closed_form = rayleigh_backscatter_power(fiber, P_in, L, alpha_R_fraction=alpha_R_frac, S_capture=S)
rel_diff = abs(numeric - closed_form) / closed_form
print(f"   brute-force numerical integral: {numeric:.6e} W")
print(f"   closed-form formula:            {closed_form:.6e} W")
print(f"   relative difference: {rel_diff:.2e}")
assert rel_diff < 1e-4, "The closed-form backscatter formula should match brute-force integration closely"
print("   PASS")

# ── 2. Saturation with length ────────────────────────────────────────

print("\n2. Saturation with fiber length:")

lengths = [1e3, 20e3, 200e3, 2e6]
powers = [rayleigh_backscatter_power(fiber, P_in, Lt, alpha_R_fraction=alpha_R_frac, S_capture=S)
          for Lt in lengths]
for Lt, p in zip(lengths, powers):
    print(f"   L={Lt/1e3:8.0f} km: P_back = {p:.6e} W")

asymptote = alpha_R * S * P_in / (2 * alpha)
print(f"   analytic asymptote (L->infinity): {asymptote:.6e} W")

assert np.all(np.diff(powers) >= 0), "Backscattered power should grow monotonically with length"
assert abs(powers[-1] - asymptote) / asymptote < 1e-3, \
    "At very large L, backscattered power should saturate to the analytic asymptote"
assert powers[0] < 0.5 * asymptote, "At short L, backscattered power should be well below the asymptote"
print("   PASS")

# ── 3. OTDR trace self-consistency ──────────────────────────────────

print("\n3. OTDR trace integrates to the same total as the CW formula:")

v_g = 2e8  # m/s, representative silica group velocity
t_max = 2 * L / v_g
t = np.linspace(0, t_max, 200_000)
dt = t[1] - t[0]
trace = rayleigh_otdr_trace(fiber, P_in, L, t, v_g, alpha_R_fraction=alpha_R_frac, S_capture=S)
integrated = np.sum(trace) * dt
rel_diff_trace = abs(integrated - closed_form) / closed_form
print(f"   integral of OTDR trace over return time: {integrated:.6e} W")
print(f"   total power formula:                     {closed_form:.6e} W")
print(f"   relative difference: {rel_diff_trace:.2e}")
assert rel_diff_trace < 1e-4, "The OTDR trace, integrated over time, should reproduce the total backscatter power"

# trace should be exactly zero beyond the round trip to L
t_beyond = np.array([t_max * 1.5])
trace_beyond = rayleigh_otdr_trace(fiber, P_in, L, t_beyond, v_g, alpha_R_fraction=alpha_R_frac, S_capture=S)
assert trace_beyond[0] == 0.0, "The OTDR trace should be exactly zero beyond the round trip to the fiber end"
print("   PASS")

print("\n" + "=" * 70)
print("  All Rayleigh backscattering checks passed.")
print("=" * 70)
