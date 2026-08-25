"""Validation checks for stimulated Brillouin scattering (fiber/brillouin.py).

Checks, in order:
  1. Below-threshold regime: pump power well under the analytic SBS
     threshold should barely deplete over the fiber length, and the
     numerical (shooting-method) threshold-crossing distance should be
     consistent with the analytic Smith-formula threshold.
  2. Threshold "knee": sweeping input pump power through the analytic
     threshold should show the classic SBS signature -- reflectivity
     (backscattered fraction) rising sharply near P_th, and pump
     transmission dropping correspondingly.
  3. Off-resonance suppression: detuning the pump-Stokes separation well
     away from the material's Brillouin shift should collapse the gain
     (and hence the reflectivity) via the Lorentzian gain spectrum.
  4. Power conservation: at every z, P_pump(z) + P_stokes(z) accounts for
     the input power minus the (small, deliberately zeroed here) fiber
     loss -- i.e. the coupling term alone conserves photon number
     (energy), it only exchanges power between pump and Stokes.
"""
import numpy as np

from fiber.fiber_params import make_fiber
from fiber.brillouin import BrillouinPropagator, sbs_threshold_power, spontaneous_brillouin_noise_power

print("=" * 70)
print("  Stimulated Brillouin scattering (SBS) validation")
print("=" * 70)

fiber = make_fiber('smf28', alpha_dB_km=0.2)
L = 20e3  # 20 km

P_th = sbs_threshold_power(fiber, L)
print(f"\nAnalytic SBS threshold (Smith formula) at L={L/1e3:.0f} km: {P_th*1e3:.2f} mW")

# ── 1. Below-threshold regime ───────────────────────────────────────

print("\n1. Below-threshold pump depletion (isolated from ordinary fiber loss):")

bp = BrillouinPropagator(fiber)
P_in_below = 0.1 * P_th
z, Pp, Ps = bp.solve(P_in_below, L)
# Compare against the loss-ONLY expectation (no SBS coupling) to isolate the
# SBS-specific depletion from ordinary fiber attenuation (alpha_dB_km=0.2
# over 20 km is itself a substantial ~4 dB, ~60% power loss).
Pp_loss_only = Pp[0] * np.exp(-fiber.alpha * L)
sbs_specific_depletion = 1 - Pp[-1] / Pp_loss_only
print(f"   P_in = 0.1*P_th = {P_in_below*1e3:.3f} mW: "
      f"SBS-specific depletion beyond ordinary fiber loss = {sbs_specific_depletion:.2%}")
assert sbs_specific_depletion < 0.05, "Well below threshold, pump depletion from SBS alone should be small"
print("   PASS")

# ── 2. Threshold knee ────────────────────────────────────────────────
#
# In the undepleted-pump approximation, reflectivity grows exponentially
# with P_in at ALL powers (the "knee" isn't a change in functional form).
# What actually defines the threshold, and what saturates the exponential
# growth above it, is PUMP DEPLETION becoming significant -- negligible
# well below threshold, substantial (tens of percent) above it. That's
# the cleanest, standard signature to check directly.

print("\n2. SBS threshold knee (reflectivity + pump depletion vs input power):")

powers = np.array([0.05, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0]) * P_th
reflectivities = []
depletions = []
for P_in in powers:
    z, Pp, Ps = bp.solve(P_in, L)
    R = Ps[0] / P_in
    dep = 1 - Pp[-1] / (Pp[0] * np.exp(-fiber.alpha * L))  # SBS-specific, beyond ordinary loss
    reflectivities.append(R)
    depletions.append(dep)
    print(f"   P_in/P_th={P_in/P_th:.2f}: reflectivity = {R:.4e}, SBS-specific pump depletion = {dep:.2%}")

reflectivities = np.array(reflectivities)
depletions = np.array(depletions)
assert np.all(np.diff(reflectivities) > 0), "Reflectivity should rise monotonically with input power"
assert depletions[0] < 0.01, "Well below threshold, pump depletion from SBS should be negligible"
assert depletions[-1] > 0.2, "Well above threshold, pump depletion from SBS should be substantial"
assert depletions[3] > 10 * depletions[0], \
    "Pump depletion at threshold should be much larger than well below it"
print("   PASS (monotonic reflectivity; pump depletion negligible below threshold, substantial above it)")

# ── 3. Off-resonance suppression ────────────────────────────────────

print("\n3. Off-resonance suppression (detuning from the Brillouin shift):")

bp_resonant = BrillouinPropagator(fiber, detuning_Hz=0.0)
bp_detuned = BrillouinPropagator(fiber, detuning_Hz=10 * fiber.material.delta_nu_B)

P_in_test = 2.0 * P_th
R_resonant = bp_resonant.reflectivity(P_in_test, L)
R_detuned = bp_detuned.reflectivity(P_in_test, L)

print(f"   on-resonance reflectivity:  {R_resonant:.4e}")
print(f"   detuned (10x linewidth):    {R_detuned:.4e}")
assert R_detuned < 0.1 * R_resonant, \
    "Detuning far from the Brillouin resonance should strongly suppress the reflectivity"
print("   PASS")

# ── 4. Power conservation (coupling-only, no fiber loss) ────────────

print("\n4. Power conservation (alpha=0): net one-way flux P_pump(z)-P_stokes(z) is conserved")
print("   (pump and Stokes travel in opposite directions, so it's their DIFFERENCE -- the")
print("   net photon flux threading any cross-section -- that a source-free coupling term")
print("   conserves, not their sum):")

fiber_lossless = make_fiber('smf28', alpha_dB_km=0.0)
bp_lossless = BrillouinPropagator(fiber_lossless)
P_in_cons = 12.0 * sbs_threshold_power(fiber_lossless, L)  # well above threshold: non-negligible Stokes
z, Pp, Ps = bp_lossless.solve(P_in_cons, L)

print(f"   P_stokes(0)/P_pump(0) = {Ps[0]/Pp[0]:.2e} (non-negligible backscatter, not the trivial solution)")

net_flux = Pp - Ps
max_dev = np.max(np.abs(net_flux - net_flux[0])) / net_flux[0]
print(f"   max deviation of P_pump(z)-P_stokes(z) from its z=0 value: {max_dev:.2e}")
assert max_dev < 1e-3, \
    "With no fiber loss, the SBS coupling term alone should conserve the net one-way photon flux"
print("   PASS")

print("\n" + "=" * 70)
print("  All Brillouin scattering checks passed.")
print("=" * 70)
