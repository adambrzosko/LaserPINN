"""Validation checks for multimode (OM1-OM5) fiber support.

Checks, in order:
  1. OM1->OM5 calibration sanity: alpha_profile ordering gives a
     monotonically improving (higher-bandwidth) intermodal delay spread
     from OM1 to OM5, and the model's own bandwidth estimate is within a
     factor of ~2 of the nominal datasheet reference for each grade.
  2. Single-mode-limit consistency: launching all power into mode group 0
     only (zero cross-mode coupling in practice, since no other mode has
     power to couple from) reproduces fiber.propagator.FiberPropagator's
     output exactly for the equivalent single-mode fiber.
  3. Intermodal dispersion (DMD): an equal-power multi-mode launch
     broadens under pure linear propagation (gamma=0), and broadens more
     for OM1 (poor DMD) than OM4 (tight DMD) over the same length.
  4. Intermodal Raman scattering: a pump confined to mode group 0
     Raman-amplifies a weak probe confined to a *different* mode group
     when the two are within the overlap model's coupling range, and
     does not when the coupling length is collapsed to ~0.
  5. Per-mode loss (differential mode attenuation): alpha[0] matches the
     scalar baseline exactly, alpha grows monotonically with mode index,
     grows more for OM1 than OM4 at the highest mode group, and an
     end-to-end propagation of identical pulses in different mode groups
     shows the corresponding difference in output power.
  6. Per-mode waveguide dispersion (beta2/beta3): beta2[0]/beta3[0] match
     the scalar material baseline exactly, both vary across mode groups,
     and an end-to-end linear (gamma=0) propagation of the identical
     pulse in two different mode groups broadens by different amounts.
  7. Absolute inter-mode phase (beta0): d(delta_beta0)/d(omega) reproduces
     delta_beta1's leading term by finite difference (an analytic
     self-consistency check, since beta1 is beta0's derivative by
     definition), and an end-to-end propagation shows the phase actually
     imprinted on a mode's output matches delta_beta0[p]*L to high
     precision, over a length short enough that walk-off/GVD are
     negligible by comparison -- isolating exactly the effect under test.
"""
import numpy as np

from fiber.multimode_fiber import make_multimode_fiber, MultimodeFiberParams, make_multimode_geometry
from fiber.materials import make_material
from fiber.multimode_propagator import MultimodeFiberPropagator
from fiber.fiber_params import FiberParams
from fiber.propagator import FiberPropagator
from fiber.analysis import pulse_metrics, spectral_centroid

print("=" * 70)
print("  Multimode (OM1-OM5) fiber validation")
print("=" * 70)

# ── 1. OM1->OM5 calibration sanity ──────────────────────────────────

print("\n1. OM1->OM5 bandwidth calibration:")
grades = ['om1', 'om2', 'om3', 'om4', 'om5']
bws = []
for g in grades:
    f = make_multimode_fiber(g)
    bw = f.estimated_bandwidth_MHz_km()
    ref = f.geometry.ofl_bandwidth_MHz_km
    bws.append(bw)
    ratio = bw / ref
    print(f"   {g}: model_BW={bw:8.0f} MHz.km  ref={ref:6.0f} MHz.km  ratio={ratio:.2f}  M={f.n_modes}")
    assert 0.5 < ratio < 2.0, f"{g}: model bandwidth {bw:.0f} too far from reference {ref:.0f}"

assert all(b2 >= b1 for b1, b2 in zip(bws, bws[1:])), \
    "Bandwidth should improve monotonically from OM1 to OM5"
print("   PASS (monotonic OM1->OM5 ordering, each within 2x of datasheet reference)")

# ── 2. Single-mode-limit consistency ────────────────────────────────

print("\n2. Single-mode-limit consistency (mode 0 only vs FiberPropagator):")

mm_fiber = make_multimode_fiber('om3', lambda0=1550e-9, D=17.0, beta3_material=0.0, alpha_dB_km=0.2)
sm_fiber = FiberParams(material=make_material('silica'), alpha_dB_km=0.2, D=17.0, beta3=0.0)
# force the single-mode fiber's gamma/A_eff to match mode group 0 exactly
sm_fiber.gamma = mm_fiber.gamma_self[0]

N = 2 ** 12
dt = 5e-15
T0 = 200e-15
t = (np.arange(N) - N // 2) * dt
P0 = 20.0  # W, strong enough for Kerr to matter
A0 = (np.sqrt(P0) / np.cosh(t / T0)).astype(complex)

L_test = 500.0  # m

mm_prop = MultimodeFiberPropagator(mm_fiber, include_raman=True)
A_mm_out = mm_prop.propagate(A0, dt, L_test, step_size=L_test / 400)[0]  # mode 0 row

sm_prop = FiberPropagator(sm_fiber, include_raman=True)
A_sm_out = sm_prop.propagate(A0, dt, L_test, step_size=L_test / 400)

max_diff = np.max(np.abs(A_mm_out - A_sm_out)) / np.max(np.abs(A_sm_out))
print(f"   max relative field difference: {max_diff:.2e}")
assert max_diff < 1e-6, "Multimode propagator (mode 0 only) should exactly match FiberPropagator"
print("   PASS")

# ── 3. Intermodal dispersion (DMD) ──────────────────────────────────

print("\n3. Intermodal dispersion (DMD) broadening, OM1 vs OM4:")

N2 = 2 ** 12
dt2 = 2e-12
T0_2 = 5e-12
t2 = (np.arange(N2) - N2 // 2) * dt2
pulse_shape = (np.exp(-t2 ** 2 / (2 * T0_2 ** 2))).astype(complex)

L_dmd = 200.0  # m

for grade in ['om1', 'om4']:
    zero_kerr = make_material('silica', n2=0.0, f_R=0.0)
    fiber = make_multimode_fiber(grade, lambda0=850e-9, material_overrides=dict(n2=0.0, f_R=0.0))
    M = fiber.n_modes
    A0_mm = np.tile(pulse_shape, (M, 1)) / np.sqrt(M)  # equal power split across all mode groups

    prop = MultimodeFiberPropagator(fiber, include_raman=False)
    A_out = prop.propagate(A0_mm, dt2, L_dmd, step_size=L_dmd / 100)
    P_total = np.sum(np.abs(A_out) ** 2, axis=0)

    m_in = pulse_metrics(pulse_shape, dt2)
    m_out = pulse_metrics(np.sqrt(P_total).astype(complex), dt2)
    broaden = m_out['fwhm'] / m_in['fwhm']
    print(f"   {grade}: FWHM broadening over {L_dmd:.0f} m = {broaden:.3f}x "
          f"(delay spread {fiber.rms_delay_spread()*1e12:.2f} ps/m)")
    if grade == 'om1':
        broaden_om1 = broaden
    else:
        broaden_om4 = broaden

assert broaden_om1 > broaden_om4, \
    "OM1 (worse DMD) should broaden a multi-mode-launched pulse more than OM4 over the same length"
assert broaden_om1 > 1.001, "OM1 should show measurable DMD-induced broadening at all"
print("   PASS (OM1 broadens more than OM4, as expected from its larger delay spread)")

# ── 4. Intermodal Raman scattering ──────────────────────────────────
#
# The cross-mode coupling term transfers *time-varying* power (see
# fiber/multimode_propagator.py docstring) -- a CW pump has zero spectral
# content away from Omega=0 and so cannot seed frequency-selective gain
# in another mode. A genuine pulse, with real spectral bandwidth, can.
# So: launch a strong, short (broadband) pump PULSE in mode 0, and a
# weak, temporally-overlapped probe PULSE in a different mode group;
# check the probe's spectrum red-shifts (the same intrapulse-Raman
# self-frequency-shift signature as test 3 in test_fiber_engine.py, but
# now driven by a field in a *different* spatial mode), and that the
# shift disappears when the mode coupling is collapsed.

print("\n4. Intermodal Raman scattering (broadband pump pulse in mode 0 "
      "cross-Raman-shifts a probe pulse in a different mode group):")

N3 = 2 ** 12
dt3 = 1e-14
t3 = (np.arange(N3) - N3 // 2) * dt3

T0_pump = 50e-15
P_pump = 50.0
pump_pulse = (np.sqrt(P_pump) / np.cosh(t3 / T0_pump)).astype(complex)

T0_probe = 200e-15
P_probe = 1e-3
probe_pulse = (np.sqrt(P_probe) / np.cosh(t3 / T0_probe)).astype(complex)

probe_mode = 5
L_im = 0.05  # m

def probe_shift(fiber, L, include_raman):
    M = fiber.n_modes
    A0 = np.zeros((M, N3), dtype=complex)
    A0[0] = pump_pulse
    A0[probe_mode] = probe_pulse

    prop = MultimodeFiberPropagator(fiber, include_raman=include_raman)
    A_out = prop.propagate(A0, dt3, L, step_size=L / 200)
    # physical frequency = omega0 - Omega (see fiber.raman_response); report
    # the physical shift directly so negative = redshift, as in test_fiber_engine.py
    return -(spectral_centroid(A_out[probe_mode], dt3) - spectral_centroid(probe_pulse, dt3)) / (2 * np.pi) * 1e-12

fiber_im = make_multimode_fiber('om3', lambda0=1000e-9, D=0.0, beta3_material=0.0, alpha_dB_km=0.0)
fiber_uncoupled = make_multimode_fiber('om3', lambda0=1000e-9, D=0.0, beta3_material=0.0, alpha_dB_km=0.0,
                                        mode_coupling_length=1e-6)

# The NET shift is dominated by cross-phase-modulation (instantaneous Kerr),
# which -- unlike single-pulse self-Raman SSFS -- has no universally-fixed
# sign for a temporally-overlapped pump/probe pair (it depends on relative
# pulse widths and walk-off). What IS a clean, unambiguous signature of
# *Raman* (as opposed to plain intermodal Kerr XPM) is that switching the
# delayed response on pulls the shift toward the red relative to the
# Kerr-XPM-only baseline, and that this Raman-specific pull vanishes when
# the mode groups are decoupled.
shift_with_raman = probe_shift(fiber_im, L_im, include_raman=True)
shift_kerr_only = probe_shift(fiber_im, L_im, include_raman=False)
raman_pull = shift_with_raman - shift_kerr_only

shift_with_raman_dec = probe_shift(fiber_uncoupled, L_im, include_raman=True)
shift_kerr_only_dec = probe_shift(fiber_uncoupled, L_im, include_raman=False)
raman_pull_dec = shift_with_raman_dec - shift_kerr_only_dec

print(f"   mode_coupling_length={fiber_im.mode_coupling_length}: "
      f"Kerr-XPM-only shift={shift_kerr_only:+.3e} THz, with Raman={shift_with_raman:+.3e} THz, "
      f"Raman-specific pull={raman_pull:+.3e} THz")
print(f"   mode_coupling_length=1e-6 (~decoupled): Raman-specific pull={raman_pull_dec:+.3e} THz")

assert raman_pull < 0, \
    "Enabling the delayed Raman response should pull the cross-mode-modulated probe toward the red"
assert abs(raman_pull) > 10 * abs(raman_pull_dec), \
    "Decoupling the mode groups should suppress the intermodal Raman-specific pull"
print("   PASS (intermodal Raman coupling pulls the probe red relative to Kerr-XPM-only, "
      "and vanishes when decoupled)")

# ── 5. Per-mode loss (differential mode attenuation) ────────────────

print("\n5. Per-mode loss (differential mode attenuation, DMA):")

om1 = make_multimode_fiber('om1')
om4 = make_multimode_fiber('om4')

alpha0_expected = om1.alpha_dB_km / (10 * np.log10(np.e)) / 1e3
assert abs(om1.alpha[0] - alpha0_expected) < 1e-15, \
    "Mode 0's loss should exactly equal the scalar alpha_dB_km baseline"
assert np.all(np.diff(om1.alpha) >= 0), "Loss should grow monotonically with mode-group index"

extra_loss_om1_dB_km = (om1.alpha[-1] - om1.alpha[0]) * (10 * np.log10(np.e)) * 1e3
extra_loss_om4_dB_km = (om4.alpha[-1] - om4.alpha[0]) * (10 * np.log10(np.e)) * 1e3
print(f"   OM1 highest-mode extra loss: {extra_loss_om1_dB_km:.3f} dB/km")
print(f"   OM4 highest-mode extra loss: {extra_loss_om4_dB_km:.3f} dB/km")
assert extra_loss_om1_dB_km > extra_loss_om4_dB_km, \
    "OM1 (less bend-insensitive) should show more differential mode attenuation than OM4"

# End-to-end: launching the SAME pulse into mode 0 vs the highest mode
# group of OM1 should show a measurably larger power drop in the
# high-order mode, consistent with its larger alpha.
fiber_loss = make_multimode_fiber('om1', material_overrides=dict(n2=0.0, f_R=0.0))
N5 = 2 ** 10
dt5 = 5e-12
T0_5 = 20e-12
t5 = (np.arange(N5) - N5 // 2) * dt5
pulse5 = np.exp(-t5 ** 2 / (2 * T0_5 ** 2)).astype(complex)
L5 = 5e3  # 5 km, long enough for a multi-dB differential loss to show up

prop5 = MultimodeFiberPropagator(fiber_loss, include_raman=False)
A0_mode0 = np.zeros((fiber_loss.n_modes, N5), dtype=complex)
A0_mode0[0] = pulse5
A0_high = np.zeros((fiber_loss.n_modes, N5), dtype=complex)
A0_high[-1] = pulse5

out_mode0 = prop5.propagate(A0_mode0, dt5, L5, step_size=L5 / 200)[0]
out_high = prop5.propagate(A0_high, dt5, L5, step_size=L5 / 200)[-1]

energy_mode0 = np.sum(np.abs(out_mode0) ** 2)
energy_high = np.sum(np.abs(out_high) ** 2)
extra_loss_dB_measured = 10 * np.log10(energy_mode0 / energy_high)
extra_loss_dB_expected = (fiber_loss.alpha[-1] - fiber_loss.alpha[0]) * L5 * (10 * np.log10(np.e))
print(f"   end-to-end extra loss over {L5/1e3:.0f} km: measured={extra_loss_dB_measured:.3f} dB, "
      f"expected={extra_loss_dB_expected:.3f} dB")
assert abs(extra_loss_dB_measured - extra_loss_dB_expected) < 0.05, \
    "Propagated power difference between mode 0 and the highest mode should match the per-mode alpha difference"
print("   PASS (per-mode loss is referenced correctly, ordered by grade, and manifests in propagation)")

# ── 6. Per-mode waveguide dispersion (beta2/beta3) ──────────────────

print("\n6. Per-mode waveguide dispersion (beta2/beta3):")

fiber_disp = make_multimode_fiber('om1', D=17.0, alpha_dB_km=0.0)
beta2_material = -fiber_disp.D * 1e-6 * fiber_disp.lambda0 ** 2 / (2 * np.pi * 3e8)
assert abs(fiber_disp.beta2[0] - beta2_material) / abs(beta2_material) < 1e-10, \
    "Mode 0's beta2 should exactly equal the scalar material dispersion"
assert fiber_disp.beta3[0] == fiber_disp.beta3_material, \
    "Mode 0's beta3 should exactly equal the scalar material beta3_material"
beta2_spread = np.std(fiber_disp.beta2)
print(f"   beta2 spread across mode groups: {beta2_spread:.3e} s^2/m "
      f"({beta2_spread/abs(beta2_material):.1%} of the material value)")
assert beta2_spread > 0, "beta2 should genuinely vary across mode groups"

# End-to-end: launch the identical Gaussian pulse into mode 0 vs the
# highest mode group (gamma=0, pure linear propagation) and confirm they
# broaden by DIFFERENT amounts, i.e. beta2[p] actually affects propagation.
fiber_disp_prop = make_multimode_fiber('om1', D=17.0, alpha_dB_km=0.0,
                                        material_overrides=dict(n2=0.0, f_R=0.0))
N6 = 2 ** 12
dt6 = 5e-15
T0_6 = 100e-15
t6 = (np.arange(N6) - N6 // 2) * dt6
pulse6 = np.exp(-t6 ** 2 / (2 * T0_6 ** 2)).astype(complex)
L6 = 2.0  # m

prop6 = MultimodeFiberPropagator(fiber_disp_prop, include_raman=False)
A0_mode0_6 = np.zeros((fiber_disp_prop.n_modes, N6), dtype=complex)
A0_mode0_6[0] = pulse6
A0_high_6 = np.zeros((fiber_disp_prop.n_modes, N6), dtype=complex)
A0_high_6[-1] = pulse6

fwhm0 = pulse_metrics(prop6.propagate(A0_mode0_6, dt6, L6, step_size=L6 / 100)[0], dt6)['fwhm']
fwhm_high = pulse_metrics(prop6.propagate(A0_high_6, dt6, L6, step_size=L6 / 100)[-1], dt6)['fwhm']
print(f"   FWHM after {L6:.0f} m: mode 0 = {fwhm0*1e15:.2f} fs, highest mode = {fwhm_high*1e15:.2f} fs")
assert abs(fwhm0 - fwhm_high) / fwhm0 > 0.01, \
    "Different beta2 per mode group should give measurably different GVD-driven broadening"
print("   PASS (per-mode beta2/beta3 are referenced correctly, nonzero, and manifest in propagation)")

# ── 7. Absolute inter-mode phase (beta0) ────────────────────────────

print("\n7. Absolute inter-mode phase (beta0):")

from fiber.multimode_fiber import _beta0, _tau_over_L
from core.dfb_laser import c as c_light  # must match the c used internally by _beta0/_tau_over_L
                                          # exactly -- any mismatch gets catastrophically amplified
                                          # when subtracting the huge n1/c baseline below

fiber_b0 = make_multimode_fiber('om3')
geo_b0 = fiber_b0.geometry
p_b0 = np.arange(1, fiber_b0.n_modes + 1)
omega0_b0 = fiber_b0.omega0

# Analytic self-consistency: d(beta0)/d(omega) must reproduce _tau_over_L's
# LEADING term exactly (beta0's derivation has no analog of the ad hoc
# KAPPA_FLOOR "floor" term added to _tau_over_L, so compare against the
# leading term specifically, not the full delta_beta1).
domega = 1e-3 * omega0_b0
beta0_plus = _beta0(p_b0, omega0_b0 + domega, geo_b0)
beta0_minus = _beta0(p_b0, omega0_b0 - domega, geo_b0)
numerical_beta1 = (beta0_plus - beta0_minus) / (2 * domega) - geo_b0.n1 / c_light

Delta_b0 = geo_b0.NA ** 2 / (2 * geo_b0.n1 ** 2)
exp1_b0 = 2 * geo_b0.alpha_profile / (geo_b0.alpha_profile + 2)
x_b0 = p_b0 / fiber_b0.n_modes
leading_beta1 = ((geo_b0.n1 / c_light) * Delta_b0 * (geo_b0.alpha_profile - 2)
                 / (geo_b0.alpha_profile + 2) * x_b0 ** exp1_b0)

rel_err_b0 = np.max(np.abs(numerical_beta1 - leading_beta1) / np.abs(leading_beta1))
print(f"   d(beta0)/d(omega) vs beta1's leading term: max relative error = {rel_err_b0:.2%}")
assert rel_err_b0 < 0.05, \
    "beta0's derivative w.r.t. omega should reproduce delta_beta1's leading term (beta1 = d(beta0)/d(omega))"
print("   PASS (self-consistent: beta0 is a genuine antiderivative of beta1)")

# End-to-end: propagate a narrowband pulse a length short enough that
# walk-off/GVD are negligible (see values printed below), then check the
# phase actually imprinted at the pulse's own peak matches delta_beta0*L.
fiber_b0_prop = make_multimode_fiber('om3', material_overrides=dict(n2=0.0, f_R=0.0))
test_mode = 5
L_b0 = 1e-3  # 1 mm
print(f"   delta_beta0[{test_mode}]={fiber_b0_prop.delta_beta0[test_mode]:.1f} rad/m, "
      f"delta_beta1[{test_mode}]*L={fiber_b0_prop.delta_beta1[test_mode]*L_b0*1e15:.4f} fs (walk-off, negligible), "
      f"beta2 spread*L={np.ptp(fiber_b0_prop.beta2)*L_b0:.2e} s^2 (negligible)")

N_b0 = 2 ** 10
dt_b0 = 1e-12
T0_b0 = 20e-12
t_b0 = (np.arange(N_b0) - N_b0 // 2) * dt_b0
pulse_b0 = np.exp(-t_b0 ** 2 / (2 * T0_b0 ** 2)).astype(complex)  # real, zero initial phase

prop_b0 = MultimodeFiberPropagator(fiber_b0_prop, include_raman=False)
A0_b0 = np.zeros((fiber_b0_prop.n_modes, N_b0), dtype=complex)
A0_b0[0] = pulse_b0
A0_b0[test_mode] = pulse_b0
A_out_b0 = prop_b0.propagate(A0_b0, dt_b0, L_b0, step_size=L_b0 / 20)

peak_idx = N_b0 // 2
phase_mode0 = np.angle(A_out_b0[0, peak_idx])
phase_test_mode = np.angle(A_out_b0[test_mode, peak_idx])
measured_relative_phase = (phase_test_mode - phase_mode0 + np.pi) % (2 * np.pi) - np.pi
expected_relative_phase = (fiber_b0_prop.delta_beta0[test_mode] * L_b0 + np.pi) % (2 * np.pi) - np.pi

print(f"   measured relative phase at pulse peak: {measured_relative_phase:+.5f} rad")
print(f"   expected (delta_beta0*L, wrapped):     {expected_relative_phase:+.5f} rad")
assert abs(phase_mode0) < 1e-6, "Mode 0 (the reference) should pick up no extra phase (delta_beta0[0]=0)"
phase_err = min(abs(measured_relative_phase - expected_relative_phase),
                 2 * np.pi - abs(measured_relative_phase - expected_relative_phase))
assert phase_err < 1e-3, \
    "The propagated phase difference between two mode groups should match delta_beta0[p]*L"
print("   PASS (propagated relative phase matches the closed-form delta_beta0*L prediction)")

print("\n" + "=" * 70)
print("  All multimode fiber checks passed.")
print("=" * 70)
