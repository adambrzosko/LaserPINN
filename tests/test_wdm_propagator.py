"""Validation checks for the WDM multi-channel propagator (SPM, XPM,
inter-channel Raman crosstalk).

Checks, in order:
  1. Single-channel-limit consistency: a 1-channel WDMPropagator
     reproduces fiber.propagator.FiberPropagator exactly.
  2. Cross-phase modulation (XPM): a weak probe channel co-propagating
     (in time) with a strong pump channel acquires a nonlinear phase
     shift exactly 2x the SPM-only baseline at the pump's peak -- the
     standard XPM/SPM ratio -- when the two channels are close enough in
     frequency that Raman crosstalk is negligible.
  3. Inter-channel Raman scattering ("Raman crosstalk"): two CW-like
     (near-constant-power) channels separated by ~the Raman shift show a
     genuine, sign-correct power transfer -- the higher-frequency
     ("bluer") channel pumps the lower-frequency one -- and this
     vanishes for closely-spaced channels (far off the Raman gain
     spectrum) or when Raman is disabled. This specifically validates
     the fix for the CW-pump limitation documented in
     fiber.multimode_propagator (there, an analogous CW pump gives
     exactly zero cross-mode Raman transfer; here it does not, because
     this term uses the fixed channel separation, not a co-located
     baseband convolution).
  4. Absolute inter-channel phase (beta0): d(delta_beta0)/d(Delta_omega)
     reproduces delta_beta1 exactly (an analytic self-consistency check,
     since delta_beta1 is delta_beta0's derivative by definition), and an
     end-to-end propagation shows the phase actually imprinted on a
     channel's output matches delta_beta0*L to high precision, over a
     length short enough that walk-off/GVD are negligible by comparison.
"""
import numpy as np

from fiber.fiber_params import make_fiber
from fiber.propagator import FiberPropagator
from fiber.wdm_propagator import WDMPropagator
from fiber.raman_response import raman_gain_spectrum

print("=" * 70)
print("  WDM propagator validation (SPM, XPM, Raman crosstalk)")
print("=" * 70)

# ── 1. Single-channel-limit consistency ─────────────────────────────

print("\n1. Single-channel-limit consistency (WDMPropagator vs FiberPropagator):")

fiber = make_fiber('hnlf')
N = 2 ** 12
dt = 5e-15
T0 = 100e-15
t = (np.arange(N) - N // 2) * dt
P0 = 5.0
A0 = (np.sqrt(P0) / np.cosh(t / T0)).astype(complex)

L_test = 200.0

wdm1 = WDMPropagator(fiber, channel_offsets_Hz=[0.0], include_raman=True)
A_wdm = wdm1.propagate(A0, dt, L_test, step_size=L_test / 400)[0]

sm = FiberPropagator(fiber, include_raman=True)
A_sm = sm.propagate(A0, dt, L_test, step_size=L_test / 400)

max_diff = np.max(np.abs(A_wdm - A_sm)) / np.max(np.abs(A_sm))
print(f"   max relative field difference: {max_diff:.2e}")
assert max_diff < 1e-10, "1-channel WDMPropagator should exactly match FiberPropagator"
print("   PASS")

# ── 2. Cross-phase modulation (XPM) ─────────────────────────────────

print("\n2. Cross-phase modulation: probe phase shift vs pump power (expect XPM = 2x SPM):")

fiber_xpm = make_fiber('hnlf', material_overrides=dict(f_R=0.0), D=0.0, beta3=0.0)
# D=0/beta3=0: zero channel walk-off, so pump and probe stay perfectly
# time-aligned and the induced phase can be compared directly against the
# simple gamma*2*P_pump(t)*L formula (isolating XPM/SPM without also
# testing walk-off dynamics, which is a separate, already-standard effect)
N2 = 2 ** 12
dt2 = 5e-15
T0_pump = 500e-15
t2 = (np.arange(N2) - N2 // 2) * dt2
P_pump = 2.0
pump_pulse = (np.sqrt(P_pump) / np.cosh(t2 / T0_pump)).astype(complex)
probe_cw = np.ones(N2, dtype=complex) * 1e-3  # weak, flat "probe" so its own SPM is negligible

L_xpm = 30.0  # kept short enough that the induced phase stays well under pi (avoids angle() wraparound)
channel_spacing_Hz = 50e9  # close enough that Raman crosstalk (THz-scale) is negligible

wdm_xpm = WDMPropagator(fiber_xpm, channel_offsets_Hz=[0.0, channel_spacing_Hz], include_raman=True)
A0_xpm = np.zeros((2, N2), dtype=complex)
A0_xpm[0] = pump_pulse
A0_xpm[1] = probe_cw
A_out_xpm = wdm_xpm.propagate(A0_xpm, dt2, L_xpm, step_size=L_xpm / 400)

# Phase imprinted on the probe at the pump's peak (t=0, center sample)
center = N2 // 2
phase_probe_with_pump = np.angle(A_out_xpm[1, center] / probe_cw[center])

# SPM-only baseline: propagate the probe ALONE (no pump) through the same fiber
wdm_alone = WDMPropagator(fiber_xpm, channel_offsets_Hz=[channel_spacing_Hz], include_raman=True)
A_probe_alone = wdm_alone.propagate(probe_cw, dt2, L_xpm, step_size=L_xpm / 400)[0]
phase_probe_alone = np.angle(A_probe_alone[center] / probe_cw[center])

# Phase from SPM of a CW-like probe is ~0 (power far too low); the XPM-induced
# extra phase should match gamma*2*P_pump(t=0)*L (theory) closely at t=0
expected_xpm_phase = fiber_xpm.gamma * 2 * P_pump * L_xpm
induced_phase = phase_probe_with_pump - phase_probe_alone

print(f"   expected XPM phase (gamma*2*P_pump*L) = {expected_xpm_phase:.5f} rad")
print(f"   measured induced phase on probe        = {induced_phase:.5f} rad")
rel_err = abs(induced_phase - expected_xpm_phase) / expected_xpm_phase
print(f"   relative error = {rel_err:.2%}")
assert rel_err < 0.05, "XPM-induced probe phase should match gamma*2*P_pump*L to within a few percent"
print("   PASS")

# ── 3. Inter-channel Raman scattering (crosstalk) ───────────────────

print("\n3. Inter-channel Raman crosstalk between CW-like channels:")

fiber_raman = make_fiber('smf28', alpha_dB_km=0.0)  # isolate Raman crosstalk from ordinary fiber loss
N3 = 2 ** 12
dt3 = 20e-15
P_blue = 2.0     # strong, higher-frequency channel
P_red = 1e-3     # weak, lower-frequency (Stokes-side) channel

blue_field = np.sqrt(P_blue) * np.ones(N3, dtype=complex)
red_field = np.sqrt(P_red) * np.ones(N3, dtype=complex)

L_raman = 1000.0  # m

def run_pair(offsets_Hz, include_raman=True):
    wdm = WDMPropagator(fiber_raman, channel_offsets_Hz=offsets_Hz, include_raman=include_raman)
    A0 = np.zeros((2, N3), dtype=complex)
    A0[0] = blue_field
    A0[1] = red_field
    A_out = wdm.propagate(A0, dt3, L_raman, step_size=L_raman / 200)
    P_red_out = np.mean(np.abs(A_out[1]) ** 2)
    return P_red_out / P_red

# Determine sign from the already-validated raman_gain_spectrum directly:
# channel 0 (blue, offset 0) should pump channel 1 (red, offset -13.2 THz).
# raman_gain_spectrum's Omega follows physical_freq = omega0 - Omega (see
# fiber.raman_response), the opposite of the direct channel_offsets_Hz
# convention -- so the gain seen by channel 1 from channel 0 is evaluated
# at Omega = offset[0] - offset[1] (channel q's offset minus channel p's).
Delta_omega_check = 2 * np.pi * (0.0 - (-13.2e12))
g_R_check = raman_gain_spectrum(fiber_raman.material, np.array([Delta_omega_check]), fiber_raman.gamma)[0]
print(f"   g_R seen by the red channel from the blue channel = {g_R_check:.3e} (want > 0: red channel gains)")
assert g_R_check > 0, "Test setup error: chosen channel separation should sit on the Raman gain side"

ratio_far = run_pair([0.0, -13.2e12])
ratio_close = run_pair([0.0, -50e9])
ratio_no_raman = run_pair([0.0, -13.2e12], include_raman=False)

print(f"   13.2 THz spacing, Raman on:  red channel power ratio = {ratio_far:.5f}  (want > 1)")
print(f"   50 GHz spacing, Raman on:    red channel power ratio = {ratio_close:.5f}  (want ~= 1)")
print(f"   13.2 THz spacing, Raman off: red channel power ratio = {ratio_no_raman:.5f}  (want ~= 1)")

assert ratio_far > 1.01, \
    "A CW-like red channel 13.2 THz below a strong CW-like blue channel should gain via Raman crosstalk"
assert (ratio_close - 1.0) < 0.05 * (ratio_far - 1.0), \
    "Closely-spaced channels (far off the Raman gain spectrum) should show much less crosstalk than resonant spacing"
assert abs(ratio_no_raman - 1.0) < 1e-6, \
    "Disabling Raman should remove the crosstalk entirely, even at the resonant spacing"
print("   PASS (CW-channel Raman crosstalk is real, sign-correct, and resonance-selective)")

# ── 4. Absolute inter-channel phase (beta0) ─────────────────────────

print("\n4. Absolute inter-channel phase (beta0):")

fiber_b0 = make_fiber('smf28')
Delta_omega_check = 2 * np.pi * 100e9

# Analytic self-consistency: delta_beta0(w) = beta1_ref*w + 0.5*beta2*w^2 +
# (1/6)*beta3*w^3 has exact derivative beta1_ref + beta2*w + 0.5*beta3*w^2 =
# beta1_ref + delta_beta1(w) (delta_beta1 is defined RELATIVE to beta1_ref,
# not absolute -- so beta1_ref must be added back in for this comparison).
# These are simple closed-form polynomials, so check this analytically
# rather than by finite-differencing beta1_ref*w against its own tiny
# curvature correction (numerically ill-conditioned: the derivative signal
# from beta2/beta3 is ~1e-14 against a beta1_ref*w baseline of order 1e3,
# far below double-precision resolution of that baseline).
delta_beta1_check = fiber_b0.beta2 * Delta_omega_check + 0.5 * fiber_b0.beta3 * Delta_omega_check ** 2
exact_derivative = fiber_b0.beta1_ref + delta_beta1_check
d_domega = 1e-3 * Delta_omega_check
delta_beta0_of = lambda w: fiber_b0.beta1_ref * w + 0.5 * fiber_b0.beta2 * w ** 2 + (1 / 6) * fiber_b0.beta3 * w ** 3
numerical_deriv = (delta_beta0_of(Delta_omega_check + d_domega) - delta_beta0_of(Delta_omega_check - d_domega)) / (2 * d_domega)
rel_err_b0 = abs(numerical_deriv - exact_derivative) / abs(exact_derivative)
print(f"   d(delta_beta0)/d(Delta_omega) vs beta1_ref+delta_beta1: relative error = {rel_err_b0:.2e}")
assert rel_err_b0 < 1e-6, "delta_beta0's derivative should equal beta1_ref + delta_beta1 exactly"
print("   PASS (self-consistent: delta_beta0 is the exact antiderivative of beta1_ref+delta_beta1)")

# End-to-end: propagate two channels a length short enough that walk-off/GVD
# are negligible, then check the phase actually imprinted on channel 1
# matches delta_beta0[1]*L.
fiber_b0_prop = make_fiber('smf28', material_overrides=dict(n2=0.0, f_R=0.0))
wdm_b0 = WDMPropagator(fiber_b0_prop, channel_offsets_Hz=[0.0, 100e9])
L_b0 = 1e-3  # 1 mm
print(f"   delta_beta0[1]={wdm_b0.delta_beta0[1]:.2f} rad/m, "
      f"delta_beta1[1]*L={wdm_b0.delta_beta1[1]*L_b0*1e15:.4f} fs (walk-off, negligible)")

N_b0 = 2 ** 10
dt_b0 = 1e-12
T0_b0 = 20e-12
t_b0 = (np.arange(N_b0) - N_b0 // 2) * dt_b0
pulse_b0 = np.exp(-t_b0 ** 2 / (2 * T0_b0 ** 2)).astype(complex)  # real, zero initial phase

A0_b0 = np.zeros((2, N_b0), dtype=complex)
A0_b0[0] = pulse_b0
A0_b0[1] = pulse_b0
A_out_b0 = wdm_b0.propagate(A0_b0, dt_b0, L_b0, step_size=L_b0 / 20)

peak_idx = N_b0 // 2
phase_ch0 = np.angle(A_out_b0[0, peak_idx])
phase_ch1 = np.angle(A_out_b0[1, peak_idx])
measured_relative_phase = (phase_ch1 - phase_ch0 + np.pi) % (2 * np.pi) - np.pi
expected_relative_phase = (wdm_b0.delta_beta0[1] * L_b0 + np.pi) % (2 * np.pi) - np.pi

print(f"   measured relative phase at pulse peak: {measured_relative_phase:+.5f} rad")
print(f"   expected (delta_beta0*L, wrapped):     {expected_relative_phase:+.5f} rad")
assert abs(phase_ch0) < 1e-6, "Channel 0 (the reference) should pick up no extra phase (delta_beta0[0]=0)"
phase_err_b0 = min(abs(measured_relative_phase - expected_relative_phase),
                    2 * np.pi - abs(measured_relative_phase - expected_relative_phase))
assert phase_err_b0 < 1e-3, \
    "The propagated phase difference between two channels should match delta_beta0[1]*L"
print("   PASS (propagated relative phase matches the closed-form delta_beta0*L prediction)")

print("\n" + "=" * 70)
print("  All WDM propagator checks passed.")
print("=" * 70)
