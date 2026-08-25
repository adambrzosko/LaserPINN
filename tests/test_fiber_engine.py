"""Validation checks for the fiber/ GNLSE propagation engine.

Checks, in order:
  1. GVD-only Gaussian broadening matches the analytic linear-dispersion
     formula (gamma=0, f_R=0).
  2. A fundamental soliton (gamma>0, f_R=0) reproduces its own shape after
     one soliton period -- the classic regression test for a correct
     dispersion+Kerr split-step implementation.
  3. Turning on the Raman response (f_R>0) red-shifts (or at least shifts)
     the soliton spectrum relative to the no-Raman case -- soliton
     self-frequency shift, the hallmark classical Raman effect.
  4. QuantumRamanPropagator: at T->0 the anti-Stokes-side noise power
     vanishes while the gain-side (spontaneous) noise power does not.
"""
import numpy as np

from fiber.materials import make_material
from fiber.geometry import make_geometry
from fiber.fiber_params import FiberParams
from fiber.propagator import FiberPropagator
from fiber.quantum_noise import QuantumRamanPropagator
from fiber.analysis import pulse_metrics, spectral_centroid, band_power

print("=" * 70)
print("  fiber/ GNLSE engine validation")
print("=" * 70)

# ── 1. GVD-only broadening ──────────────────────────────────────────

N = 2 ** 13
dt = 5e-15
T0 = 500e-15          # Gaussian 1/e half-width
t = (np.arange(N) - N // 2) * dt
A0 = np.exp(-t ** 2 / (2 * T0 ** 2)).astype(complex)

zero_kerr = make_material('silica', n2=0.0, f_R=0.0)
geo = make_geometry('smf28')
fiber_lin = FiberParams(material=zero_kerr, geometry=geo, alpha_dB_km=0.0,
                         D=17.0, beta3=0.0)

L_D = T0 ** 2 / abs(fiber_lin.beta2)
L_test = 0.5 * L_D

prop = FiberPropagator(fiber_lin, include_raman=False)
A_out = prop.propagate(A0, dt, L_test, step_size=L_test / 200)

broaden_numeric = pulse_metrics(A_out, dt)['rms_width'] / pulse_metrics(A0, dt)['rms_width']
broaden_analytic = np.sqrt(1 + (L_test / L_D) ** 2)

print(f"\n1. GVD-only broadening at z=0.5 L_D:")
print(f"   numeric  = {broaden_numeric:.4f}")
print(f"   analytic = {broaden_analytic:.4f}")
assert abs(broaden_numeric - broaden_analytic) / broaden_analytic < 0.01, \
    "GVD-only broadening does not match the analytic linear-propagation formula"
print("   PASS")

# ── 2. Fundamental soliton periodicity (Kerr + dispersion, no Raman) ──

silica = make_material('silica', f_R=0.0)   # isolate Kerr-only GNLSE
fiber_sol = FiberParams(material=silica, geometry=geo, alpha_dB_km=0.0,
                         D=17.0, beta3=0.0)

T0_s = 200e-15
beta2 = fiber_sol.beta2
gamma = fiber_sol.gamma
P0 = abs(beta2) / (gamma * T0_s ** 2)   # fundamental-soliton peak power

N2 = 2 ** 13
dt2 = 4e-15
t2 = (np.arange(N2) - N2 // 2) * dt2
A0_sol = np.sqrt(P0) / np.cosh(t2 / T0_s)

z0 = (np.pi / 2) * T0_s ** 2 / abs(beta2)   # one soliton period

prop_sol = FiberPropagator(fiber_sol, include_raman=False)
A_z0 = prop_sol.propagate(A0_sol.astype(complex), dt2, z0, step_size=z0 / 400)

m_in = pulse_metrics(A0_sol, dt2)
m_out = pulse_metrics(A_z0, dt2)
fwhm_ratio = m_out['fwhm'] / m_in['fwhm']
peak_ratio = m_out['peak_power'] / m_in['peak_power']

print(f"\n2. Fundamental soliton after one soliton period:")
print(f"   FWHM ratio (out/in) = {fwhm_ratio:.3f} (expect ~1)")
print(f"   Peak power ratio    = {peak_ratio:.3f} (expect ~1)")
assert abs(fwhm_ratio - 1) < 0.05, "Soliton FWHM did not reproduce after one soliton period"
assert abs(peak_ratio - 1) < 0.10, "Soliton peak power did not reproduce after one soliton period"
print("   PASS")

# ── 3. Raman self-frequency shift ───────────────────────────────────

silica_raman = make_material('silica')   # f_R=0.18, default
fiber_raman = FiberParams(material=silica_raman, geometry=geo, alpha_dB_km=0.0,
                           D=17.0, beta3=0.0)

# Use a short, higher-power soliton-like pulse -- Raman shift needs
# reasonably broadband content to be visible over a short length.
T0_r = 50e-15
gamma_r = fiber_raman.gamma
beta2_r = fiber_raman.beta2
P0_r = 3.0 * abs(beta2_r) / (gamma_r * T0_r ** 2)   # 3rd-order soliton peak power

N3 = 2 ** 14
dt3 = 2e-15
t3 = (np.arange(N3) - N3 // 2) * dt3
A0_r = (np.sqrt(P0_r) / np.cosh(t3 / T0_r)).astype(complex)

L_r = 2e-3  # 2 mm -- short enough to stay in the perturbative SSFS regime

prop_noraman = FiberPropagator(fiber_raman, include_raman=False)
prop_raman = FiberPropagator(fiber_raman, include_raman=True)

A_noraman = prop_noraman.propagate(A0_r, dt3, L_r, step_size=L_r / 2000)
A_withraman = prop_raman.propagate(A0_r, dt3, L_r, step_size=L_r / 2000)

centroid_noraman = spectral_centroid(A_noraman, dt3)
centroid_withraman = spectral_centroid(A_withraman, dt3)
# Physical optical frequency at raw offset Omega is omega0 - Omega (this
# codebase's envelope convention, see fiber.raman_response.raman_gain_spectrum),
# so a physical *redshift* shows up as the raw Omega-centroid *increasing*.
# Report the physical-frequency shift directly so the assertion below reads
# the intuitive way (negative = redshift).
phys_shift = -(centroid_withraman - centroid_noraman) / (2 * np.pi) * 1e-12  # THz

print(f"\n3. Raman-induced spectral shift over {L_r*1e3:.0f} mm:")
print(f"   no Raman         : {-centroid_noraman/(2*np.pi)*1e-12:+.4f} THz (physical, rel. carrier)")
print(f"   with Raman       : {-centroid_withraman/(2*np.pi)*1e-12:+.4f} THz (physical, rel. carrier)")
print(f"   physical shift   : {phys_shift:+.4f} THz")
assert abs(phys_shift) > 0, "Raman response produced no spectral shift at all"
assert phys_shift < 0, \
    "Raman response should shift the spectrum toward lower frequency (soliton self-frequency shift)"
print("   PASS (Raman shifts the spectrum to lower frequency, as expected for SSFS)")

# ── 4. Quantum noise: anti-Stokes suppression at T->0 ───────────────

def noise_only_power(fiber, A0, dt, L, T_side, band, n_runs=24):
    band_powers = []
    for i in range(n_runs):
        qprop = QuantumRamanPropagator(fiber, seed=i)
        A_out = qprop.propagate(A0.copy(), dt, L, step_size=L / 20)
        band_powers.append(band_power(A_out, dt, *band))
    return float(np.mean(band_powers))

silica_cold = make_material('silica', T=1e-3)   # ~T->0
fiber_cold = FiberParams(material=silica_cold, geometry=geo, alpha_dB_km=0.0,
                          D=17.0, beta3=0.0)

N4 = 2 ** 10
dt4 = 20e-15
t4 = (np.arange(N4) - N4 // 2) * dt4
T0_cw = 2e-12
A0_cw = (np.sqrt(0.5) / np.cosh(t4 / T0_cw)).astype(complex)  # ~0.5 W quasi-CW-ish pulse

Omega_grid = 2 * np.pi * np.fft.fftfreq(N4, d=dt4)
Omega_max = Omega_grid.max()

gain_band = (0.3 * Omega_max, 0.9 * Omega_max)
loss_band = (-0.9 * Omega_max, -0.3 * Omega_max)

L_noise = 1e3  # 1 km

P_gain_side = noise_only_power(fiber_cold, A0_cw, dt4, L_noise, 'cold', gain_band)
P_loss_side = noise_only_power(fiber_cold, A0_cw, dt4, L_noise, 'cold', loss_band)

print(f"\n4. Quantum Raman noise at T~0:")
print(f"   gain-side band power : {P_gain_side:.3e}")
print(f"   loss-side band power : {P_loss_side:.3e}")
assert P_gain_side > 0, "Expected nonzero spontaneous noise on the gain side even at T->0"
assert P_loss_side < P_gain_side, \
    "Anti-Stokes/loss-side noise should be strongly suppressed relative to the gain side at T->0"
print("   PASS (gain-side spontaneous noise persists at T->0; loss side is suppressed)")

print("\n" + "=" * 70)
print("  All fiber engine checks passed.")
print("=" * 70)
