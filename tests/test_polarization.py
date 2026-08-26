"""Validation checks for PolarizationPropagator (PMD, polarization-dependent
XPM/Raman).

Checks, in order:
  1. Single-component-limit consistency: launching purely into the x
     component with no PMD reproduces fiber.propagator.FiberPropagator
     exactly, and the y component stays exactly zero.
  2. Exact power conservation: with no loss/nonlinearity, strong PMD
     conserves total energy (summed over both components, integrated
     over time) to floating-point precision -- both the random axis
     rotation and the DGD phase are unitary/Parseval-preserving by
     construction.
  3. Cross-polarization XPM factor: a weak orthogonally-polarized probe
     picks up exactly gamma*(2/3)*P_pump*L of induced phase from a
     strong co-propagating pump -- the standard 2/3 ratio (vs. the
     factor of 2 for co-polarized/different-channel XPM elsewhere in
     this codebase).
  4. PMD accumulation statistics: using Jones Matrix Eigenanalysis (JME)
     on the composed per-step rotation+DGD operators -- the rigorous way
     to check the underlying stochastic PROCESS, independent of how any
     particular pulse responds to it -- the RMS accumulated DGD across
     many realizations matches the target D_PMD*sqrt(L) scaling to
     within statistical sampling error, checked at three lengths
     spanning a 16x range.
"""
import numpy as np

from fiber.fiber_params import make_fiber
from fiber.propagator import FiberPropagator
from fiber.polarization import PolarizationPropagator

print("=" * 70)
print("  Polarization (PMD, polarization-dependent XPM/Raman) validation")
print("=" * 70)

# ── 1. Single-component-limit consistency ───────────────────────────

print("\n1. Single-component-limit consistency (x only, no PMD vs FiberPropagator):")

fiber = make_fiber('hnlf')
N = 2 ** 11
dt = 5e-15
T0 = 100e-15
t = (np.arange(N) - N // 2) * dt
P0 = 5.0
A0 = (np.sqrt(P0) / np.cosh(t / T0)).astype(complex)
L_test = 200.0

pol_prop = PolarizationPropagator(fiber, D_PMD=0.0, seed=0)
A_out_pol = pol_prop.propagate(A0, dt, L_test, step_size=L_test / 400)

sm_prop = FiberPropagator(fiber, include_raman=True)
A_out_sm = sm_prop.propagate(A0, dt, L_test, step_size=L_test / 400)

max_diff = np.max(np.abs(A_out_pol[0] - A_out_sm)) / np.max(np.abs(A_out_sm))
y_leakage = np.max(np.abs(A_out_pol[1]))
print(f"   max relative difference (x vs FiberPropagator): {max_diff:.2e}")
print(f"   max |A_y| (should be exactly 0):                {y_leakage:.2e}")
assert max_diff < 1e-12, "x-only, no-PMD PolarizationPropagator should exactly match FiberPropagator"
assert y_leakage == 0.0, "With no PMD and no y launch, the y component should stay exactly zero"
print("   PASS")

# ── 2. Exact power conservation ──────────────────────────────────────

print("\n2. Exact power conservation under strong PMD (no loss/nonlinearity):")

fiber_lossless = make_fiber('smf28', alpha_dB_km=0.0, material_overrides=dict(n2=0.0, f_R=0.0))
N2 = 2 ** 10
dt2 = 5e-12
T0_2 = 15e-12
t2 = (np.arange(N2) - N2 // 2) * dt2
pulse2 = np.exp(-t2 ** 2 / (2 * T0_2 ** 2)).astype(complex)
A0_2 = np.zeros((2, N2), dtype=complex)
A0_2[0] = pulse2
energy_in = np.sum(np.abs(A0_2) ** 2) * dt2

prop_pmd = PolarizationPropagator(fiber_lossless, D_PMD=0.5, seed=0)
A_out_2 = prop_pmd.propagate(A0_2, dt2, 5e3, step_size=50.0)
energy_out = np.sum(np.abs(A_out_2) ** 2) * dt2
rel_diff = abs(energy_out - energy_in) / energy_in
print(f"   energy in={energy_in:.6e}, out={energy_out:.6e}, relative difference={rel_diff:.2e}")
assert rel_diff < 1e-10, "PMD (rotation + DGD phase) should conserve total energy exactly"
print("   PASS")

# ── 3. Cross-polarization XPM factor ────────────────────────────────

print("\n3. Cross-polarization XPM factor (expect 2/3, vs. 2 for co-polarized/WDM XPM):")

fiber_xpm = make_fiber('hnlf', material_overrides=dict(f_R=0.0))
N3 = 2 ** 12
dt3 = 5e-15
T0_pump = 300e-15
t3 = (np.arange(N3) - N3 // 2) * dt3
P_pump = 1.0
pump_pulse = (np.sqrt(P_pump) / np.cosh(t3 / T0_pump)).astype(complex)
probe_cw = np.ones(N3, dtype=complex) * 1e-3
L_xpm = 30.0

prop_xpm = PolarizationPropagator(fiber_xpm, D_PMD=0.0, seed=0)

A0_xpm = np.zeros((2, N3), dtype=complex)
A0_xpm[0] = pump_pulse
A0_xpm[1] = probe_cw
A_out_xpm = prop_xpm.propagate(A0_xpm, dt3, L_xpm, step_size=L_xpm / 400)

A0_alone = np.zeros((2, N3), dtype=complex)
A0_alone[1] = probe_cw
A_out_alone = prop_xpm.propagate(A0_alone, dt3, L_xpm, step_size=L_xpm / 400)

center = N3 // 2
phase_with_pump = np.angle(A_out_xpm[1, center] / probe_cw[center])
phase_alone = np.angle(A_out_alone[1, center] / probe_cw[center])
induced_phase = phase_with_pump - phase_alone
expected_phase = fiber_xpm.gamma * (2.0 / 3.0) * P_pump * L_xpm
rel_err = abs(induced_phase - expected_phase) / expected_phase

print(f"   expected XPM phase (gamma*(2/3)*P_pump*L) = {expected_phase:.5f} rad")
print(f"   measured induced phase                     = {induced_phase:.5f} rad")
print(f"   relative error = {rel_err:.2%}")
assert rel_err < 0.05, "Cross-polarization XPM phase should match gamma*(2/3)*P_pump*L to a few percent"
print("   PASS")

# ── 4. PMD accumulation statistics (Jones Matrix Eigenanalysis) ────

print("\n4. PMD accumulation statistics (RMS DGD vs D_PMD*sqrt(L), via JME):")


def total_jones_matrix(prop, omega, L, dz, seed):
    """Compose the SAME per-step rotation+DGD operators the propagator's
    own _pmd_step uses, to isolate and directly test the stochastic
    process itself (not how a particular pulse responds to it)."""
    prop.rng = np.random.default_rng(seed)
    n_steps = int(round(L / dz))
    M = np.eye(2, dtype=complex)
    for _ in range(n_steps):
        R = prop._random_su2()
        dgd = prop.D_PMD_SI * np.sqrt(dz)
        B = np.diag([np.exp(-1j * (dgd / 2) * omega), np.exp(1j * (dgd / 2) * omega)])
        M = B @ R @ M
    return M


def rms_dgd(prop, L, dz, domega, n_realizations):
    dgds = []
    for seed in range(n_realizations):
        M1 = total_jones_matrix(prop, 0.0, L, dz, seed)
        M2 = total_jones_matrix(prop, domega, L, dz, seed)
        R_between = M2 @ np.linalg.inv(M1)
        thetas = np.angle(np.linalg.eigvals(R_between))
        dgds.append(abs(thetas[0] - thetas[1]) / domega)
    return float(np.sqrt(np.mean(np.array(dgds) ** 2)))


fiber_pmd = make_fiber('smf28')
D_PMD_ps_sqrtkm = 0.2
prop_stat = PolarizationPropagator(fiber_pmd, D_PMD=D_PMD_ps_sqrtkm, seed=0)
dz_stat = 100.0
domega_stat = 1e9
n_realizations = 40

ratios = []
for L in [5e3, 20e3, 80e3]:
    rms = rms_dgd(prop_stat, L, dz_stat, domega_stat, n_realizations)
    expected = D_PMD_ps_sqrtkm * 1e-12 / np.sqrt(1e3) * np.sqrt(L)
    ratio = rms / expected
    ratios.append(ratio)
    print(f"   L={L/1e3:5.0f} km: RMS_DGD={rms*1e12:.4f} ps, expected={expected*1e12:.4f} ps, ratio={ratio:.3f}")

assert all(0.6 < r < 1.6 for r in ratios), \
    "RMS DGD should match the D_PMD*sqrt(L) scaling to within statistical sampling error"
print("   PASS (sqrt(L)-scaled DGD accumulation confirmed across a 16x range in L)")

print("\n" + "=" * 70)
print("  All polarization checks passed.")
print("=" * 70)
