"""Validation for fiber.hybrid_crosstalk.HybridCrosstalkPropagator.

Checks the module's central design claim -- that it correctly reduces to
the two mechanisms it was built by combining:

  1. mode_offset -> 0 (qkd_mode == bright_mode): a hybrid problem with no
     actual spatial separation is exactly a 2-channel single-mode WDM
     problem, so it must reproduce fiber.wdm_propagator.WDMPropagator's
     output exactly (same gamma, alpha, beta2/beta3, and channel offsets).

  2. Deterministic Raman crosstalk vanishes at zero channel separation
     (raman_gain_spectrum(material, 0, gamma) = 0 exactly, since Im(H_R)
     at Omega=0 is 0) -- checked directly on g_R_cross.
"""
import numpy as np

from core.dfb_laser import c
from fiber.multimode_fiber import make_multimode_fiber
from fiber.fiber_params import FiberParams
from fiber.wdm_propagator import WDMPropagator
from fiber.quantum_wdm import QuantumWDMPropagator
from fiber.hybrid_crosstalk import HybridCrosstalkPropagator


def _make_equivalent_single_mode_fiber(mmf, mode):
    """A FiberParams whose scalar parameters exactly match one mode group
    of an MultimodeFiberParams fiber, for the mode_offset=0 reduction check."""
    fp = FiberParams(lambda0=mmf.lambda0, material=mmf.material)
    fp.alpha = mmf.alpha[mode]
    fp.beta2 = mmf.beta2[mode]
    fp.beta3 = mmf.beta3[mode]
    fp.gamma = mmf.gamma_matrix[mode, mode]
    fp.omega0 = mmf.omega0
    return fp


def check_mode_offset_zero_matches_wdm():
    mmf = make_multimode_fiber('om3', lambda0=1550e-9, D=17.0, alpha_dB_km=0.3)
    mode = 0
    separation_Hz = 200e9

    N = 2 ** 10
    dt = 2e-12
    t = (np.arange(N) - N // 2) * dt
    T0 = 40e-12
    qkd_field = (np.sqrt(1e-9) / np.cosh(t / T0)).astype(complex)
    bright_field = (np.sqrt(1e-3) / np.cosh(t / (5 * T0))).astype(complex)

    hybrid = HybridCrosstalkPropagator(mmf, qkd_mode=mode, bright_mode=mode,
                                        channel_separation_Hz=separation_Hz,
                                        include_raman=True, noise=False)
    A_qkd_out, A_bright_out = hybrid.propagate(qkd_field, bright_field, dt, L=2000.0, step_size=20.0)

    fp = _make_equivalent_single_mode_fiber(mmf, mode)
    wdm = WDMPropagator(fp, channel_offsets_Hz=[0.0, separation_Hz], include_raman=True)
    A0 = np.zeros((2, N), dtype=complex)
    A0[0] = qkd_field
    A0[1] = bright_field
    A_wdm = wdm.propagate(A0, dt, L=2000.0, step_size=20.0)

    err_qkd = np.max(np.abs(A_qkd_out - A_wdm[0])) / np.max(np.abs(A_wdm[0]))
    err_bright = np.max(np.abs(A_bright_out - A_wdm[1])) / np.max(np.abs(A_wdm[1]))
    assert err_qkd < 1e-7, f"QKD-row mismatch vs WDMPropagator: {err_qkd:.3e}"
    assert err_bright < 1e-7, f"bright-row mismatch vs WDMPropagator: {err_bright:.3e}"
    print(f"check 1 OK: mode_offset=0 matches WDMPropagator exactly "
          f"(err_qkd={err_qkd:.2e}, err_bright={err_bright:.2e})")


def check_zero_separation_raman_crosstalk_vanishes():
    mmf = make_multimode_fiber('om3', lambda0=1550e-9, D=17.0, alpha_dB_km=0.3)
    hybrid = HybridCrosstalkPropagator(mmf, qkd_mode=0, bright_mode=1,
                                        channel_separation_Hz=0.0,
                                        include_raman=True, noise=False)
    assert abs(hybrid.g_R_cross) < 1e-30, f"g_R_cross should vanish at zero separation, got {hybrid.g_R_cross}"
    print(f"check 2 OK: g_R_cross={hybrid.g_R_cross:.3e} at zero channel separation")


def check_gamma_cross_matches_gamma_matrix():
    mmf = make_multimode_fiber('om3', lambda0=1550e-9, D=17.0, alpha_dB_km=0.3)
    hybrid = HybridCrosstalkPropagator(mmf, qkd_mode=0, bright_mode=1,
                                        channel_separation_Hz=200e9,
                                        include_raman=True, noise=False)
    assert hybrid.gamma_cross == mmf.gamma_matrix[0, 1]
    assert mmf.gamma_matrix[0, 1] < mmf.gamma_matrix[0, 0]  # off-diagonal overlap < 1
    print(f"check 3 OK: gamma_cross={hybrid.gamma_cross:.4e} matches gamma_matrix[0,1]")


def check_mode_offset_zero_noise_matches_quantum_wdm():
    """Ensemble-averaged noise energy landing in the (empty) QKD slot must
    statistically match QuantumWDMPropagator's inter-channel noise term
    when qkd_mode == bright_mode (no real spatial separation)."""
    mmf = make_multimode_fiber('om3', lambda0=1550e-9, D=17.0, alpha_dB_km=0.3)
    mode = 0
    separation_Hz = 200e9

    N = 2 ** 11
    dt = 2e-12
    t = (np.arange(N) - N // 2) * dt
    T0_cw = 2e-12
    P_bright_W = 1e-3
    qkd_field = np.zeros(N, dtype=complex)
    bright_field = (np.sqrt(P_bright_W) / np.cosh(np.clip(t / T0_cw, -700, 700))).astype(complex)

    L = 5000.0
    n_runs = 20

    energies_hybrid = []
    for seed in range(n_runs):
        hybrid = HybridCrosstalkPropagator(mmf, qkd_mode=mode, bright_mode=mode,
                                            channel_separation_Hz=separation_Hz,
                                            include_raman=True, noise=True, seed=seed)
        A_qkd_out, _ = hybrid.propagate(qkd_field, bright_field, dt, L=L, step_size=50.0)
        energies_hybrid.append(np.sum(np.abs(A_qkd_out) ** 2) * dt)

    fp = FiberParams(lambda0=mmf.lambda0, material=mmf.material)
    fp.alpha = mmf.alpha[mode]
    fp.beta2 = mmf.beta2[mode]
    fp.beta3 = mmf.beta3[mode]
    fp.gamma = mmf.gamma_matrix[mode, mode]
    fp.omega0 = mmf.omega0
    fp.beta1_ref = mmf.material.n_g / c

    energies_wdm = []
    for seed in range(n_runs):
        qwdm = QuantumWDMPropagator(fp, channel_offsets_Hz=[0.0, separation_Hz], seed=seed)
        A0 = np.zeros((2, N), dtype=complex)
        A0[0] = qkd_field
        A0[1] = bright_field
        A_out = qwdm.propagate(A0, dt, L=L, step_size=50.0)
        energies_wdm.append(np.sum(np.abs(A_out[0]) ** 2) * dt)

    m_h, m_w = np.mean(energies_hybrid), np.mean(energies_wdm)
    ratio = m_h / m_w
    assert 0.7 < ratio < 1.3, f"noise energy ratio {ratio:.3f} outside statistical tolerance"
    print(f"check 4 OK: mode_offset=0 noise floor matches QuantumWDMPropagator "
          f"(hybrid={m_h:.3e}, wdm={m_w:.3e}, ratio={ratio:.3f})")


if __name__ == '__main__':
    check_mode_offset_zero_matches_wdm()
    check_zero_separation_raman_crosstalk_vanishes()
    check_gamma_cross_matches_gamma_matrix()
    check_mode_offset_zero_noise_matches_quantum_wdm()
    print("\nAll fiber.hybrid_crosstalk checks passed.")
