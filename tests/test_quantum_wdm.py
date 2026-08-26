"""Validation checks for QuantumWDMPropagator (spontaneous Raman noise
between WDM channels).

Checks, in order:
  1. Single-channel-limit consistency: with only one channel, the intra-
     channel noise arrays exactly match QuantumRamanPropagator's, and the
     inter-channel noise gain matrix is identically zero (no other
     channel to receive/emit spontaneous noise).
  2. Inter-channel noise asymmetry at T~0: a strong channel spontaneously
     seeds noise in a *weak, initially-empty* channel on the Raman gain
     side (~13.2 THz below it) but not on the loss side -- reproducing,
     for inter-channel noise, the same Stokes/anti-Stokes asymmetry
     already validated for intra-channel noise in test_fiber_engine.py.
  3. Thermal activation: raising the temperature turns on measurable
     noise on the loss side too (a real phonon is now available to
     absorb), confirming the T~0 result above isn't just a zero-power
     coincidence.
"""
import numpy as np

from fiber.fiber_params import make_fiber
from fiber.quantum_wdm import QuantumWDMPropagator
from fiber.quantum_noise import QuantumRamanPropagator

print("=" * 70)
print("  QuantumWDMPropagator validation (inter-channel spontaneous Raman)")
print("=" * 70)

# ── 1. Single-channel-limit consistency ─────────────────────────────

print("\n1. Single-channel-limit consistency (vs QuantumRamanPropagator):")

fiber_sm = make_fiber('smf28')
N = 2 ** 10
dt = 20e-15
Omega = 2 * np.pi * np.fft.fftfreq(N, d=dt)

qwdm = QuantumWDMPropagator(fiber_sm, channel_offsets_Hz=[0.0], seed=0)
qsm = QuantumRamanPropagator(fiber_sm, seed=0)
qwdm._prepare(Omega, dt)
qsm._prepare(Omega, dt)

noise_gain_match = np.allclose(qwdm._intra_noise_gain, qsm._noise_gain)
omega_abs_match = np.allclose(qwdm._intra_omega_abs[0], qsm._omega_abs)
inter_is_zero = np.allclose(qwdm._inter_noise_gain, 0.0)

print(f"   intra-channel noise_gain matches QuantumRamanPropagator: {noise_gain_match}")
print(f"   intra-channel omega_abs matches QuantumRamanPropagator:  {omega_abs_match}")
print(f"   inter-channel noise_gain is identically zero (no peer):  {inter_is_zero}")
assert noise_gain_match and omega_abs_match, \
    "With one channel, intra-channel noise should reduce exactly to QuantumRamanPropagator's"
assert inter_is_zero, "With no other channel, inter-channel noise gain must be exactly zero"
print("   PASS")

# ── 2 & 3. Inter-channel noise: Stokes/anti-Stokes asymmetry + thermal activation ──

print("\n2. Inter-channel spontaneous noise asymmetry (T~0) and thermal activation:")

N2 = 2 ** 10
dt2 = 20e-15
t2 = (np.arange(N2) - N2 // 2) * dt2
T0_pump = 2e-12
P_pump = 0.5
pump_pulse = (np.sqrt(P_pump) / np.cosh(t2 / T0_pump)).astype(complex)
empty = np.zeros(N2, dtype=complex)

L_test = 1e3  # 1 km
n_runs = 20

# From tests/test_wdm_propagator.py's already-validated sign convention:
# with channel 0 at offset 0 and channel 1 at offset -13.2 THz, channel 1
# sits on the Raman GAIN side (it gets amplified by channel 0 classically);
# +13.2 THz is the LOSS side.
gain_side_offsets = [0.0, -13.2e12]
loss_side_offsets = [0.0, +13.2e12]


def peer_channel_noise_energy(fiber, offsets, n_runs):
    energies = []
    for seed in range(n_runs):
        A0 = np.zeros((2, N2), dtype=complex)
        A0[0] = pump_pulse
        A0[1] = empty
        prop = QuantumWDMPropagator(fiber, channel_offsets_Hz=offsets, seed=seed)
        A_out = prop.propagate(A0, dt2, L_test, step_size=50.0)
        energies.append(np.sum(np.abs(A_out[1]) ** 2))
    return float(np.mean(energies))


fiber_cold = make_fiber('smf28', alpha_dB_km=0.0, material_overrides=dict(T=1e-3))
gain_noise_cold = peer_channel_noise_energy(fiber_cold, gain_side_offsets, n_runs)
loss_noise_cold = peer_channel_noise_energy(fiber_cold, loss_side_offsets, n_runs)

print(f"   T~0: gain-side peer-channel noise energy = {gain_noise_cold:.3e}")
print(f"   T~0: loss-side peer-channel noise energy = {loss_noise_cold:.3e}")
assert gain_noise_cold > 0, \
    "Spontaneous inter-channel Raman noise should persist on the gain side even at T->0"
assert loss_noise_cold == 0.0, \
    "Inter-channel noise on the loss side should vanish exactly at T->0 (no phonons to absorb)"

fiber_hot = make_fiber('smf28', alpha_dB_km=0.0, material_overrides=dict(T=500.0))
loss_noise_hot = peer_channel_noise_energy(fiber_hot, loss_side_offsets, n_runs)
print(f"   T=500K: loss-side peer-channel noise energy = {loss_noise_hot:.3e}")
assert loss_noise_hot > 0, \
    "Raising the temperature should activate measurable loss-side inter-channel noise"

print("   PASS (gain-side spontaneous noise persists at T->0; loss-side vanishes there "
      "and switches on with temperature)")

print("\n" + "=" * 70)
print("  All QuantumWDMPropagator checks passed.")
print("=" * 70)
