"""Validation for fiber.receiver_leakage."""
import numpy as np

from fiber.multimode_fiber import make_multimode_fiber
from fiber.receiver_leakage import filter_leakage_photons, required_floor_dB

from fiber.constants import hbar


def check_10dB_halves_in_log():
    fiber = make_multimode_fiber('om3', lambda0=1552.524e-9, D=17.0, alpha_dB_km=0.3)
    omega = fiber.omega0
    n1 = filter_leakage_photons(1e-3, 5000.0, fiber, 1, 30.0, omega, 500e-12)
    n2 = filter_leakage_photons(1e-3, 5000.0, fiber, 1, 40.0, omega, 500e-12)
    ratio = n1 / n2
    assert abs(ratio - 10.0) < 1e-9, f"10dB floor step should give exactly 10x photons, got {ratio}"
    print(f"check 1 OK: +10dB floor -> 10x fewer leaked photons exactly (ratio={ratio:.6f})")


def check_attenuation_matches_fiber_alpha():
    fiber = make_multimode_fiber('om3', lambda0=1552.524e-9, D=17.0, alpha_dB_km=0.3)
    omega = fiber.omega0
    n_0km = filter_leakage_photons(1e-3, 0.0, fiber, 1, 40.0, omega, 500e-12)
    n_10km = filter_leakage_photons(1e-3, 10000.0, fiber, 1, 40.0, omega, 500e-12)
    expected_ratio = np.exp(fiber.alpha[1] * 10000.0)
    ratio = n_0km / n_10km
    err = abs(ratio - expected_ratio) / expected_ratio
    assert err < 1e-9, f"attenuation mismatch: {err:.2e}"
    print(f"check 2 OK: leakage attenuates exactly with fiber.alpha[bright_mode] (ratio={ratio:.4f})")


def check_required_floor_roundtrips():
    fiber = make_multimode_fiber('om3', lambda0=1552.524e-9, D=17.0, alpha_dB_km=0.3)
    omega = fiber.omega0
    P = 1e-3
    L = 5000.0
    target = 0.05
    floor = required_floor_dB(P, L, fiber, 1, omega, 500e-12, target)
    n = filter_leakage_photons(P, L, fiber, 1, floor, omega, 500e-12)
    err = abs(n - target) / target
    assert err < 1e-9, f"round-trip mismatch: {err:.2e}"
    print(f"check 3 OK: required_floor_dB round-trips through filter_leakage_photons "
          f"(floor={floor:.2f} dB -> {n:.4f} photons, target={target})")


if __name__ == '__main__':
    check_10dB_halves_in_log()
    check_attenuation_matches_fiber_alpha()
    check_required_floor_roundtrips()
    print("\nAll fiber.receiver_leakage checks passed.")
