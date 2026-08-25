"""
Nonlinear fiber Raman scattering: classical soliton self-frequency shift
and quantum spontaneous-Raman noise, using the fiber/ package.

Demonstrates the two capabilities fiber/ adds beyond the plain-Kerr NLSE
used by studies/fiber_propagation.py:

  1. Classical Raman: soliton self-frequency shift (SSFS) -- the
     well-known continuous redshift of a soliton's spectrum from
     intrapulse Raman scattering -- compared across fiber geometries
     (SMF-28, HNLF, small-core PCF).

  2. Quantum Raman noise: the spontaneous-Raman noise floor added by
     QuantumRamanPropagator, and its Stokes/anti-Stokes asymmetry as a
     function of temperature -- the physics behind classical-channel
     Raman crosstalk into a co-propagating quantum channel (relevant to
     this suite's QKD source-characterisation focus).
"""
import numpy as np
import time as _time

from gsdfb.plotting import setup_plotting, save_fig
from fiber.materials import make_material
from fiber.fiber_params import make_fiber, FiberParams
from fiber.propagator import FiberPropagator
from fiber.quantum_noise import QuantumRamanPropagator
from fiber.analysis import spectral_centroid, band_power


def fundamental_soliton(fiber, T0, N, dt):
    """Build a fundamental (N=1) soliton envelope for the given fiber/T0."""
    P0 = abs(fiber.beta2) / (fiber.gamma * T0 ** 2)
    t = (np.arange(N) - N // 2) * dt
    return (np.sqrt(P0) / np.cosh(t / T0)).astype(complex), P0


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    setup_plotting()

    print("=" * 70)
    print("  Nonlinear Fiber Raman Scattering (classical + quantum)")
    print("=" * 70)

    # ── Part 1: soliton self-frequency shift across fiber types ───────

    # Pulse duration per fiber, chosen so the fundamental-soliton power
    # (P0 = |beta2|/(gamma*T0^2)) stays in a realistic few-hundred-mW to
    # few-W range for each fiber's dispersion/nonlinearity -- SMF-28's much
    # larger |beta2| would otherwise demand a MW-scale pulse at the same
    # short T0 as HNLF/PCF, pushing it into non-perturbative soliton
    # dynamics rather than the clean SSFS this plot is meant to illustrate.
    fiber_T0 = {'smf28': 300e-15, 'hnlf': 50e-15, 'pcf_supercontinuum': 50e-15}
    N = 2 ** 14
    dt = 2e-15

    z_km = np.linspace(0, 2.0, 15)  # 0-2 km

    print("\n1. Soliton self-frequency shift vs distance:")
    fig1, ax1 = plt.subplots(figsize=(7, 5))

    for name in fiber_T0:
        fiber = make_fiber(name)
        T0 = fiber_T0[name]
        A0, P0 = fundamental_soliton(fiber, T0, N, dt)
        prop = FiberPropagator(fiber, include_raman=True)

        shifts_THz = []
        t0 = _time.time()
        for L_km in z_km:
            L_m = L_km * 1e3
            if L_m == 0:
                A_out = A0
            else:
                A_out = prop.propagate(A0, dt, L_m, step_size=max(L_m / 500, 1e-3))
            # physical frequency = omega0 - Omega (see fiber.raman_response)
            shift = -spectral_centroid(A_out, dt) / (2 * np.pi) * 1e-12
            shifts_THz.append(shift)
        elapsed = _time.time() - t0

        ax1.plot(z_km, shifts_THz, 'o-', lw=2, ms=4,
                 label=f'{name} (T0={T0*1e15:.0f} fs, P0={P0*1e3:.0f} mW)')
        print(f"   {name:22s} (T0={T0*1e15:.0f} fs, P0={P0*1e3:.0f} mW): "
              f"{shifts_THz[-1]:+.3f} THz at {z_km[-1]:.1f} km ({elapsed:.1f}s)")

    ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel('Spectral shift (THz, physical, rel. carrier)')
    ax1.set_title('Soliton Self-Frequency Shift (fundamental soliton per fiber)')
    ax1.axhline(0, color='gray', lw=0.8, alpha=0.5)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    save_fig(fig1, 'images/raman_fiber/soliton_self_frequency_shift.png')
    print("   Saved: images/raman_fiber/soliton_self_frequency_shift.png")

    # ── Part 2: quantum Raman noise -- Stokes/anti-Stokes vs temperature ──

    print("\n2. Spontaneous-Raman noise asymmetry vs temperature:")

    hnlf_base = make_fiber('hnlf')
    N2 = 2 ** 10
    dt2 = 20e-15
    T0_cw = 2e-12
    t2 = (np.arange(N2) - N2 // 2) * dt2
    A0_cw = (np.sqrt(0.5) / np.cosh(t2 / T0_cw)).astype(complex)  # quasi-CW, ~0.5 W

    Omega_grid = 2 * np.pi * np.fft.fftfreq(N2, d=dt2)
    Omega_max = Omega_grid.max()
    gain_band = (0.3 * Omega_max, 0.9 * Omega_max)    # Stokes (physical, see fiber.raman_response)
    loss_band = (-0.9 * Omega_max, -0.3 * Omega_max)  # anti-Stokes

    L_noise_km = 1.0
    n_runs = 16
    temperatures = [4.0, 77.0, 200.0, 300.0, 450.0]

    stokes_power = []
    antistokes_power = []

    for T in temperatures:
        fiber_T = FiberParams(
            material=make_material('silica', T=T),
            geometry=hnlf_base.geometry,
            alpha_dB_km=hnlf_base.alpha_dB_km, D=hnlf_base.D, beta3=hnlf_base.beta3)

        stokes_runs, anti_runs = [], []
        for i in range(n_runs):
            qprop = QuantumRamanPropagator(fiber_T, seed=i)
            A_out = qprop.propagate(A0_cw.copy(), dt2, L_noise_km * 1e3, step_size=50.0)
            stokes_runs.append(band_power(A_out, dt2, *gain_band))
            anti_runs.append(band_power(A_out, dt2, *loss_band))

        stokes_power.append(np.mean(stokes_runs))
        antistokes_power.append(np.mean(anti_runs))
        print(f"   T={T:6.1f} K:  Stokes={stokes_power[-1]:.3e}  "
              f"anti-Stokes={antistokes_power[-1]:.3e}  "
              f"ratio={antistokes_power[-1]/stokes_power[-1]:.3f}")

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(temperatures, stokes_power, 'o-', color='C0', lw=2, ms=6, label='Stokes band')
    ax2.plot(temperatures, antistokes_power, 's-', color='C3', lw=2, ms=6, label='anti-Stokes band')
    ax2.set_xlabel('Temperature (K)')
    ax2.set_ylabel('Mean spontaneous-Raman band power (a.u.)')
    ax2.set_title(f'Spontaneous Raman Noise vs Temperature\nHNLF, {L_noise_km:.0f} km, {n_runs} realizations')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    save_fig(fig2, 'images/raman_fiber/quantum_noise_vs_temperature.png')
    print("   Saved: images/raman_fiber/quantum_noise_vs_temperature.png")

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)
