"""
Multimode (OM1-OM5) graded-index fiber: intermodal dispersion (DMD) and
intermodal Raman scattering, using fiber/multimode_fiber.py and
fiber/multimode_propagator.py.

Demonstrates:

  1. Intermodal dispersion: an equal-power multi-mode launch broadens as
     the mode groups walk off from each other at different group
     velocities (differential mode delay) -- compared across OM1-OM5,
     reproducing the standard bandwidth-grade ordering (OM1 worst, OM5
     best) from a single underlying physical model.

  2. Intermodal Raman scattering: a strong, short pump pulse confined to
     mode group 0 spectrally pulls a weaker, temporally-overlapped probe
     pulse in a different mode group toward the red via the delayed
     Raman response, on top of (and distinguishable from) the
     instantaneous Kerr cross-phase-modulation.
"""
import numpy as np
import time as _time

from gsdfb.plotting import setup_plotting, save_fig
from fiber.multimode_fiber import make_multimode_fiber
from fiber.multimode_propagator import MultimodeFiberPropagator
from fiber.analysis import pulse_metrics, spectral_centroid


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    setup_plotting()

    print("=" * 70)
    print("  Multimode (OM1-OM5) Fiber: Intermodal Dispersion + Raman")
    print("=" * 70)

    # ── Part 1: intermodal dispersion (DMD) across OM1-OM5 ─────────────

    print("\n1. Intermodal dispersion (DMD) vs distance, OM1-OM5:")

    grades = ['om1', 'om2', 'om3', 'om4', 'om5']
    N = 2 ** 12
    dt = 2e-12
    T0 = 5e-12
    t = (np.arange(N) - N // 2) * dt
    pulse_shape = np.exp(-t ** 2 / (2 * T0 ** 2)).astype(complex)

    z_m = np.array([0, 50, 100, 200, 400, 800])

    fig1, ax1 = plt.subplots(figsize=(7, 5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(grades)))

    for color, grade in zip(colors, grades):
        fiber = make_multimode_fiber(grade, material_overrides=dict(n2=0.0, f_R=0.0))
        M = fiber.n_modes
        A0 = np.tile(pulse_shape, (M, 1)) / np.sqrt(M)
        prop = MultimodeFiberPropagator(fiber, include_raman=False)

        fwhm_ps = []
        t0 = _time.time()
        for L in z_m:
            if L == 0:
                P_total = np.abs(pulse_shape) ** 2
            else:
                A_out = prop.propagate(A0, dt, float(L), step_size=max(L / 50, 1.0))
                P_total = np.sum(np.abs(A_out) ** 2, axis=0)
            m = pulse_metrics(np.sqrt(P_total).astype(complex), dt)
            fwhm_ps.append(m['fwhm'] * 1e12)
        elapsed = _time.time() - t0

        ax1.plot(z_m, fwhm_ps, 'o-', color=color, lw=2, ms=5,
                  label=f'{grade.upper()} ({fiber.geometry.ofl_bandwidth_MHz_km:.0f} MHz·km)')
        print(f"   {grade}: FWHM {fwhm_ps[0]:.2f} -> {fwhm_ps[-1]:.2f} ps over {z_m[-1]} m "
              f"({elapsed:.1f}s, M={M} mode groups)")

    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('Total pulse FWHM (ps)')
    ax1.set_title('Intermodal Dispersion (DMD): Equal-Power Multi-Mode Launch')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    save_fig(fig1, 'images/multimode_fiber/intermodal_dispersion.png')
    print("   Saved: images/multimode_fiber/intermodal_dispersion.png")

    # ── Part 2: intermodal Raman scattering ────────────────────────────

    print("\n2. Intermodal Raman scattering: pump pulse in mode 0, probe in a different mode group:")

    N2 = 2 ** 12
    dt2 = 1e-14
    t2 = (np.arange(N2) - N2 // 2) * dt2

    T0_pump = 50e-15
    P_pump = 50.0
    pump_pulse = (np.sqrt(P_pump) / np.cosh(t2 / T0_pump)).astype(complex)

    T0_probe = 200e-15
    P_probe = 1e-3
    probe_pulse = (np.sqrt(P_probe) / np.cosh(t2 / T0_probe)).astype(complex)

    probe_mode = 5
    z_im_m = np.linspace(0, 0.1, 11)

    fig2, ax2 = plt.subplots(figsize=(7, 5))

    for include_raman, label, color in [(True, 'Kerr + Raman', 'C3'), (False, 'Kerr only', 'C0')]:
        fiber = make_multimode_fiber('om3', lambda0=1000e-9, D=0.0, beta3_material=0.0, alpha_dB_km=0.0)
        M = fiber.n_modes
        A0 = np.zeros((M, N2), dtype=complex)
        A0[0] = pump_pulse
        A0[probe_mode] = probe_pulse
        prop = MultimodeFiberPropagator(fiber, include_raman=include_raman)

        shifts_THz = []
        c0 = spectral_centroid(probe_pulse, dt2)
        for L in z_im_m:
            if L == 0:
                A_out_probe = probe_pulse
            else:
                A_out = prop.propagate(A0, dt2, float(L), step_size=max(L / 200, 1e-4))
                A_out_probe = A_out[probe_mode]
            # physical frequency = omega0 - Omega (see fiber.raman_response)
            shift = -(spectral_centroid(A_out_probe, dt2) - c0) / (2 * np.pi) * 1e-12
            shifts_THz.append(shift)

        ax2.plot(z_im_m * 100, shifts_THz, 'o-', color=color, lw=2, ms=5, label=label)
        print(f"   {label}: probe shift {shifts_THz[0]:+.2e} -> {shifts_THz[-1]:+.2e} THz "
              f"over {z_im_m[-1]*100:.0f} cm")

    ax2.set_xlabel('Distance (cm)')
    ax2.set_ylabel('Probe spectral shift (THz, physical)')
    ax2.set_title(f'Intermodal Cross-Modulation: Pump (mode 0) -> Probe (mode {probe_mode})')
    ax2.axhline(0, color='gray', lw=0.8, alpha=0.5)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    save_fig(fig2, 'images/multimode_fiber/intermodal_raman.png')
    print("   Saved: images/multimode_fiber/intermodal_raman.png")

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)
