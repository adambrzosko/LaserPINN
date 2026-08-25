"""
WDM cross-phase modulation / Raman crosstalk, and stimulated Brillouin
scattering, using fiber/wdm_propagator.py and fiber/brillouin.py.

Demonstrates:

  1. Cross-phase modulation (XPM): a strong pump pulse imprints a phase
     (and hence a frequency chirp) on a weak co-propagating probe channel,
     at exactly the standard factor of 2 relative to its own SPM.

  2. Inter-channel Raman scattering ("Raman tilt"): across a WDM comb
     spanning several THz, shorter-wavelength channels pump longer-
     wavelength ones via the Raman gain spectrum evaluated at each
     channel pair's separation -- the classic power-tilt impairment in
     wideband WDM systems.

  3. Stimulated Brillouin scattering: the threshold "knee" in reflectivity
     and pump transmission vs input power, and the narrow (~tens of MHz)
     resonance the effect lives on.
"""
import numpy as np
import time as _time

from gsdfb.plotting import setup_plotting, save_fig
from fiber.fiber_params import make_fiber
from fiber.wdm_propagator import WDMPropagator
from fiber.brillouin import BrillouinPropagator, sbs_threshold_power


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    setup_plotting()

    print("=" * 70)
    print("  WDM (XPM + Raman Crosstalk) and Stimulated Brillouin Scattering")
    print("=" * 70)

    # ── Part 1: XPM-induced chirp on a probe channel ────────────────────

    print("\n1. Cross-phase modulation: chirp imprinted on a probe by a pump pulse:")

    fiber_xpm = make_fiber('hnlf', D=0.0, beta3=0.0)  # zero walk-off: isolate XPM cleanly
    N = 2 ** 12
    dt = 5e-15
    T0_pump = 300e-15
    t = (np.arange(N) - N // 2) * dt
    P_pump = 1.0
    pump_pulse = (np.sqrt(P_pump) / np.cosh(t / T0_pump)).astype(complex)
    probe_cw = np.ones(N, dtype=complex) * 1e-3

    L_xpm = 50.0
    wdm_xpm = WDMPropagator(fiber_xpm, channel_offsets_Hz=[0.0, 100e9])
    A0 = np.zeros((2, N), dtype=complex)
    A0[0] = pump_pulse
    A0[1] = probe_cw
    A_out = wdm_xpm.propagate(A0, dt, L_xpm, step_size=L_xpm / 400)

    xpm_phase = np.unwrap(np.angle(A_out[1] / probe_cw))
    xpm_chirp_GHz = -np.gradient(xpm_phase, dt) / (2 * np.pi) * 1e-9

    fig1, axes1 = plt.subplots(1, 2, figsize=(13, 5))
    axes1[0].plot(t * 1e15, np.abs(pump_pulse) ** 2 * 1e3, 'C3', lw=2)
    axes1[0].set_xlabel('Time (fs)')
    axes1[0].set_ylabel('Pump power (mW)')
    axes1[0].set_title(f'Pump Pulse (T0={T0_pump*1e15:.0f} fs)')
    axes1[0].grid(True, alpha=0.3)

    axes1[1].plot(t * 1e15, xpm_chirp_GHz, 'C0', lw=2)
    axes1[1].set_xlabel('Time (fs)')
    axes1[1].set_ylabel('XPM-induced probe chirp (GHz)')
    axes1[1].set_title(f'Probe Frequency Chirp from XPM ({L_xpm:.0f} m)')
    axes1[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig1, 'images/wdm_brillouin/xpm_chirp.png')
    print(f"   Peak XPM chirp: {np.max(np.abs(xpm_chirp_GHz)):.2f} GHz")
    print("   Saved: images/wdm_brillouin/xpm_chirp.png")

    # ── Part 2: Raman crosstalk / tilt across a WDM comb ────────────────

    print("\n2. Inter-channel Raman crosstalk (Raman tilt) across a WDM comb:")

    fiber_tilt = make_fiber('smf28', alpha_dB_km=0.0)  # isolate Raman tilt from ordinary loss
    n_ch = 9
    offsets_THz = np.linspace(-6, 6, n_ch)  # spans +-6 THz, comparable to the ~13 THz Raman shift
    offsets_Hz = offsets_THz * 1e12

    N2 = 2 ** 10
    dt2 = 50e-15
    P_ch = 0.05  # 50 mW per channel, CW-like
    fields = np.ones((n_ch, N2), dtype=complex) * np.sqrt(P_ch)

    L_tilt = 20e3  # 20 km
    wdm_tilt = WDMPropagator(fiber_tilt, channel_offsets_Hz=offsets_Hz)
    A_out_tilt = wdm_tilt.propagate(fields, dt2, L_tilt, step_size=50.0)
    P_out_ch = np.mean(np.abs(A_out_tilt) ** 2, axis=1)
    tilt_dB = 10 * np.log10(P_out_ch / P_ch)

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(offsets_THz, tilt_dB, 'o-', color='C1', lw=2, ms=7)
    ax2.axhline(0, color='gray', lw=0.8, alpha=0.5)
    ax2.set_xlabel('Channel offset from comb center (THz)')
    ax2.set_ylabel('Power change (dB)')
    ax2.set_title(f'Raman Tilt Across a {n_ch}-Channel WDM Comb ({L_tilt/1e3:.0f} km)')
    ax2.grid(True, alpha=0.3)
    save_fig(fig2, 'images/wdm_brillouin/raman_tilt.png')
    for f_THz, d in zip(offsets_THz, tilt_dB):
        print(f"   {f_THz:+.1f} THz: {d:+.3f} dB")
    print("   Saved: images/wdm_brillouin/raman_tilt.png")

    # ── Part 3: SBS threshold curve ──────────────────────────────────────

    print("\n3. Stimulated Brillouin scattering threshold curve:")

    fiber_sbs = make_fiber('smf28')
    L_sbs = 20e3
    P_th = sbs_threshold_power(fiber_sbs, L_sbs)
    print(f"   Analytic SBS threshold at L={L_sbs/1e3:.0f} km: {P_th*1e3:.2f} mW")

    bp = BrillouinPropagator(fiber_sbs)
    power_ratios = np.logspace(np.log10(0.02), np.log10(4.0), 20)
    reflectivity = []
    transmission = []
    t0 = _time.time()
    for r in power_ratios:
        P_in = r * P_th
        _, Pp, Ps = bp.solve(P_in, L_sbs)
        reflectivity.append(Ps[0] / P_in)
        transmission.append(Pp[-1] / P_in)
    print(f"   Solved {len(power_ratios)} boundary-value points in {_time.time()-t0:.1f}s")

    fig3, ax3 = plt.subplots(figsize=(7, 5))
    ax3.semilogy(power_ratios, reflectivity, 'o-', color='C3', lw=2, ms=5, label='Reflectivity (backscatter)')
    ax3.semilogy(power_ratios, transmission, 's-', color='C0', lw=2, ms=5, label='Pump transmission')
    ax3.axvline(1.0, color='gray', ls='--', lw=1.5, label='Analytic threshold')
    ax3.set_xlabel('$P_{in} / P_{th}$')
    ax3.set_ylabel('Fraction of input power')
    ax3.set_title(f'SBS Threshold Knee (SMF-28, {L_sbs/1e3:.0f} km)')
    ax3.legend(fontsize=9)
    ax3.grid(True, which='both', alpha=0.3)
    save_fig(fig3, 'images/wdm_brillouin/sbs_threshold.png')
    print("   Saved: images/wdm_brillouin/sbs_threshold.png")

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)
