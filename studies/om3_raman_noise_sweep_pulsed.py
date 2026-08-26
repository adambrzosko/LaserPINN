"""
Spontaneous Raman noise floor in OM3 multimode fiber at 1550 nm, PULSED
case: a 1 GHz-clocked pulse train (100 ps FWHM, 10% duty cycle) instead
of the quasi-CW signal in studies/om3_raman_noise_sweep.py.

Motivation
----------
A real classical/reference channel sharing a multimode fiber with a QKD
channel is more realistically a CLOCKED pulse train (matching the QKD
system's own repetition rate, e.g. for synchronization) than an
idealised quasi-CW signal. This sweeps the SAME (length, injected
average power) grid as the quasi-CW study, but launches a single
representative pulse (100 ps FWHM) at PEAK power = average power /
duty cycle, since spontaneous-Raman noise here is driven by local PEAK
power (see fiber/quantum_multimode.py) -- so the same nominal average
dBm now drives a materially higher noise floor than the quasi-CW case.

A 100 ps pulse is still ~3000x longer than the material's Raman
correlation time (tau2=32 fs), so the classical (deterministic) self-
Raman spectral shift stays negligible even at the highest power/length
in this grid (~30 Hz, vs THz-scale shifts for genuinely short pulses --
checked directly, not just asserted, before committing to this sweep);
only the noise floor is materially different from the quasi-CW case.
"""
import os
import numpy as np
import time as _time

from gsdfb.plotting import setup_plotting, save_fig
from fiber.multimode_fiber import make_multimode_fiber
from fiber.quantum_multimode import QuantumMultimodePropagator
from fiber.analysis import band_power


def dbm_to_watts(dbm):
    return 10 ** (dbm / 10.0) * 1e-3


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    setup_plotting()

    print("=" * 70)
    print("  OM3 Spontaneous Raman Noise: Pulsed (1 GHz) Sweep (1550 nm)")
    print("=" * 70)

    lengths_m = np.array([100, 500, 1000, 2000, 3000, 5000, 7000, 8000,
                           10000, 12000, 15000, 17000], dtype=float)
    powers_dBm = np.arange(-30, 10.01, 5.0)  # average power
    n_runs = 8

    REP_RATE = 1e9         # 1 GHz
    PULSE_FWHM = 100e-12   # 100 ps
    DUTY_CYCLE = PULSE_FWHM * REP_RATE  # ~0.1

    fiber = make_multimode_fiber('om3', lambda0=1550e-9, D=17.0, alpha_dB_km=0.3)
    M = fiber.n_modes
    print(f"\nOM3 @ 1550 nm: n_modes={M}, alpha={fiber.alpha_dB_km} dB/km, D={fiber.D} ps/nm/km")
    print(f"Pulse: {PULSE_FWHM*1e12:.0f} ps FWHM @ {REP_RATE/1e9:.0f} GHz -> duty cycle {DUTY_CYCLE:.2%}")
    print(f"Lengths: {lengths_m/1e3} km")
    print(f"Average powers: {powers_dBm} dBm (peak = average / duty cycle)")
    print(f"Ensemble size per point: {n_runs}")

    N = 2 ** 11
    dt = 2e-12
    t = (np.arange(N) - N // 2) * dt
    T0 = PULSE_FWHM / 1.7627  # sech FWHM -> T0

    Omega_grid = 2 * np.pi * np.fft.fftfreq(N, d=dt)
    Omega_max = Omega_grid.max()
    noise_band = (0.3 * Omega_max, 0.9 * Omega_max)

    intramodal_noise = np.zeros((len(lengths_m), len(powers_dBm)))
    intermodal_noise = np.zeros((len(lengths_m), len(powers_dBm)))

    total_t0 = _time.time()
    for i, L in enumerate(lengths_m):
        for j, P_dBm in enumerate(powers_dBm):
            P_avg_W = dbm_to_watts(P_dBm)
            P_peak_W = P_avg_W / DUTY_CYCLE
            pulse = (np.sqrt(P_peak_W) / np.cosh(t / T0)).astype(complex)
            A0 = np.zeros((M, N), dtype=complex)
            A0[0] = pulse

            intra_vals, inter_vals = [], []
            for seed in range(n_runs):
                prop = QuantumMultimodePropagator(fiber, seed=1000 * i + 10 * j + seed)
                A_out = prop.propagate(A0, dt, L, step_size=50.0)
                intra_vals.append(band_power(A_out[0], dt, *noise_band))
                inter_vals.append(np.sum(np.abs(A_out[1:]) ** 2) * dt)

            intramodal_noise[i, j] = np.mean(intra_vals)
            intermodal_noise[i, j] = np.mean(inter_vals)

        print(f"  L={L/1e3:5.1f} km done ({_time.time()-total_t0:.1f}s elapsed)")

    total_elapsed = _time.time() - total_t0
    print(f"\nTotal sweep time: {total_elapsed:.1f}s "
          f"({len(lengths_m)*len(powers_dBm)*n_runs} propagations)")

    os.makedirs('images/om3_raman_noise_pulsed', exist_ok=True)
    np.savez('images/om3_raman_noise_pulsed/sweep_results.npz',
              lengths_m=lengths_m, powers_dBm=powers_dBm,
              intramodal_noise=intramodal_noise, intermodal_noise=intermodal_noise,
              pulse_fwhm_s=PULSE_FWHM, rep_rate_Hz=REP_RATE, duty_cycle=DUTY_CYCLE)
    print("Saved raw results: images/om3_raman_noise_pulsed/sweep_results.npz")

    def plot_heatmap(data, title, fname):
        fig, ax = plt.subplots(figsize=(8, 6))
        floor = np.min(data[data > 0]) if np.any(data > 0) else 1e-300
        log_data = np.log10(np.clip(data, floor, None))
        im = ax.pcolormesh(powers_dBm, lengths_m / 1e3, log_data, shading='nearest', cmap='inferno')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('log10(noise energy, a.u.)')
        ax.set_xlabel('Average injected power (dBm)')
        ax.set_ylabel('Fiber length (km)')
        ax.set_title(title)
        plt.tight_layout()
        save_fig(fig, fname)
        print(f"Saved: {fname}")

    plot_heatmap(intramodal_noise,
                 f'Intramodal Spontaneous Raman Noise\n(mode 0, OM3 @ 1550 nm, {PULSE_FWHM*1e12:.0f} ps @ {REP_RATE/1e9:.0f} GHz)',
                 'images/om3_raman_noise_pulsed/intramodal_noise_heatmap.png')
    plot_heatmap(intermodal_noise,
                 f'Intermodal Spontaneous Raman Noise\n(leakage into other mode groups, {PULSE_FWHM*1e12:.0f} ps @ {REP_RATE/1e9:.0f} GHz)',
                 'images/om3_raman_noise_pulsed/intermodal_noise_heatmap.png')

    # ── Comparison with the quasi-CW case ───────────────────────────

    cw_data = np.load('images/om3_raman_noise/sweep_results.npz')
    cw_intra = cw_data['intramodal_noise']
    ratio_intra = intramodal_noise / cw_intra

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    im = ax3.pcolormesh(powers_dBm, lengths_m / 1e3, ratio_intra, shading='nearest', cmap='viridis')
    cbar = fig3.colorbar(im, ax=ax3)
    cbar.set_label('Pulsed / quasi-CW noise ratio')
    ax3.set_xlabel('Average injected power (dBm)')
    ax3.set_ylabel('Fiber length (km)')
    ax3.set_title(f'Pulsed ({PULSE_FWHM*1e12:.0f} ps @ {REP_RATE/1e9:.0f} GHz) vs Quasi-CW\nIntramodal Noise Ratio, Same Average Power')
    plt.tight_layout()
    save_fig(fig3, 'images/om3_raman_noise_pulsed/pulsed_vs_cw_ratio.png')
    print("Saved: images/om3_raman_noise_pulsed/pulsed_vs_cw_ratio.png")
    print(f"\nMean pulsed/CW ratio: {np.mean(ratio_intra):.2f} "
          f"(expected ~1/duty_cycle = {1/DUTY_CYCLE:.2f} in the noise-dominated regime)")

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)
