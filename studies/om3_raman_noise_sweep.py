"""
Spontaneous Raman noise floor in OM3 multimode fiber at 1550 nm: a
(length, injected power) sweep using fiber/quantum_multimode.py.

Motivation
----------
A quasi-CW/classical signal (e.g. a bright reference or data channel)
launched into one spatial mode of a multimode fiber generates no
classical self-Raman distortion of itself (CW light has no spectral
content to seed the delayed-response mechanism -- see
fiber/multimode_propagator.py's docstring), but it DOES generate a real
spontaneous Raman noise floor, both within its own mode and (via the
overlap-weighted intermodal coupling) leaking into other, otherwise-
empty spatial modes -- exactly the physics relevant to a bright classical
channel sharing an OM3 link with a weak signal in a different spatial
mode (e.g. a QKD channel).

This sweeps injected power (-30 to +10 dBm, 5 dBm steps) across a set of
representative link lengths, launching a quasi-CW pulse purely into mode
group 0, and reports two noise metrics per (length, power) point,
ensemble-averaged over several stochastic noise realizations:

  intramodal noise power: spontaneous-Raman noise spectrally separated
      from the launched carrier within mode 0 itself (a band well away
      from DC, isolating the noise from the deterministic CW carrier).
  intermodal noise power: total energy that has appeared in every OTHER
      mode group (all of it is spontaneous noise, since nothing classical
      is launched there and the deterministic intermodal Raman term gives
      exactly zero for a CW driver).
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
    print("  OM3 Spontaneous Raman Noise: Length x Power Sweep (1550 nm)")
    print("=" * 70)

    lengths_m = np.array([100, 500, 1000, 2000, 3000, 5000, 7000, 8000,
                           10000, 12000, 15000, 17000], dtype=float)
    powers_dBm = np.arange(-30, 10.01, 5.0)
    n_runs = 8

    fiber = make_multimode_fiber('om3', lambda0=1550e-9, D=17.0, alpha_dB_km=0.3)
    M = fiber.n_modes
    print(f"\nOM3 @ 1550 nm: n_modes={M}, alpha={fiber.alpha_dB_km} dB/km, D={fiber.D} ps/nm/km")
    print(f"Lengths: {lengths_m/1e3} km")
    print(f"Powers: {powers_dBm} dBm  ({dbm_to_watts(powers_dBm)*1e3} mW)")
    print(f"Ensemble size per point: {n_runs}")

    N = 2 ** 10
    dt = 20e-15
    t = (np.arange(N) - N // 2) * dt
    T0_cw = 2e-12  # quasi-CW: broad envelope, negligible bandwidth

    Omega_grid = 2 * np.pi * np.fft.fftfreq(N, d=dt)
    Omega_max = Omega_grid.max()
    noise_band = (0.3 * Omega_max, 0.9 * Omega_max)  # away from the carrier (DC)

    intramodal_noise = np.zeros((len(lengths_m), len(powers_dBm)))
    intermodal_noise = np.zeros((len(lengths_m), len(powers_dBm)))

    total_t0 = _time.time()
    for i, L in enumerate(lengths_m):
        for j, P_dBm in enumerate(powers_dBm):
            P_W = dbm_to_watts(P_dBm)
            pulse = (np.sqrt(P_W) / np.cosh(t / T0_cw)).astype(complex)
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

    os.makedirs('images/om3_raman_noise', exist_ok=True)
    np.savez('images/om3_raman_noise/sweep_results.npz',
              lengths_m=lengths_m, powers_dBm=powers_dBm,
              intramodal_noise=intramodal_noise, intermodal_noise=intermodal_noise)
    print("Saved raw results: images/om3_raman_noise/sweep_results.npz")

    # ── Heatmaps ─────────────────────────────────────────────────────

    def plot_heatmap(data, title, fname):
        fig, ax = plt.subplots(figsize=(8, 6))
        floor = np.min(data[data > 0]) if np.any(data > 0) else 1e-300
        log_data = np.log10(np.clip(data, floor, None))
        im = ax.pcolormesh(powers_dBm, lengths_m / 1e3, log_data, shading='nearest', cmap='inferno')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('log10(noise energy, a.u.)')
        ax.set_xlabel('Injected power (dBm)')
        ax.set_ylabel('Fiber length (km)')
        ax.set_title(title)
        plt.tight_layout()
        save_fig(fig, fname)
        print(f"Saved: {fname}")

    plot_heatmap(intramodal_noise,
                 'Intramodal Spontaneous Raman Noise\n(mode 0, OM3 @ 1550 nm)',
                 'images/om3_raman_noise/intramodal_noise_heatmap.png')
    plot_heatmap(intermodal_noise,
                 'Intermodal Spontaneous Raman Noise\n(leakage into other mode groups, OM3 @ 1550 nm)',
                 'images/om3_raman_noise/intermodal_noise_heatmap.png')

    # ── Line plots: noise vs power, one curve per length ────────────

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(lengths_m)))
    for i, L in enumerate(lengths_m):
        axes2[0].semilogy(powers_dBm, intramodal_noise[i], 'o-', color=colors[i],
                           lw=1.5, ms=4, label=f'{L/1e3:.1f} km')
        axes2[1].semilogy(powers_dBm, np.clip(intermodal_noise[i], 1e-300, None), 'o-', color=colors[i],
                           lw=1.5, ms=4, label=f'{L/1e3:.1f} km')
    axes2[0].set_title('Intramodal Noise vs Injected Power')
    axes2[1].set_title('Intermodal Noise vs Injected Power')
    for ax in axes2:
        ax.set_xlabel('Injected power (dBm)')
        ax.set_ylabel('Noise energy (a.u.)')
        ax.grid(True, which='both', alpha=0.3)
    axes2[0].legend(fontsize=7, ncol=2, title='Length')
    plt.tight_layout()
    save_fig(fig2, 'images/om3_raman_noise/noise_vs_power_lines.png')
    print("Saved: images/om3_raman_noise/noise_vs_power_lines.png")

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)
