"""
Combined mode + wavelength crosstalk on a phase-encoded BB84 QKD signal:
a bright classical reference launched into OM3's FIRST EXCITED spatial
mode group (mode 1) on DWDM Channel 32 (192.90 THz, 1554.134 nm),
co-propagating with a weak coherent phase-encoded BB84 signal launched
into the FUNDAMENTAL mode group (mode 0) on DWDM Channel 34 (193.10 THz,
1552.524 nm, 200 GHz channel separation on the 100 GHz ITU grid) -- using
fiber/hybrid_crosstalk.py, the first propagator in this codebase to
combine spatial-mode and wavelength-channel diversity.

Both bright-signal cases already characterized for the co-channel/
co-mode noise floor (studies/om3_raman_noise_sweep.py: quasi-CW;
studies/om3_raman_noise_sweep_pulsed.py: 1 GHz/100 ps pulsed) are run
here at a fixed representative average power (0 dBm) across the same 12
representative OM3 lengths, reporting two QKD-relevant metrics:

  1. Spontaneous Raman noise landing in the QKD frame -- mean photon
     count integrated over an 800 ps window around the two BB84 time
     bins, ensemble-averaged over several stochastic noise realizations
     (QKD field launched empty, isolating exactly the noise the bright
     channel + fiber's own thermal bath add to that slot -- the same
     convention as the earlier CW/pulsed sweeps).

  2. XPM-induced DIFFERENTIAL PHASE ERROR between the QKD signal's two
     time bins -- the metric that actually matters for a phase-encoded
     protocol, since the receiver's interferometer measures exactly this
     relative phase. A perfectly CW bright signal has constant power, so
     both bins see IDENTICAL instantaneous XPM regardless of length --
     zero differential phase by construction. A PULSED bright signal
     does not: even launched symmetrically about the QKD frame center,
     accumulated group-velocity walk-off between the bright pulse (mode
     1, ch 32) and the QKD frame (mode 0, ch 34, the phase reference)
     shifts the bright pulse asymmetrically relative to the two bins as
     length grows, breaking that symmetry -- deterministic, run without
     noise. Converted to a QBER contribution via the standard
     phase-encoding formula e_phase = 0.5*(1 - cos(delta_phi)).

QKD launch: mean photon number mu=0.5 per pulse-pair (typical BB84
signal-state value), split equally between two 50 ps FWHM (sech) time
bins separated by 500 ps, zero phase difference at launch (so any
nonzero phase difference at the output is entirely crosstalk-induced).
"""
import os
import numpy as np
import time as _time

from gsdfb.plotting import setup_plotting, save_fig
from fiber.multimode_fiber import make_multimode_fiber
from fiber.hybrid_crosstalk import HybridCrosstalkPropagator

hbar = 1.0545718e-34  # J.s


def dbm_to_watts(dbm):
    return 10 ** (dbm / 10.0) * 1e-3


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    setup_plotting()

    print("=" * 70)
    print("  OM3 Hybrid Mode+Wavelength Crosstalk: Phase-Encoded BB84 QKD")
    print("  Bright: mode 1, DWDM Ch 32 (192.90 THz)")
    print("  QKD:    mode 0, DWDM Ch 34 (193.10 THz), 200 GHz separation")
    print("=" * 70)

    lengths_m = np.array([100, 500, 1000, 2000, 3000, 5000, 7000, 8000,
                           10000, 12000, 15000, 17000], dtype=float)
    CHANNEL_34_THZ = 193.10
    CHANNEL_32_THZ = 192.90
    SEPARATION_HZ = (CHANNEL_34_THZ - CHANNEL_32_THZ) * 1e12
    LAMBDA_QKD = 1552.524e-9

    BRIGHT_AVG_POWER_DBM = 0.0
    BRIGHT_AVG_POWER_W = dbm_to_watts(BRIGHT_AVG_POWER_DBM)
    REP_RATE = 1e9
    PULSE_FWHM = 100e-12
    DUTY_CYCLE = PULSE_FWHM * REP_RATE
    BRIGHT_PEAK_POWER_PULSED_W = BRIGHT_AVG_POWER_W / DUTY_CYCLE

    MU = 0.5                 # mean photon number per QKD pulse-pair
    BIN_SEP = 500e-12         # time-bin separation (s)
    BIN_FWHM = 50e-12         # each bin's sech FWHM (s)
    NOISE_WINDOW_HALF = 400e-12  # noise integration window half-width (s)
    N_NOISE_RUNS = 8

    fiber = make_multimode_fiber('om3', lambda0=LAMBDA_QKD, D=17.0, alpha_dB_km=0.3)
    print(f"\nOM3 @ QKD ref {LAMBDA_QKD*1e9:.3f} nm: n_modes={fiber.n_modes}, "
          f"alpha={fiber.alpha_dB_km} dB/km, D={fiber.D} ps/nm/km")
    print(f"Channel separation: {SEPARATION_HZ/1e9:.1f} GHz")
    print(f"gamma_matrix[0,1] = {fiber.gamma_matrix[0,1]:.4e} 1/W/m "
          f"(vs gamma_matrix[0,0] = {fiber.gamma_matrix[0,0]:.4e})")
    print(f"Bright average power: {BRIGHT_AVG_POWER_DBM} dBm "
          f"({BRIGHT_AVG_POWER_W*1e3:.3f} mW); pulsed peak = {BRIGHT_PEAK_POWER_PULSED_W*1e3:.3f} mW")
    print(f"QKD: mu={MU} photons/pulse-pair, bins {BIN_FWHM*1e12:.0f} ps FWHM, "
          f"{BIN_SEP*1e12:.0f} ps separation")

    N = 2 ** 12
    dt = 2e-12
    t = (np.arange(N) - N // 2) * dt

    omega_qkd = fiber.omega0
    E_photon = hbar * omega_qkd
    E_bin = 0.5 * MU * E_photon
    T0_bin = BIN_FWHM / 1.7627
    P0_bin = E_bin / (2 * T0_bin)

    t_bin1 = -BIN_SEP / 2
    t_bin2 = +BIN_SEP / 2
    qkd_field = (np.sqrt(P0_bin) / np.cosh((t - t_bin1) / T0_bin)
                 + np.sqrt(P0_bin) / np.cosh((t - t_bin2) / T0_bin)).astype(complex)

    T0_bright_pulsed = PULSE_FWHM / 1.7627
    bright_pulsed_field = (np.sqrt(BRIGHT_PEAK_POWER_PULSED_W)
                            / np.cosh(np.clip(t / T0_bright_pulsed, -700, 700))).astype(complex)
    bright_cw_field = np.full(N, np.sqrt(BRIGHT_AVG_POWER_W), dtype=complex)

    def extract_bin_phase(A_qkd_out, t_bin, T0):
        window = np.exp(-0.5 * ((t - t_bin) / (2 * T0)) ** 2)
        weighted = A_qkd_out * window
        return np.angle(np.sum(weighted))

    def noise_photon_count(fiber, qkd_mode, bright_mode, bright_field, L, seed0):
        counts = []
        for i in range(N_NOISE_RUNS):
            prop = HybridCrosstalkPropagator(fiber, qkd_mode=qkd_mode, bright_mode=bright_mode,
                                              channel_separation_Hz=SEPARATION_HZ,
                                              include_raman=True, noise=True, seed=seed0 + i)
            A_qkd_out, _ = prop.propagate(np.zeros(N, dtype=complex), bright_field, dt, L, step_size=50.0)
            mask = np.abs(t) < NOISE_WINDOW_HALF
            energy = np.sum(np.abs(A_qkd_out[mask]) ** 2) * dt
            counts.append(energy / E_photon)
        return float(np.mean(counts)), float(np.std(counts) / np.sqrt(N_NOISE_RUNS))

    results = {'cw': {'noise': [], 'noise_err': [], 'dphi': [], 'qber': []},
               'pulsed': {'noise': [], 'noise_err': [], 'dphi': [], 'qber': []}}

    total_t0 = _time.time()
    for i, L in enumerate(lengths_m):
        for case, bright_field in [('cw', bright_cw_field), ('pulsed', bright_pulsed_field)]:
            prop_det = HybridCrosstalkPropagator(fiber, qkd_mode=0, bright_mode=1,
                                                  channel_separation_Hz=SEPARATION_HZ,
                                                  include_raman=True, noise=False)
            A_qkd_out, _ = prop_det.propagate(qkd_field, bright_field, dt, L, step_size=50.0)
            phi1 = extract_bin_phase(A_qkd_out, t_bin1, T0_bin)
            phi2 = extract_bin_phase(A_qkd_out, t_bin2, T0_bin)
            dphi = np.angle(np.exp(1j * (phi2 - phi1)))  # wrap to [-pi, pi]
            qber_phase = 0.5 * (1 - np.cos(dphi))

            noise_mean, noise_err = noise_photon_count(fiber, 0, 1, bright_field, L, seed0=1000 * i)

            results[case]['dphi'].append(dphi)
            results[case]['qber'].append(qber_phase)
            results[case]['noise'].append(noise_mean)
            results[case]['noise_err'].append(noise_err)

        print(f"  L={L/1e3:5.1f} km done ({_time.time()-total_t0:.1f}s elapsed)")

    for case in results:
        for key in results[case]:
            results[case][key] = np.array(results[case][key])

    total_elapsed = _time.time() - total_t0
    print(f"\nTotal sweep time: {total_elapsed:.1f}s")

    os.makedirs('images/hybrid_bb84_crosstalk', exist_ok=True)
    np.savez('images/hybrid_bb84_crosstalk/results.npz',
              lengths_m=lengths_m,
              cw_noise=results['cw']['noise'], cw_noise_err=results['cw']['noise_err'],
              cw_dphi=results['cw']['dphi'], cw_qber=results['cw']['qber'],
              pulsed_noise=results['pulsed']['noise'], pulsed_noise_err=results['pulsed']['noise_err'],
              pulsed_dphi=results['pulsed']['dphi'], pulsed_qber=results['pulsed']['qber'],
              bright_avg_power_dbm=BRIGHT_AVG_POWER_DBM, separation_Hz=SEPARATION_HZ,
              mu=MU, bin_sep_s=BIN_SEP, bin_fwhm_s=BIN_FWHM)
    print("Saved: images/hybrid_bb84_crosstalk/results.npz")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].errorbar(lengths_m / 1e3, results['cw']['noise'], yerr=results['cw']['noise_err'],
                      fmt='o-', label='CW bright', lw=1.5, ms=5)
    axes[0].errorbar(lengths_m / 1e3, results['pulsed']['noise'], yerr=results['pulsed']['noise_err'],
                      fmt='s-', label='Pulsed (1 GHz) bright', lw=1.5, ms=5)
    axes[0].axhline(MU, color='gray', ls='--', lw=1, label=f'QKD signal (mu={MU})')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Fiber length (km)')
    axes[0].set_ylabel('Noise photons / QKD frame')
    axes[0].set_title(f'Spontaneous Raman Noise in QKD Slot\n(bright @ {BRIGHT_AVG_POWER_DBM} dBm avg)')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, which='both', alpha=0.3)

    axes[1].plot(lengths_m / 1e3, np.degrees(results['cw']['dphi']), 'o-', label='CW bright', lw=1.5, ms=5)
    axes[1].plot(lengths_m / 1e3, np.degrees(results['pulsed']['dphi']), 's-', label='Pulsed (1 GHz) bright', lw=1.5, ms=5)
    axes[1].set_xlabel('Fiber length (km)')
    axes[1].set_ylabel('Induced bin-to-bin phase error (deg)')
    axes[1].set_title('XPM-Induced Differential Phase')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(lengths_m / 1e3, results['cw']['qber'] * 100, 'o-', label='CW bright', lw=1.5, ms=5)
    axes[2].plot(lengths_m / 1e3, results['pulsed']['qber'] * 100, 's-', label='Pulsed (1 GHz) bright', lw=1.5, ms=5)
    axes[2].set_xlabel('Fiber length (km)')
    axes[2].set_ylabel('QBER contribution (%)')
    axes[2].set_title('Phase-Error QBER Contribution')
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, 'images/hybrid_bb84_crosstalk/summary.png')
    print("Saved: images/hybrid_bb84_crosstalk/summary.png")

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)
