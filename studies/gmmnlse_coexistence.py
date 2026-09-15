"""
Spontaneous Raman noise delivered to the phase-encoded QKD channel (DWDM Ch34, LP01)
by the classical 1 GbE channel (Ch30, LP11b, -30 dBm) in OM3, using the GMMNLSE solver
on computed GRIN modes. The photonic lantern's 15 dB launch extinction puts a copy of
the classical carrier into LP01, and its contribution is kept separate from the
intermodal one. Also recomputes the receiver-filter-leakage QBER maps shown in Chapter 4.

Channel frequencies follow the bench convention f = 190 THz + n x 0.1 THz (Ch30 = 193.0,
Ch32 = 193.2, Ch34 = 193.4 THz). The envelope is centred on Ch32, so both channels sit
exactly 200 GHz either side on frequency bins.

QBER model, as in the existing Chapter 4 figures: QBER = n / (2 (mu_rx + n)), with n the
noise photons per gate (Raman + filter leakage) and mu_rx = mu exp(-alpha L).

Writes images/gmmnlse_coexistence/data/{parameters.json, noise_vs_length.npz, qber.npz}.

    python studies/gmmnlse_coexistence.py
"""
import json
import time
from pathlib import Path

import numpy as np

from fiber.constants import c, hbar
from fiber.gmmnlse import GMMNLSE, TimeGrid
from fiber.grin_modes import DESIGNS, FibreModes
from fiber.raman_models import LinAgrawal

OUT = Path(__file__).resolve().parent.parent / 'images' / 'gmmnlse_coexistence' / 'data'


def channel_Hz(n):
    return 190e12 + n * 100e9


CENTRE_HZ = channel_Hz(32)
CLASSICAL_OFFSET_HZ = channel_Hz(30) - CENTRE_HZ
QKD_OFFSET_HZ = channel_Hz(34) - CENTRE_HZ

CLASSICAL_POWER_DBM = -30.0
LAUNCH_EXTINCTION_DB = 15.0
QKD_MU = 0.4
GATE_S = 500e-12
FILTER_NM = 0.05
ALPHA_DB_KM = 0.3
TEMPERATURE = 295.0
N2 = 2.6e-20
RAMAN = LinAgrawal()

LENGTHS_KM = [0.1, 0.5, 1, 2, 3, 5, 7, 8, 10, 12, 15, 17]
FLOORS_DB = np.linspace(20, 70, 101)

GRID = TimeGrid(n_points=4096, dt=1 / (4096 * 2e9), wavelength=c / CENTRE_HZ)
DZ = 50.0


def _dbm(p):
    return 10 ** (p / 10) * 1e-3


def run():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    modes = FibreModes(DESIGNS['om3'], GRID.wavelength, span_Hz=4.2e12)
    z_save = np.asarray(LENGTHS_KM, float) * 1e3
    k_qkd = int(round(-QKD_OFFSET_HZ / GRID.df)) % GRID.n_points   # Omega = -2 pi f_offset
    omega_q = GRID.omega[k_qkd]
    filter_Hz = c * FILTER_NM * 1e-9 / (c / channel_Hz(34)) ** 2
    to_photons = filter_Hz * GATE_S / (hbar * omega_q)

    P_c = _dbm(CLASSICAL_POWER_DBM)
    P_leak = P_c * 10 ** (-LAUNCH_EXTINCTION_DB / 10)

    def noise_photons(P_lp11b, P_lp01):
        gmm = GMMNLSE(modes, GRID, ['LP01', 'LP11b'], n2=N2, raman=RAMAN, alpha_dB_km=ALPHA_DB_KM,
                      noise='mean', noise_modes=['LP01'], temperature=TEMPERATURE)
        A0 = np.stack([GRID.cw(P_lp01, CLASSICAL_OFFSET_HZ), GRID.cw(P_lp11b, CLASSICAL_OFFSET_HZ)])
        res = gmm.propagate(A0, z_save[-1], dz=DZ, z_save=z_save)
        return res.noise_psd[:, 0, k_qkd] * to_photons

    via_overlap = noise_photons(P_c, 0.0)
    via_leak = noise_photons(0.0, P_leak)

    def qkd_phase(P_lp11b, P_lp01):
        gmm = GMMNLSE(modes, GRID, ['LP01', 'LP11b'], n2=N2, raman=RAMAN, alpha_dB_km=ALPHA_DB_KM)
        A0 = np.stack([GRID.cw(P_lp01, CLASSICAL_OFFSET_HZ) + GRID.cw(1e-9, QKD_OFFSET_HZ),
                       GRID.cw(P_lp11b, CLASSICAL_OFFSET_HZ)])
        res = gmm.propagate(A0, z_save[-1], dz=DZ, z_save=z_save)
        return np.angle(np.fft.fft(res.fields[:, 0], axis=-1)[:, k_qkd])

    xpm_phase = np.unwrap(qkd_phase(P_c, P_leak) - qkd_phase(0.0, 0.0))

    alpha = ALPHA_DB_KM / (10 * np.log10(np.e)) / 1e3
    raman_total = via_overlap + via_leak
    mu_rx = QKD_MU * np.exp(-alpha * z_save)
    leak_per_floor = (P_c * np.exp(-alpha * z_save)[None, :] * 10 ** (-FLOORS_DB[:, None] / 10)
                      * GATE_S / (hbar * 2 * np.pi * channel_Hz(30)))
    n = raman_total[None, :] + leak_per_floor
    qber = n / (2 * (mu_rx[None, :] + n))

    np.savez(OUT / 'noise_vs_length.npz', length_km=np.asarray(LENGTHS_KM, float),
             raman_via_overlap=via_overlap, raman_via_leak=via_leak, raman_total=raman_total,
             mu_rx=mu_rx, xpm_phase_rad=xpm_phase)
    np.savez(OUT / 'qber.npz', floors_dB=FLOORS_DB, length_km=np.asarray(LENGTHS_KM, float), qber=qber,
             raman_only_qber=raman_total / (2 * (mu_rx + raman_total)))

    params = {
        'channels_THz': {'classical_ch30': channel_Hz(30) / 1e12, 'qkd_ch34': channel_Hz(34) / 1e12,
                         'envelope_centre_ch32': CENTRE_HZ / 1e12},
        'classical_power_dBm': CLASSICAL_POWER_DBM, 'launch_extinction_dB': LAUNCH_EXTINCTION_DB,
        'qkd_mu': QKD_MU, 'gate_s': GATE_S, 'filter_nm': FILTER_NM, 'filter_Hz': filter_Hz,
        'alpha_dB_km': ALPHA_DB_KM, 'temperature_K': TEMPERATURE, 'n2_m2_W': N2, 'raman_model': RAMAN.name,
        'A_eff_lp01_um2': modes.effective_area(modes.index_of('LP01')) * 1e12,
        'S_01_11b_over_S_0000': float(modes.intensity_overlaps([modes.index_of('LP01')],
                                                               [modes.index_of('LP11b')])[0, 0]
                                      * modes.effective_area(modes.index_of('LP01'))),
        'max_raman_photons_per_gate': float(raman_total.max()),
        'max_raman_qber': float((raman_total / (2 * (mu_rx + raman_total))).max()),
        'max_xpm_phase_rad': float(np.abs(xpm_phase).max()),
        'grid': {'n_points': GRID.n_points, 'dt_s': GRID.dt, 'df_Hz': GRID.df}, 'dz_m': DZ,
        'runtime_s': time.time() - t0,
    }
    (OUT / 'parameters.json').write_text(json.dumps(params, indent=2))

    print(f"Raman photons/gate at QKD channel: max {raman_total.max():.2e} "
          f"(overlap {via_overlap.max():.2e}, launch leak {via_leak.max():.2e})")
    print(f"Raman-only QBER contribution: max {params['max_raman_qber']:.2e}")
    print(f"max XPM phase on QKD: {params['max_xpm_phase_rad']:.2e} rad")
    for floor in (30, 40, 50, 60):
        j = int(np.argmin(np.abs(FLOORS_DB - floor)))
        print(f"  filter floor {floor} dB: QBER {qber[j, 0]:.1%} (0.1 km) .. {qber[j, -1]:.1%} (17 km)")
    print(f"saved to {OUT}  [{params['runtime_s']:.1f} s]")


if __name__ == '__main__':
    run()
