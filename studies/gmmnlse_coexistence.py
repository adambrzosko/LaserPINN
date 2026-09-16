"""
Spontaneous Raman noise delivered to the phase-encoded QKD channel (DWDM Ch34, LP01)
by a classical channel co-propagating in LP11b of OM3, using the GMMNLSE solver on
computed GRIN modes. Two classical configurations:

    sfp_ch30     1 GbE SFP on Ch30 (193.0 THz), -30 dBm in the fibre (original run)
    laser_1547   CW laser at 1547.0 nm (193.79 THz), 2.6 mW, filtered to 3 nm at the
                 transmitter (current run)

The photonic lantern's 15 dB launch extinction also puts a copy of the classical carrier
into LP01; that contribution is kept separate from the intermodal one. Launch powers are
treated as power in the fibre, which is an upper bound if part of the quoted launch is
lost in the input lantern. The receiver-filter-leakage QBER map is recomputed for the
original configuration.

Frequencies follow the bench convention f = 190 THz + n x 0.1 THz. The envelope is
centred on Ch34 and both classical carriers sit on exact frequency bins.

QBER model, as in the Chapter 4 figures: QBER = n / (2 (mu_rx + n)), with n the noise
photons per gate and mu_rx = mu exp(-alpha L). Receiver loss and detector efficiency act
on signal and noise alike and cancel.

Writes images/gmmnlse_coexistence/data/{parameters.json, noise_vs_length.npz, qber.npz}.

    python studies/gmmnlse_coexistence.py
"""
import json
import time
from pathlib import Path

import numpy as np

from fiber.constants import c, hbar
from fiber.gmmnlse import GMMNLSE, ModeCoupling, TimeGrid
from fiber.grin_modes import DESIGNS, FibreModes
from fiber.raman_models import LinAgrawal
from fiber.receiver_leakage import leaked_photons_per_gate

OUT = Path(__file__).resolve().parent.parent / 'images' / 'gmmnlse_coexistence' / 'data'


def channel_Hz(n):
    return 190e12 + n * 100e9


QKD_HZ = channel_Hz(34)
CASES = {
    'sfp_ch30': {'label': '1 GbE SFP, Ch30, -30 dBm', 'offset_Hz': channel_Hz(30) - QKD_HZ, 'power_W': 1e-6},
    # 193.79 THz = 1547.01 nm, the bin nearest the 1547 nm laser
    'laser_1547': {'label': 'CW laser, 1547 nm, 2.6 mW', 'offset_Hz': 390e9, 'power_W': 2.6e-3},
}

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

# Distributed random linear mode coupling (bends, splices, connectors), which moves
# classical power into the QKD channel's own mode along the span, where it scatters with
# the full same-mode coefficient. Parametrised by the measurable quantity -- the
# LP11b -> LP01 crosstalk accumulated at COUPLING_REF_KM -- rather than by kappa, which
# has no datasheet value. COUPLING_LC is the assumed perturbation correlation length.
COUPLING_CASE = 'laser_1547'
COUPLING_LC = 1e-3
COUPLING_XT_DB = [-25.0, -20.0, -15.0]
COUPLING_REF_KM = 10.0
COUPLING_SEEDS = 8

GRID = TimeGrid(n_points=4096, dt=1 / (4096 * 2e9), wavelength=c / QKD_HZ)
DZ = 50.0


def run():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    modes = FibreModes(DESIGNS['om3'], GRID.wavelength, span_Hz=4.2e12)
    z_save = np.asarray(LENGTHS_KM, float) * 1e3
    alpha = ALPHA_DB_KM / (10 * np.log10(np.e)) / 1e3
    filter_Hz = c * FILTER_NM * 1e-9 / GRID.wavelength ** 2
    to_photons = filter_Hz * GATE_S / (hbar * GRID.omega0)
    mu_rx = QKD_MU * np.exp(-alpha * z_save)

    def noise_photons(P_lp11b, P_lp01, offset):
        gmm = GMMNLSE(modes, GRID, ['LP01', 'LP11b'], n2=N2, raman=RAMAN, alpha_dB_km=ALPHA_DB_KM,
                      noise='mean', noise_modes=['LP01'], temperature=TEMPERATURE)
        A0 = np.stack([GRID.cw(P_lp01, offset), GRID.cw(P_lp11b, offset)])
        return gmm.propagate(A0, z_save[-1], dz=DZ, z_save=z_save).noise_psd[:, 0, 0] * to_photons

    def qkd_phase(P_lp11b, P_lp01, offset):
        gmm = GMMNLSE(modes, GRID, ['LP01', 'LP11b'], n2=N2, raman=RAMAN, alpha_dB_km=ALPHA_DB_KM)
        A0 = np.stack([GRID.cw(P_lp01, offset) + GRID.cw(1e-9), GRID.cw(P_lp11b, offset)])
        fields = gmm.propagate(A0, z_save[-1], dz=DZ, z_save=z_save).fields[:, 0]
        return np.angle(np.fft.fft(fields, axis=-1)[:, 0])

    arrays = {'length_km': np.asarray(LENGTHS_KM, float), 'mu_rx': mu_rx,
              'case_keys': np.array(list(CASES)), 'case_labels': np.array([v['label'] for v in CASES.values()])}
    case_params = {}
    for key, case in CASES.items():
        P_c, offset = case['power_W'], case['offset_Hz']
        P_leak = P_c * 10 ** (-LAUNCH_EXTINCTION_DB / 10)
        via_overlap = noise_photons(P_c, 0.0, offset)
        via_leak = noise_photons(0.0, P_leak, offset)
        total = via_overlap + via_leak
        xpm = np.unwrap(qkd_phase(P_c, P_leak, offset) - qkd_phase(0.0, 0.0, offset))
        raman_qber = total / (2 * (mu_rx + total))
        arrays.update({f'{key}_raman_via_overlap': via_overlap, f'{key}_raman_via_leak': via_leak,
                       f'{key}_raman_total': total, f'{key}_xpm_phase_rad': xpm,
                       f'{key}_raman_only_qber': raman_qber})
        case_params[key] = {
            **case, 'wavelength_nm': c / (QKD_HZ + offset) * 1e9, 'stokes_side': bool(offset > 0),
            'max_raman_photons_per_gate': float(total.max()), 'max_raman_qber': float(raman_qber.max()),
            'max_xpm_phase_rad': float(np.abs(xpm).max()),
            'raman_via_leak_fraction': float(via_leak.max() / total.max()),
        }

    # --- sensitivity to distributed linear mode coupling (laser case) ---------------
    i01, i11b = modes.index_of('LP01'), modes.index_of('LP11b')
    dbeta = abs(modes.beta0[i01] - modes.beta0[i11b])
    w_cross = 1.0 / (1.0 + (dbeta * COUPLING_LC) ** 2)
    case = CASES[COUPLING_CASE]
    P_leak = case['power_W'] * 10 ** (-LAUNCH_EXTINCTION_DB / 10)
    coupled = []
    for xt_dB in COUPLING_XT_DB:
        kappa = np.sqrt(10 ** (xt_dB / 10) / (w_cross * COUPLING_REF_KM * 1e3))
        runs = []
        for seed in range(COUPLING_SEEDS):
            gmm = GMMNLSE(modes, GRID, ['LP01', 'LP11b'], pumps=['LP01', 'LP11b'], n2=N2,
                          raman=RAMAN, alpha_dB_km=ALPHA_DB_KM, noise='mean',
                          noise_modes=['LP01'], temperature=TEMPERATURE,
                          mode_coupling=ModeCoupling(kappa=kappa, correlation_length=COUPLING_LC),
                          seed=seed)
            A0 = np.stack([GRID.cw(P_leak, case['offset_Hz']),
                           GRID.cw(case['power_W'], case['offset_Hz'])])
            runs.append(gmm.propagate(A0, z_save[-1], dz=DZ, z_save=z_save).noise_psd[:, 0, 0])
        coupled.append(np.mean(runs, axis=0) * to_photons)
    coupled = np.asarray(coupled)
    coupled_qber = coupled / (2 * (mu_rx[None, :] + coupled))
    arrays.update({'coupling_xt_dB': np.asarray(COUPLING_XT_DB, float),
                   'coupling_raman_total': coupled, 'coupling_raman_qber': coupled_qber})

    sfp = CASES['sfp_ch30']
    sfp_total = arrays['sfp_ch30_raman_total']
    leak_per_floor = leaked_photons_per_gate(
        sfp['power_W'] * np.exp(-alpha * z_save)[None, :], FLOORS_DB[:, None],
        2 * np.pi * (QKD_HZ + sfp['offset_Hz']), GATE_S)
    n = sfp_total[None, :] + leak_per_floor
    np.savez(OUT / 'noise_vs_length.npz', **arrays)
    np.savez(OUT / 'qber.npz', floors_dB=FLOORS_DB, length_km=np.asarray(LENGTHS_KM, float),
             qber=n / (2 * (mu_rx[None, :] + n)), raman_only_qber=arrays['sfp_ch30_raman_only_qber'])

    params = {
        'qkd_channel': {'channel': 34, 'frequency_THz': QKD_HZ / 1e12, 'wavelength_nm': GRID.wavelength * 1e9},
        'cases': case_params, 'launch_extinction_dB': LAUNCH_EXTINCTION_DB, 'qkd_mu': QKD_MU,
        'gate_s': GATE_S, 'filter_nm': FILTER_NM, 'filter_Hz': filter_Hz, 'alpha_dB_km': ALPHA_DB_KM,
        'temperature_K': TEMPERATURE, 'n2_m2_W': N2, 'raman_model': RAMAN.name,
        'A_eff_lp01_um2': modes.effective_area(modes.index_of('LP01')) * 1e12,
        'S_01_11b_over_S_0000': float(modes.intensity_overlaps([modes.index_of('LP01')],
                                                               [modes.index_of('LP11b')])[0, 0]
                                      * modes.effective_area(modes.index_of('LP01'))),
        'grid': {'n_points': GRID.n_points, 'dt_s': GRID.dt, 'df_Hz': GRID.df}, 'dz_m': DZ,
        'mode_coupling': {
            'case': COUPLING_CASE, 'correlation_length_m': COUPLING_LC,
            'crosstalk_dB_at_ref': COUPLING_XT_DB, 'reference_length_km': COUPLING_REF_KM,
            'seeds': COUPLING_SEEDS, 'lorentzian_weight': float(w_cross),
            'dbeta0_rad_m': float(dbeta),
            'max_raman_photons_per_gate': [float(x) for x in coupled.max(axis=1)],
            'max_raman_qber': [float(x) for x in coupled_qber.max(axis=1)],
        },
        'runtime_s': time.time() - t0,
    }
    (OUT / 'parameters.json').write_text(json.dumps(params, indent=2))

    for key, p in case_params.items():
        print(f"{p['label']} ({p['wavelength_nm']:.2f} nm, {'Stokes' if p['stokes_side'] else 'anti-Stokes'} side): "
              f"Raman {p['max_raman_photons_per_gate']:.2e} photons/gate, QBER {p['max_raman_qber']:.2e}, "
              f"XPM {p['max_xpm_phase_rad']:.1e} rad, leak share {p['raman_via_leak_fraction']:.1%}")
    for xt_dB, n, q in zip(COUPLING_XT_DB, coupled.max(axis=1), coupled_qber.max(axis=1)):
        print(f"  + distributed coupling at {xt_dB:g} dB crosstalk/{COUPLING_REF_KM:g} km: "
              f"{n:.2e} photons/gate, QBER {q:.2e}")
    print(f"saved to {OUT}  [{params['runtime_s']:.1f} s]")


if __name__ == '__main__':
    run()
