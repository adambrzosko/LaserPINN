"""
Spontaneous Raman spectrum of OM3 (and SMF-28 for reference) for the Chapter 4
characterisation, using the GMMNLSE solver on computed GRIN modes.

Mirrors the bench: a CW pump at 1551.72 nm (DWDM Ch32) launched into LP01 through the
photonic lantern, 9 mW in the fibre. The ensemble-mean forward-scattered PSD is
accumulated in every guided mode, so both the LP01-only reading (the SMF-coupled OSA
behind the output lantern) and the all-mode total are available. Backward-scattered PSD
follows analytically from the same calibrated source term.

Writes images/gmmnlse_raman_spectrum/data/
    parameters.json          inputs and derived scalars
    spectra.npz              forward/backward PSDs vs wavelength for each length
    modes.npz                guided-mode table (A_eff, D, DMD, beta0 offsets)
    length_optimisation.npz  peak Stokes PSD vs length at fixed and SBS-limited launch

Plot with the Chapter 4 cells of studies/replot_paper_10ghz.ipynb.

    python studies/gmmnlse_raman_spectrum.py
"""
import json
import time
from pathlib import Path

import numpy as np

from fiber.constants import c, hbar
from fiber.gmmnlse import GMMNLSE, TimeGrid
from fiber.grin_modes import DESIGNS, FibreModes
from fiber.raman_models import LinAgrawal, peak_gain_coefficient, spontaneous_shape

OUT = Path(__file__).resolve().parent.parent / 'images' / 'gmmnlse_raman_spectrum' / 'data'

PUMP_WAVELENGTH = 1551.72e-9
N2 = 2.6e-20
TEMPERATURE = 295.0
RAMAN = LinAgrawal()

OM3_LAUNCH_W = 9e-3
OM3_ALPHA_DB_KM = 0.3
OM3_LENGTHS_KM = [1.0, 2.0, 5.0, 10.0, 17.0]

SMF_ALPHA_DB_KM = 0.2
SMF_CASES = {'30km_30mW': (30.0, 30e-3), '10km_9mW': (10.0, 9e-3)}

GRID = TimeGrid(n_points=2 ** 13, dt=16e-15, wavelength=PUMP_WAVELENGTH)
SPAN_HZ = 31.3e12
DZ = 100.0

G_B = 5e-11                 # peak Brillouin gain, narrow-linewidth CW pump
LAUNCH_CEILING_W = 10e-3    # measured maximum after the input lantern
SBS_MARGIN = 0.8
LENGTH_SCAN_KM = np.linspace(0.25, 30.0, 120)
PEAK_SHIFT_HZ = 13.2e12


def _db_to_power_attenuation(dB_km):
    return dB_km / (10 * np.log10(np.e)) / 1e3


def principal_group(mode):
    return 2 * mode.m + mode.l - 1


def backward_psd(modes, receivers, P0, alpha, L):
    """Backward spontaneous-Raman PSD at the input: source a_q P0 exp(-alpha z), returning
    with exp(-alpha z), integrated over the fibre."""
    omega, Omega = GRID.omega, GRID.Omega
    S = modes.intensity_overlaps(receivers, [modes.index_of('LP01')])[:, 0]
    guided = np.asarray([modes.is_guided(q, omega) for q in receivers])
    a = (hbar * omega * (N2 * omega / c) * spontaneous_shape(RAMAN, Omega, TEMPERATURE))[None, :]
    return a * S[:, None] * guided * P0 * (1 - np.exp(-2 * alpha * L)) / (2 * alpha)


def run():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    order = np.argsort(2 * np.pi * c / GRID.omega)
    wavelength_nm = (2 * np.pi * c / GRID.omega)[order] * 1e9

    # ---------------------------------------------------------------- OM3
    om3 = FibreModes(DESIGNS['om3'], PUMP_WAVELENGTH, span_Hz=SPAN_HZ)
    i01, i11 = om3.index_of('LP01'), [om3.index_of('LP11a'), om3.index_of('LP11b')]
    groups = np.array([principal_group(m) for m in om3.modes])
    gmm = GMMNLSE(om3, GRID, ['LP01'], n2=N2, raman=RAMAN, alpha_dB_km=OM3_ALPHA_DB_KM,
                  noise='mean', temperature=TEMPERATURE)
    z_save = np.asarray(OM3_LENGTHS_KM) * 1e3
    res = gmm.propagate(GRID.cw(OM3_LAUNCH_W)[None, :], z_save[-1], dz=DZ, z_save=z_save)

    fwd = res.noise_psd[:, :, order]                               # (Nz, modes, N)
    unique_groups = np.unique(groups)
    fwd_groups = np.stack([fwd[:, groups == G].sum(axis=1) for G in unique_groups], axis=1)
    fwd_lp01, fwd_lp11, fwd_total = fwd[:, i01], fwd[:, i11].sum(axis=1), fwd.sum(axis=1)

    alpha = _db_to_power_attenuation(OM3_ALPHA_DB_KM)
    receivers = list(range(len(om3)))
    bwd = np.stack([backward_psd(om3, receivers, OM3_LAUNCH_W, alpha, L)[:, order] for L in z_save])
    bwd_lp01, bwd_total = bwd[:, i01], bwd.sum(axis=1)

    # ---------------------------------------------------------------- SMF-28
    smf = FibreModes(DESIGNS['smf28'], PUMP_WAVELENGTH, span_Hz=SPAN_HZ, dr=0.02e-6)
    smf_fwd = {}
    for key, (L_km, P) in SMF_CASES.items():
        r = GMMNLSE(smf, GRID, ['LP01'], n2=N2, raman=RAMAN, alpha_dB_km=SMF_ALPHA_DB_KM,
                    noise='mean', temperature=TEMPERATURE).propagate(GRID.cw(P)[None, :], L_km * 1e3, dz=DZ)
        smf_fwd[key] = r.noise_psd[-1, 0, order]

    # ---------------------------------------------------------------- length optimisation
    omega_s = GRID.omega0 - 2 * np.pi * PEAK_SHIFT_HZ
    A_eff = om3.effective_area(i01)
    a01 = (hbar * omega_s * (N2 * omega_s / c) / A_eff
           * float(spontaneous_shape(RAMAN, np.array([2 * np.pi * PEAK_SHIFT_HZ]), TEMPERATURE)[0]))
    L = LENGTH_SCAN_KM * 1e3
    L_eff = (1 - np.exp(-alpha * L)) / alpha
    p_sbs = 21 * A_eff / (G_B * L_eff)
    p_limited = np.minimum(LAUNCH_CEILING_W, SBS_MARGIN * p_sbs)
    peak_fixed = a01 * OM3_LAUNCH_W * L * np.exp(-alpha * L)
    peak_limited = a01 * p_limited * L * np.exp(-alpha * L)

    k_peak = int(np.argmin(np.abs(GRID.Omega - 2 * np.pi * PEAK_SHIFT_HZ)))
    check = {f'{Lk:g}km': float(res.noise_psd[j, i01, k_peak]
                                / (a01 * OM3_LAUNCH_W * Lk * 1e3 * np.exp(-alpha * Lk * 1e3)))
             for j, Lk in enumerate(OM3_LENGTHS_KM)}

    # ---------------------------------------------------------------- save
    lam_peak_nm = 2 * np.pi * c / omega_s * 1e9
    j_peak = int(np.argmin(np.abs(wavelength_nm - lam_peak_nm)))
    band = (wavelength_nm >= 1600) & (wavelength_nm <= 1700)
    np.savez(OUT / 'spectra.npz',
             wavelength_nm=wavelength_nm, length_km=np.asarray(OM3_LENGTHS_KM),
             fwd_lp01=fwd_lp01, fwd_lp11=fwd_lp11, fwd_total=fwd_total,
             fwd_groups=fwd_groups, group_numbers=unique_groups,
             bwd_lp01=bwd_lp01, bwd_total=bwd_total,
             pump_out_W=res.mean_power()[:, 0],
             smf_keys=np.array(list(SMF_CASES)),
             smf_fwd=np.stack([smf_fwd[k] for k in SMF_CASES]))
    b1 = om3.beta_derivative(i01, 1)
    np.savez(OUT / 'modes.npz',
             labels=np.array(om3.labels), l=np.array([m.l for m in om3.modes]),
             m=np.array([m.m for m in om3.modes]), group=groups,
             dbeta0_rad_m=om3.beta0 - om3.beta0[i01],
             dmd_ps_km=np.array([(om3.beta_derivative(p, 1) - b1) * 1e15 for p in range(len(om3))]),
             D_ps_nm_km=np.array([om3.dispersion_parameter(p) for p in range(len(om3))]),
             A_eff_um2=np.array([om3.effective_area(p) * 1e12 for p in range(len(om3))]),
             capture_rel_lp01=om3.intensity_overlaps(receivers, [i01])[:, 0] / (1 / A_eff))
    np.savez(OUT / 'length_optimisation.npz',
             length_km=LENGTH_SCAN_KM, p_sbs_W=p_sbs, p_limited_W=p_limited,
             peak_psd_fixed_W_Hz=peak_fixed, peak_psd_limited_W_Hz=peak_limited)

    params = {
        'pump_wavelength_nm': PUMP_WAVELENGTH * 1e9, 'n2_m2_W': N2, 'temperature_K': TEMPERATURE,
        'raman_model': RAMAN.name, 'raman_peak_gR_m_W': peak_gain_coefficient(RAMAN, N2, PUMP_WAVELENGTH),
        'om3': {
            'design': vars(DESIGNS['om3']), 'launch_W': OM3_LAUNCH_W, 'alpha_dB_km': OM3_ALPHA_DB_KM,
            'lengths_km': OM3_LENGTHS_KM, 'n_guided_modes': len(om3),
            'A_eff_lp01_um2': A_eff * 1e12, 'gamma_lp01_W_km': N2 * GRID.omega0 / c / A_eff * 1e3,
            'D_lp01_ps_nm_km': om3.dispersion_parameter(i01),
            'dmd_lp11_ps_km': (om3.beta_derivative(i11[0], 1) - b1) * 1e15,
            'lp01_fraction_at_peak': [float(fwd_lp01[j, j_peak] / fwd_total[j, j_peak]) for j in range(len(z_save))],
            'lp01_fraction_1600_1700nm': [float(fwd_lp01[j, band].sum() / fwd_total[j, band].sum()) for j in range(len(z_save))],
            # the analytic reference omits Raman gain on the noise, worth ~ g P L_eff / 2 (1% at 17 km)
            'solver_over_analytic_without_raman_gain': check,
        },
        'smf28': {'design': vars(DESIGNS['smf28']), 'alpha_dB_km': SMF_ALPHA_DB_KM, 'cases': SMF_CASES,
                  'A_eff_um2': smf.effective_area(0) * 1e12, 'D_ps_nm_km': smf.dispersion_parameter(0)},
        'length_optimisation': {'g_B_m_W': G_B, 'launch_ceiling_W': LAUNCH_CEILING_W,
                                'sbs_margin': SBS_MARGIN, 'peak_shift_Hz': PEAK_SHIFT_HZ,
                                'optimum_limited_km': float(LENGTH_SCAN_KM[np.argmax(peak_limited)]),
                                'optimum_fixed_km': float(LENGTH_SCAN_KM[np.argmax(peak_fixed)])},
        'grid': {'n_points': GRID.n_points, 'dt_s': GRID.dt, 'df_Hz': GRID.df}, 'dz_m': DZ,
        'runtime_s': time.time() - t0,
    }
    (OUT / 'parameters.json').write_text(json.dumps(params, indent=2, default=float))

    om = params['om3']
    print(f"OM3: {om['n_guided_modes']} guided modes, A_eff(LP01) = {om['A_eff_lp01_um2']:.1f} um^2, "
          f"D = {om['D_lp01_ps_nm_km']:.2f} ps/nm/km, LP11 DMD = {om['dmd_lp11_ps_km']:.1f} ps/km")
    print(f"LP01 share of forward Raman at {lam_peak_nm:.0f} nm: "
          + ', '.join(f'{Lk:g} km {f:.1%}' for Lk, f in zip(OM3_LENGTHS_KM, om['lp01_fraction_at_peak'])))
    print(f"solver / analytic peak PSD: {check}")
    print(f"optimum length: {params['length_optimisation']['optimum_limited_km']:.1f} km (SBS/ceiling limited), "
          f"{params['length_optimisation']['optimum_fixed_km']:.1f} km (fixed 9 mW)")
    print(f"saved to {OUT}  [{params['runtime_s']:.1f} s]")


if __name__ == '__main__':
    run()
