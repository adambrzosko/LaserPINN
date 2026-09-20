"""FP phase-randomisation results re-measured with the power-weighted estimator.

Usage:  python studies/fp_validated_sweeps.py {headline2g,inj300,injlong}

Each finished point is appended to images/fp_randomisation/validated_<study>.csv
and skipped on rerun, so an interrupted run resumes instead of starting over.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, '.')

from core.dfb_laser import DFBLaserParams
from core.fp_laser import FPLaserParams, gain_switch_fp, mode_power_stats
from core.phase_estimators import r1_summed_field, r1_power_weighted
from core.sld_injection import solve_transient_injection_stochastic

OUT_DIR = 'images/fp_randomisation'
DT = 0.5e-12
DUTY = 0.30
H = 6.626e-34
C = 3e8

# DFB is single-mode, so its summed-field r1 values from earlier runs remain valid.
DFB_2G_BY_SEED = {42: 0.2538, 7: 0.2717, 1234: 0.2857, 2025: 0.2866, 99: 0.2766}
DFB_10G_BY_TOTAL = {0.0: 0.9879, 2.1e18: 0.8030, 2.1e19: 0.3795,
                    2.1e20: 0.0322, 1.05e21: 0.0052}


def dfb_10g_at(total):
    for key, val in DFB_10G_BY_TOTAL.items():
        if (key == 0.0 and total == 0.0) or (key > 0 and abs(total - key) / key < 0.01):
            return val
    return float('nan')


def eta_required(fp, per_mode, p_sld=19e-3, sld_bw=33e-9):
    if per_mode <= 0:
        return 0.0
    nu = C / fp.lambda0
    dnu_cav = 1 / (2 * np.pi * fp.tau_p)
    dnu_sld = C * sld_bw / fp.lambda0**2
    return per_mode * H * nu * fp.V / (p_sld * fp.tau_p * (dnu_cav / dnu_sld))


def run_point(fp, f_rep, n_pulses, seed, s_inj):
    I_th = fp.threshold_current()
    res = gain_switch_fp(fp, f_rep=f_rep, n_pulses=n_pulses,
                         I_off=0.9 * I_th, I_on=3.0 * I_th,
                         duty=DUTY, dt=DT, seed=seed, S_inj_density=s_inj)
    E = res['E_peak']
    st = mode_power_stats(res['S_peak'])
    return {
        'r1_summed': r1_summed_field(E),
        'r1_weighted': r1_power_weighted(E),
        'switches': int(np.sum(np.diff(st['dominant_mode']) != 0)),
    }


def done_keys(path, key):
    if not os.path.exists(path):
        return set()
    with open(path, newline='') as f:
        return {float(row[key]) for row in csv.DictReader(f)}


def append_row(path, row):
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def header(name, fp, f_rep, n_pulses):
    print(f'== {name} ==  L={fp.L*1e6:.0f}um  M={fp.n_modes}  tau_p={fp.tau_p*1e12:.2f}ps  '
          f'I_th={fp.threshold_current()*1e3:.1f}mA  f_rep={f_rep*1e-9:.0f}GHz  '
          f'N={n_pulses}  floor={1/np.sqrt(n_pulses):.4f}', flush=True)


def headline2g():
    fp, f_rep, n = FPLaserParams(), 2e9, 3000
    path = f'{OUT_DIR}/validated_headline2g.csv'
    header('headline2g', fp, f_rep, n)
    done = done_keys(path, 'seed')
    for seed, dfb in DFB_2G_BY_SEED.items():
        if float(seed) in done:
            print(f'  seed {seed}: already done, skipping', flush=True)
            continue
        t0 = time.time()
        row = {'seed': seed, 'n_pulses': n, **run_point(fp, f_rep, n, seed, 0.0),
               'dfb_r1': dfb}
        append_row(path, row)
        print(f'  {row}  [{time.time()-t0:.0f}s]', flush=True)


def injection_sweep(name, fp, per_mode_levels):
    f_rep, n, seed = 10e9, 20000, 42
    path = f'{OUT_DIR}/validated_{name}.csv'
    header(name, fp, f_rep, n)
    done = done_keys(path, 'per_mode')
    for pm in per_mode_levels:
        if pm in done:
            print(f'  per-mode {pm:.3e}: already done, skipping', flush=True)
            continue
        t0 = time.time()
        total = pm * fp.n_modes
        row = {'per_mode': pm, 'total': total, 'eta_c_required': eta_required(fp, pm),
               'n_pulses': n, **run_point(fp, f_rep, n, seed, pm),
               'dfb_r1_matched_total': dfb_10g_at(total)}
        append_row(path, row)
        print(f'  {row}  [{time.time()-t0:.0f}s]', flush=True)


def inj300():
    injection_sweep('inj300', FPLaserParams(), [0.0, 1e17, 1e18, 1e19, 5e19])


def injlong():
    fp = FPLaserParams(L=1200e-6, n_modes=85)
    # per-mode levels chosen so totals match the DFB points already computed
    injection_sweep('injlong', fp, [t / fp.n_modes for t in sorted(DFB_10G_BY_TOTAL)])


# Refine the r1 = 0.01 crossings: both FP cavities cross between totals 2.1e19
# and 2.1e20, the DFB between 2.1e20 and 1.05e21.
BRACKET_FP_TOTALS = [4e19, 7e19, 1.2e20]
BRACKET_DFB_TOTALS = [3e20, 4.5e20, 7e20]


def dfb_point(f_rep, n_pulses, seed, s_inj):
    las = DFBLaserParams()
    I_th = las.threshold_current()
    T = 1 / f_rep
    t_on = DUTY * T
    t_rise = min(20e-12, t_on / 4)
    I_off, I_on = 0.9 * I_th, 3.0 * I_th

    def I_func(t):
        ph = t % T
        if ph < t_rise:
            return I_off + (I_on - I_off) * 0.5 * (1 - np.cos(np.pi * ph / t_rise))
        if ph < t_on - t_rise:
            return I_on
        if ph < t_on:
            return I_off + (I_on - I_off) * 0.5 * (
                1 + np.cos(np.pi * (ph - t_on + t_rise) / t_rise))
        return I_off

    warmup = 20
    t_total = (warmup + n_pulses) * T
    sol = solve_transient_injection_stochastic(
        las, I_func, inj=None, S_inj=s_inj, sld_tau_coh=None,
        t_span=[0, t_total], t_eval=np.arange(0, t_total, DT), seed=seed)
    S, phi = sol.y[1], sol.y[2]
    steps = int(T / DT)
    E = []
    for p in range(n_pulses):
        i0 = (warmup + p) * steps
        i1 = min(i0 + steps, len(S))
        if i1 <= i0:
            break
        k = np.argmax(S[i0:i1]) + i0
        E.append(np.sqrt(S[k]) * np.exp(1j * phi[k]))
    return r1_power_weighted(np.array(E))


def bracket300():
    fp = FPLaserParams()
    injection_sweep('bracket300', fp, [t / fp.n_modes for t in BRACKET_FP_TOTALS])


def bracketlong():
    fp = FPLaserParams(L=1200e-6, n_modes=85)
    injection_sweep('bracketlong', fp, [t / fp.n_modes for t in BRACKET_FP_TOTALS])


def bracketdfb():
    f_rep, n, seed = 10e9, 20000, 42
    las = DFBLaserParams()
    path = f'{OUT_DIR}/validated_bracketdfb.csv'
    print(f'== bracketdfb ==  tau_p={las.tau_p*1e12:.2f}ps  f_rep={f_rep*1e-9:.0f}GHz  '
          f'N={n}  floor={1/np.sqrt(n):.4f}', flush=True)
    done = done_keys(path, 'total')
    for total in BRACKET_DFB_TOTALS:
        if total in done:
            print(f'  total {total:.3e}: already done, skipping', flush=True)
            continue
        t0 = time.time()
        row = {'total': total, 'eta_c_required': eta_required(las, total),
               'n_pulses': n, 'r1': dfb_point(f_rep, n, seed, total)}
        append_row(path, row)
        print(f'  {row}  [{time.time()-t0:.0f}s]', flush=True)


STUDIES = {'headline2g': headline2g, 'inj300': inj300, 'injlong': injlong,
           'bracket300': bracket300, 'bracketlong': bracketlong,
           'bracketdfb': bracketdfb}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('study', choices=STUDIES)
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    STUDIES[args.study]()
