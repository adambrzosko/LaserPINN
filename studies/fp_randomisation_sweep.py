"""
FP laser phase randomisation: high-statistics injection sweep, with a
single-mode DFB run at the same drive conditions as reference.

The scientific question is whether the multi-longitudinal-mode structure of
an FP laser assists phase randomisation relative to a single-mode DFB, and
whether the broader collective acceptance bandwidth of M modes resolves the
unphysical (>1) coupling fraction found for the DFB in Chapter 5.

Run:  python studies/fp_randomisation_sweep.py
"""

import sys
import time
import numpy as np

sys.path.insert(0, '.')

from core.fp_laser import (FPLaserParams, gain_switch_fp, compute_r1,
                           mode_power_stats, amzi_splitting_ratio)
from core.dfb_laser import DFBLaserParams
from core.sld_injection import solve_transient_injection_stochastic

# ── Sweep configuration ───────────────────────────────────────────────────────
# Injection levels (per-mode photon density, m^-3).  Bounded by the coupling
# calculation: the FP's 21 modes collectively accept 1.57 THz of the SLD's
# 4.12 THz bandwidth (spectral overlap 0.381, against 0.018 for a single DFB
# mode), so at 19 mW the reachable per-mode range is ~5e17 (eta_c = 0.01) to
# ~5e19 (eta_c = 1).  1e20 would demand eta_c = 2.1 and is excluded: at that
# level the injected light is amplified rather than seeding, which shows up as
# the mean spectrum peaking many FSRs off the gain peak.
S_INJ_LEVELS = [0.0, 1e18, 3e18, 1e19, 3e19]

F_REP = 2e9          # matches the experimental FP_GS_2GHz measurement
# Coarse scan first: 5000 pulses gives an r1 floor of ~0.014, enough to
# establish the trend and locate the threshold region.  Resolving r1 < 0.01
# needs N >> 1e4, so a targeted high-statistics run at the candidate point
# follows rather than paying that cost at every point here.
N_PULSES = 5000
DUTY = 0.30
DT = 0.5e-12
SEED = 42

OUT_NPZ = 'images/fp_randomisation/fp_sweep.npz'


def run_fp_sweep():
    fp = FPLaserParams()
    I_th = fp.threshold_current()
    floor = 1 / np.sqrt(N_PULSES)

    print('=' * 78)
    print('  FP multimode phase randomisation sweep')
    print('=' * 78)
    print(f'  I_th = {I_th*1e3:.1f} mA   FSR = {fp.FSR_hz*1e-9:.1f} GHz   '
          f'M = {fp.n_modes} modes')
    print(f'  f_rep = {F_REP*1e-9:.0f} GHz   N = {N_PULSES} pulses   '
          f'r1 floor ~ {floor:.4f}')
    print()
    print(f'{"S_inj (m^-3)":>14} | {"r1":>8} | {"switches":>11} | '
          f'{"eta std":>8} | {"S_tot pk":>10} | {"pk mode":>8}')
    print('-' * 78)

    records = []
    for S_inj in S_INJ_LEVELS:
        t0 = time.time()
        res = gain_switch_fp(
            fp, f_rep=F_REP, n_pulses=N_PULSES,
            I_off=0.9 * I_th, I_on=3.0 * I_th,
            duty=DUTY, dt=DT, seed=SEED, S_inj_density=S_inj,
        )
        r1 = compute_r1(res['E_peak'])
        st = mode_power_stats(res['S_peak'])
        nsw = int(np.sum(np.diff(st['dominant_mode']) != 0))
        eta = amzi_splitting_ratio(res['E_peak'], res['sol']['dnu'], 1 / F_REP)
        S_pk = float(np.mean(res['S_peak_total']))
        pk_mode = int(np.argmax(st['mean_spectrum']) - fp.n_modes // 2)

        lbl = 'free-run' if S_inj == 0 else f'{S_inj:.0e}'
        print(f'{lbl:>14} | {r1:8.4f} | {nsw:5d}/{N_PULSES-1:5d} | '
              f'{np.std(eta):8.3f} | {S_pk:10.2e} | {pk_mode:+8d}'
              f'   [{time.time()-t0:.0f}s]')

        records.append({
            'S_inj': S_inj, 'r1': r1, 'n_switches': nsw,
            'eta_std': float(np.std(eta)), 'eta': eta,
            'S_peak_total': S_pk, 'pk_mode': pk_mode,
            'mean_spectrum': st['mean_spectrum'],
            'mpn_k': st['mpn_k'],
        })

    return fp, records, floor


def run_dfb_reference():
    """Single-mode DFB at the same rate and drive, for comparison."""
    laser = DFBLaserParams()
    I_th = laser.threshold_current()
    T_rep = 1 / F_REP
    t_on = DUTY * T_rep
    t_rise = min(20e-12, t_on / 4)
    I_off, I_on = 0.9 * I_th, 3.0 * I_th

    def I_func(t):
        ph = t % T_rep
        if ph < t_rise:
            return I_off + (I_on - I_off) * 0.5 * (1 - np.cos(np.pi * ph / t_rise))
        elif ph < t_on - t_rise:
            return I_on
        elif ph < t_on:
            return I_off + (I_on - I_off) * 0.5 * (
                1 + np.cos(np.pi * (ph - t_on + t_rise) / t_rise))
        return I_off

    warmup = 20
    n_rec = min(N_PULSES, 2000)   # pure-Python loop; cap to keep runtime sane
    total = warmup + n_rec
    t_total = total * T_rep
    t_eval = np.arange(0, t_total, DT)

    print()
    print('-' * 78)
    print(f'  DFB single-mode reference at {F_REP*1e-9:.0f} GHz '
          f'({n_rec} pulses, I_th = {I_th*1e3:.1f} mA)')
    print('-' * 78)
    print(f'{"S_inj (m^-3)":>14} | {"r1":>8}')
    print('-' * 28)

    dfb_records = []
    for S_inj in [0.0, 1e19, 1e20]:
        sol = solve_transient_injection_stochastic(
            laser, I_func, inj=None, S_inj=S_inj,
            sld_tau_coh=None, t_span=[0, t_total],
            t_eval=t_eval, seed=SEED,
        )
        S = sol.y[1]
        phi = sol.y[2]
        steps_per = int(T_rep / DT)
        E_pk = []
        for p in range(n_rec):
            i0 = (warmup + p) * steps_per
            i1 = min(i0 + steps_per, len(S))
            if i1 <= i0:
                break
            ipk = np.argmax(S[i0:i1]) + i0
            E_pk.append(np.sqrt(S[ipk]) * np.exp(1j * phi[ipk]))
        E_pk = np.array(E_pk)
        ph = np.angle(E_pk)
        r1 = np.abs(np.mean(np.exp(1j * np.diff(ph))))
        lbl = 'free-run' if S_inj == 0 else f'{S_inj:.0e}'
        print(f'{lbl:>14} | {r1:8.4f}')
        dfb_records.append({'S_inj': S_inj, 'r1': r1})

    return dfb_records


if __name__ == '__main__':
    fp, records, floor = run_fp_sweep()

    try:
        dfb_records = run_dfb_reference()
    except Exception as e:
        print(f'\n  DFB reference failed: {e}')
        dfb_records = []

    import os
    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    np.savez(
        OUT_NPZ,
        S_inj=np.array([r['S_inj'] for r in records]),
        r1=np.array([r['r1'] for r in records]),
        n_switches=np.array([r['n_switches'] for r in records]),
        eta_std=np.array([r['eta_std'] for r in records]),
        S_peak_total=np.array([r['S_peak_total'] for r in records]),
        pk_mode=np.array([r['pk_mode'] for r in records]),
        mean_spectrum=np.array([r['mean_spectrum'] for r in records]),
        mpn_k=np.array([r['mpn_k'] for r in records]),
        dfb_S_inj=np.array([r['S_inj'] for r in dfb_records]),
        dfb_r1=np.array([r['r1'] for r in dfb_records]),
        r1_floor=floor,
        f_rep=F_REP, n_pulses=N_PULSES, n_modes=fp.n_modes,
    )
    print(f'\n  Saved: {OUT_NPZ}')
