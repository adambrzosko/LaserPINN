"""
Does the multi-longitudinal-mode structure of an FP laser assist phase
randomisation, and how strongly does the benefit depend on the number of
lasing modes?

The real device's mode count is not known (the Chapter 7 spectra are
instrument screenshots with illegible axes; peak counting suggests many tens
of lines but is at the resolution limit).  M is therefore treated as an
unknown and bracketed.

Design note — why per-mode S_inj is held FIXED across M:
    S_inj,total = P_SLD * eta_c * (M * dnu_cav / dnu_SLD) * tau_p / (h nu V)
    S_inj,per-mode = S_inj,total / M
The M cancels: at fixed SLD power and coupling efficiency, the per-mode
injected density does not depend on how many modes the laser has.  Adding
modes widens the collective acceptance bandwidth, but the captured power is
shared among proportionally more modes.  What does improve with M is the
RESIDUAL coherent field per mode at turn-on, which is divided among M modes
and so is easier for a fixed per-mode injection to overwrite.  Sweeping M at
fixed per-mode S_inj therefore isolates exactly that effect.

Run:  python studies/fp_mode_count_sweep.py
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, '.')

from core.fp_laser import (FPLaserParams, gain_switch_fp, compute_r1,
                           mode_power_stats, amzi_splitting_ratio)

# ── Configuration ─────────────────────────────────────────────────────────────
MODE_COUNTS = [5, 11, 21, 41]

# Per-mode injected photon density (m^-3).  Bounded by the coupling
# calculation: at 19 mW SLD these stay within eta_c <= 1 for every M above.
S_INJ_LEVELS = [0.0, 3e18, 1e19]

F_REP = 2e9
N_PULSES = 5000       # r1 floor ~ 0.014
DUTY = 0.30
DT = 0.5e-12
SEED = 42

OUT_NPZ = 'images/fp_randomisation/fp_mode_count_sweep.npz'


def main():
    floor = 1 / np.sqrt(N_PULSES)
    print('=' * 80)
    print('  FP randomisation vs longitudinal mode count')
    print('=' * 80)
    print(f'  f_rep = {F_REP*1e-9:.0f} GHz   N = {N_PULSES} pulses   '
          f'r1 floor ~ {floor:.4f}')
    print(f'  per-mode S_inj held fixed across M (see module docstring)')
    print()
    print(f'{"M":>4} | {"FSR (GHz)":>10} | {"S_inj":>9} | {"r1":>8} | '
          f'{"switches":>11} | {"eta std":>8} | {"pk mode":>8}')
    print('-' * 80)

    rows = []
    for M in MODE_COUNTS:
        fp = FPLaserParams(n_modes=M)
        I_th = fp.threshold_current()

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
            eta = amzi_splitting_ratio(res['E_peak'], res['sol']['dnu'],
                                       1 / F_REP)
            pk_mode = int(np.argmax(st['mean_spectrum']) - M // 2)

            lbl = 'free-run' if S_inj == 0 else f'{S_inj:.0e}'
            print(f'{M:4d} | {fp.FSR_hz*1e-9:10.1f} | {lbl:>9} | {r1:8.4f} | '
                  f'{nsw:5d}/{N_PULSES-1:5d} | {np.std(eta):8.3f} | '
                  f'{pk_mode:+8d}   [{time.time()-t0:.0f}s]')

            rows.append({
                'M': M, 'S_inj': S_inj, 'r1': r1, 'n_switches': nsw,
                'eta_std': float(np.std(eta)),
                'pk_mode': pk_mode,
                'S_peak_total': float(np.mean(res['S_peak_total'])),
                'mpn_k_max': float(np.max(st['mpn_k'])),
            })
        print('-' * 80)

    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    np.savez(
        OUT_NPZ,
        M=np.array([r['M'] for r in rows]),
        S_inj=np.array([r['S_inj'] for r in rows]),
        r1=np.array([r['r1'] for r in rows]),
        n_switches=np.array([r['n_switches'] for r in rows]),
        eta_std=np.array([r['eta_std'] for r in rows]),
        pk_mode=np.array([r['pk_mode'] for r in rows]),
        S_peak_total=np.array([r['S_peak_total'] for r in rows]),
        mpn_k_max=np.array([r['mpn_k_max'] for r in rows]),
        r1_floor=floor, f_rep=F_REP, n_pulses=N_PULSES,
    )
    print(f'\n  Saved: {OUT_NPZ}')

    # Headline comparison: free-running r1 against M
    print('\n  Free-running r1 vs M (does multimode structure help by itself?)')
    for r in rows:
        if r['S_inj'] == 0.0:
            print(f'    M = {r["M"]:3d}  ->  r1 = {r["r1"]:.4f}')


if __name__ == '__main__':
    main()
