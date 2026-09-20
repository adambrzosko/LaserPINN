"""Diagnostic: is the coherent 2 GHz FP AMZI measurement explained by the laser
never being switched off between pulses?

Sweeps the minimum drive current around threshold for
  rc   - raised-cosine drive (core.fp_laser.gain_switch_fp), duty 0.30
  sin  - sinusoidal drive I_dc + I_rf sin(2 pi f t) (Chapter 5 DFB style)
  dfb  - single-mode DFB reference, raised-cosine drive, free running
with I_on (peak) = 3.0 I_th, and for the FP at per-mode SLD injection
S_inj_density in {0, 1e18, 1e19} m^-3.

Usage:
  python studies/fp_diag_drive.py rc  [--mins 0.9 1.0] [--inj 0 1e18]
  python studies/fp_diag_drive.py sin [--mins ...] [--inj ...]
  python studies/fp_diag_drive.py dfb [--mins ...]
  python studies/fp_diag_drive.py plot

Every finished condition is appended to images/fp_randomisation/diag_drive.csv
and skipped on rerun.

Column meanings (all per condition, N_PULSES consecutive pulses after 20 warm-up periods)
  r1_w_peak       r1_power_weighted(E_peak), E sampled at each period's total-intensity peak
  r1_top_peak     r1 of the single highest-mean-power mode, same sampling
  r1_w_fd, r1_top_fd
                  same, but the reference field is taken exactly one period (1000 steps)
                  before the peak instead of at the previous peak.  This is what an AMZI
                  with a 500 ps delay compares, and it is immune to peak-timing jitter
                  multiplying each mode's frequency offset.
  x_*             normalised AMZI output x = 2 Re(E E_prev*) / (|E|^2 + |E_prev|^2) with the
                  fixed-delay reference, for the top line (top), the 3 strongest lines (few)
                  and all lines on one photodiode (full); std and lag-1 autocorrelation.
                  Random equal-amplitude phases give std 0.707 (arcsine), locked gives std ~0.
  N_min_off_rel   median over periods of min N in the off-interval / N_th
  frac_per_N_below
                  fraction of periods in which N < N_th somewhere in the off-interval
  frac_time_N_below
                  fraction of off-interval time with N < N_th
  S_min_off_rel   median over periods of min S_total in the off-interval / that period's peak
  S_trough_rel    median over periods of min S_total between consecutive peaks / peak
  S_trough_top    median top-mode photon density at that trough (m^-3), to compare with S_inj
  off-interval    rc/dfb: t mod T >= duty*T (current at I_off); sin: t mod T >= T/2 (I < I_dc)
"""

import argparse
import csv
import gc
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.dfb_laser import DFBLaserParams
from core.fp_laser import FPLaserParams, gain_switch_fp, solve_fp_stochastic
from core.phase_estimators import r1_power_weighted
from core.sld_injection import solve_transient_injection_stochastic
from gsdfb import save_fig

OUT_DIR = os.path.join(ROOT, 'images/fp_randomisation')
CSV_PATH = os.path.join(OUT_DIR, 'diag_drive.csv')
TRACE_PATH = os.path.join(OUT_DIR, 'diag_drive_traces.npz')
FIG_PATH = os.path.join(OUT_DIR, 'diag_drive.png')

F_REP = 2e9
T_REP = 1 / F_REP
DT = 0.5e-12
DUTY = 0.30
N_PULSES = 1500
WARMUP = 20
SEED = 42
I_ON_REL = 3.0
RC_MINS = [0.9, 0.95, 1.0, 1.05, 1.2, 1.5]
SIN_MINS = [0.9, 1.0, 1.2]
INJ_LEVELS = [0.0, 1e18, 1e19]
STEPS = int(round(T_REP / DT))


def n_threshold(p):
    return p.N_tr + (p.alpha_i + p.alpha_m) / (p.Gamma * p.a)


def raised_cosine(I_off, I_on, T=T_REP, duty=DUTY):
    """Identical to the waveform inside core.fp_laser.gain_switch_fp."""
    t_on = duty * T
    t_rise = min(20e-12, t_on / 4)

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
    return I_func


def sinusoid(I_min, I_max, f=F_REP):
    I_dc, I_rf = 0.5 * (I_max + I_min), 0.5 * (I_max - I_min)
    w = 2 * np.pi * f

    def I_func(t):
        return I_dc + I_rf * np.sin(w * t)
    return I_func


# ── analysis ────────────────────────────────────────────────────────────────────

def _r1_modes(E, E_ref):
    d = np.angle(E) - np.angle(E_ref)
    return np.abs(np.mean(np.exp(1j * d), axis=0))


def _amzi_x(E, E_ref):
    num = 2 * np.real(E * np.conj(E_ref)).sum(axis=1)
    den = (np.abs(E)**2 + np.abs(E_ref)**2).sum(axis=1)
    return num / np.maximum(den, 1e-300)


def _ac1(x):
    x = x - x.mean()
    v = np.dot(x, x)
    return float(np.dot(x[1:], x[:-1]) / v) if v > 0 else float('nan')


def analyse(N, S, get_E, N_th, off_start, n_pulses=None, steps=STEPS, warmup=WARMUP):
    """N: (n_steps,), S: (n_steps, M) photon densities, get_E(idx) -> (len(idx), M) complex."""
    n_pulses = n_pulses or N_PULSES
    i0 = warmup * steps
    i1 = i0 + n_pulses * steps
    S_tot = S.sum(axis=1)
    St = S_tot[i0:i1].reshape(n_pulses, steps)
    ar = np.arange(n_pulses)
    k_pk = St.argmax(axis=1)
    idx_pk = i0 + ar * steps + k_pk
    S_pk = St[ar, k_pk]

    S_min_off = St[:, off_start:].min(axis=1)
    Nn = N[i0:i1].reshape(n_pulses, steps)[:, off_start:]
    N_min_off = Nn.min(axis=1)

    trough_idx = np.array([idx_pk[p - 1] + np.argmin(S_tot[idx_pk[p - 1]:idx_pk[p] + 1])
                           for p in range(1, n_pulses)])
    S_trough = S_tot[trough_idx]

    E_pk = get_E(idx_pk)
    E_ref = get_E(idx_pk - steps)          # exactly one period earlier (AMZI delay)
    P = np.mean(np.abs(E_pk)**2, axis=0)
    order = np.argsort(P)[::-1]
    top = int(order[0])
    few = order[:3]

    r1_fd_modes = _r1_modes(E_pk, E_ref)
    x_top = _amzi_x(E_pk[:, [top]], E_ref[:, [top]])
    x_few = _amzi_x(E_pk[:, few], E_ref[:, few])
    x_full = _amzi_x(E_pk, E_ref)
    dom = np.argmax(np.abs(E_pk)**2, axis=1)

    return {
        'r1_w_peak': r1_power_weighted(E_pk),
        'r1_top_peak': r1_power_weighted(E_pk[:, top]),
        'r1_w_fd': float(np.sum(r1_fd_modes * P) / np.sum(P)),
        'r1_top_fd': float(r1_fd_modes[top]),
        'top_mode': top,
        'top_frac': float(P[top] / P.sum()),
        'switches': int(np.sum(np.diff(dom) != 0)),
        'x_top_std': float(np.std(x_top)), 'x_top_ac1': _ac1(x_top),
        'x_few_std': float(np.std(x_few)), 'x_few_ac1': _ac1(x_few),
        'x_full_std': float(np.std(x_full)), 'x_full_ac1': _ac1(x_full),
        'N_min_off_rel': float(np.median(N_min_off) / N_th),
        'N_min_off_rel_max': float(np.max(N_min_off) / N_th),
        'frac_per_N_below': float(np.mean(N_min_off < N_th)),
        'frac_time_N_below': float(np.mean(Nn < N_th)),
        'S_min_off_rel': float(np.median(S_min_off / S_pk)),
        'S_trough_rel': float(np.median(S_trough / S_pk[1:])),
        'S_trough_rel_p90': float(np.percentile(S_trough / S_pk[1:], 90)),
        'S_trough_top': float(np.median(S[trough_idx, top])),
        'S_peak_tot': float(np.median(S_pk)),
    }, idx_pk


def trace_snippet(t, I_func, N, S_tot, N_th, idx_pk, n_periods=4, stride=2):
    i0 = (WARMUP + min(100, N_PULSES - n_periods)) * STEPS
    sl = slice(i0, i0 + n_periods * STEPS, stride)
    tt = t[sl]
    pk = np.median(S_tot[idx_pk])
    return np.stack([tt - tt[0], np.array([I_func(x) for x in tt]),
                     N[sl] / N_th, S_tot[sl] / pk])


# ── runners ─────────────────────────────────────────────────────────────────────

def run_fp(drive, min_rel, s_inj, seed=SEED):
    fp = FPLaserParams()
    I_th = fp.threshold_current()
    N_th = n_threshold(fp)
    I_min, I_max = min_rel * I_th, I_ON_REL * I_th
    if drive == 'rc':
        res = gain_switch_fp(fp, F_REP, N_PULSES, I_min, I_max, duty=DUTY, dt=DT,
                             S_inj_density=s_inj, seed=seed, warmup_pulses=WARMUP)
        sol = res['sol']
        E_peak_core = res['E_peak']
        I_func = raised_cosine(I_min, I_max)
        off_start = int(round(DUTY * STEPS))
        del res
    else:
        I_func = sinusoid(I_min, I_max)
        sol = solve_fp_stochastic(fp, I_func, (0, (WARMUP + N_PULSES) * T_REP), dt=DT,
                                  S_inj_density=s_inj, seed=seed)
        E_peak_core = None
        off_start = STEPS // 2
    E_all = sol['E']
    out, idx_pk = analyse(sol['N'], sol['S'], lambda idx: E_all[idx], N_th, off_start)
    if E_peak_core is not None:
        assert np.array_equal(E_peak_core, E_all[idx_pk]), 'peak extraction mismatch'
    snip = trace_snippet(sol['t'], I_func, sol['N'], sol['S'].sum(axis=1), N_th, idx_pk)
    del sol, E_all
    gc.collect()
    return out, snip


def run_dfb(min_rel, seed=SEED):
    las = DFBLaserParams()
    I_th = las.threshold_current()
    N_th = n_threshold(las)
    I_func = raised_cosine(min_rel * I_th, I_ON_REL * I_th)
    t_total = (WARMUP + N_PULSES) * T_REP
    sol = solve_transient_injection_stochastic(
        las, I_func, inj=None, S_inj=0.0, sld_tau_coh=None,
        t_span=[0, t_total], t_eval=np.arange(0, t_total, DT), seed=seed)
    N, S, phi = sol.y[0], sol.y[1], sol.y[2]
    out, idx_pk = analyse(N, S[:, None],
                          lambda idx: (np.sqrt(S[idx]) * np.exp(1j * phi[idx]))[:, None],
                          N_th, int(round(DUTY * STEPS)))
    snip = trace_snippet(sol.t, I_func, N, S, N_th, idx_pk)
    return out, snip


# ── bookkeeping ─────────────────────────────────────────────────────────────────

def label(model, drive, min_rel, s_inj, seed):
    return f'{model}|{drive}|{min_rel:g}|{s_inj:g}|{seed}'


def done_labels():
    if not os.path.exists(CSV_PATH):
        return set()
    with open(CSV_PATH, newline='') as f:
        return {row['label'] for row in csv.DictReader(f)}


def append_row(row):
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def save_trace(key, snip):
    data = {}
    if os.path.exists(TRACE_PATH):
        with np.load(TRACE_PATH) as z:
            data = {k: z[k] for k in z.files}
    data[key.replace('|', '__')] = snip
    np.savez(TRACE_PATH, **data)


def run_condition(model, drive, min_rel, s_inj, seed=SEED):
    key = label(model, drive, min_rel, s_inj, seed)
    if key in done_labels():
        print(f'  {key}: done, skipping', flush=True)
        return
    t0 = time.time()
    if model == 'fp':
        out, snip = run_fp(drive, min_rel, s_inj, seed)
    else:
        out, snip = run_dfb(min_rel, seed)
    row = {'label': key, 'model': model, 'drive': drive, 'I_min_rel': min_rel,
           'I_on_rel': I_ON_REL, 'S_inj_density': s_inj, 'seed': seed,
           'n_pulses': N_PULSES, **out, 'runtime_s': round(time.time() - t0, 1)}
    append_row(row)
    save_trace(key, snip)
    print(f'  {key}: r1_w={out["r1_w_peak"]:.3f} r1_top={out["r1_top_peak"]:.3f} '
          f'r1_top_fd={out["r1_top_fd"]:.3f} x_top_std={out["x_top_std"]:.3f} '
          f'Nmin/Nth={out["N_min_off_rel"]:.3f} Str/Spk={out["S_trough_rel"]:.2e} '
          f'Str_top={out["S_trough_top"]:.2e} [{row["runtime_s"]}s]', flush=True)


# ── plot ────────────────────────────────────────────────────────────────────────

def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with open(CSV_PATH, newline='') as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    styles = {0.0: ('k', 'o', 'free'), 1e18: ('tab:blue', 's', 'SLD 1e18'),
              1e19: ('tab:red', '^', 'SLD 1e19')}
    for ax, drive, model, title in [(axes[0], 'rc', 'fp', 'FP raised cosine'),
                                    (axes[1], 'sin', 'fp', 'FP sinusoid')]:
        for s_inj, (col, mk, lab) in styles.items():
            sel = sorted([r for r in rows if r['model'] == model and r['drive'] == drive
                          and float(r['S_inj_density']) == s_inj],
                         key=lambda r: float(r['I_min_rel']))
            if not sel:
                continue
            x = [float(r['I_min_rel']) for r in sel]
            ax.plot(x, [float(r['r1_top_peak']) for r in sel], color=col, marker=mk,
                    label=f'{lab}: top line')
            ax.plot(x, [float(r['r1_w_peak']) for r in sel], color=col, marker=mk,
                    ls='--', mfc='none', label=f'{lab}: power-weighted')
        if drive == 'rc':
            sel = sorted([r for r in rows if r['model'] == 'dfb'],
                         key=lambda r: float(r['I_min_rel']))
            if sel:
                ax.plot([float(r['I_min_rel']) for r in sel],
                        [float(r['r1_top_peak']) for r in sel], color='tab:green',
                        marker='D', label='DFB free')
        ax.axhline(0.5, color='grey', lw=0.8, ls=':')
        ax.set(xlabel='I_min / I_th', ylabel='r1', title=f'{title}, 2 GHz, I_on = 3 I_th',
               ylim=(-0.02, 1.02))
        ax.legend(fontsize=7)
    ax = axes[2]
    for drive, col in [('rc', 'k'), ('sin', 'tab:purple')]:
        sel = sorted([r for r in rows if r['model'] == 'fp' and r['drive'] == drive
                      and float(r['S_inj_density']) == 0.0],
                     key=lambda r: float(r['I_min_rel']))
        if sel:
            ax.semilogy([float(r['I_min_rel']) for r in sel],
                        [float(r['S_trough_top']) for r in sel], color=col, marker='o',
                        label=f'FP {drive}: top-line trough density')
    for s_inj, (col, _, lab) in styles.items():
        if s_inj > 0:
            ax.axhline(s_inj, color=col, ls=':', label=f'{lab} per-mode S_inj')
    ax.set(xlabel='I_min / I_th', ylabel='photon density (m$^{-3}$)',
           title='Residual lasing photons vs injected')
    ax.legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, FIG_PATH, dpi=150, bbox_inches=None, pad_inches=None)
    print('wrote', FIG_PATH)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['rc', 'sin', 'dfb', 'plot'])
    ap.add_argument('--mins', type=float, nargs='*')
    ap.add_argument('--inj', type=float, nargs='*')
    ap.add_argument('--seed', type=int, default=SEED)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.what == 'plot':
        plot()
        sys.exit()
    if args.what == 'dfb':
        for m in args.mins or RC_MINS:
            run_condition('dfb', 'rc', m, 0.0, args.seed)
    else:
        mins = args.mins or (RC_MINS if args.what == 'rc' else SIN_MINS)
        for m in mins:
            for s in (INJ_LEVELS if args.inj is None else args.inj):
                run_condition('fp', args.what, m, s, args.seed)
