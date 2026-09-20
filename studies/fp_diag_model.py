"""Why does the FP model randomise so much faster than the DFB model at 2 GHz?

Usage:  python studies/fp_diag_model.py {validate,decomp,seeds,fwm}

Both core solvers are re-implemented here as numba kernels that reproduce
core.fp_laser.solve_fp_stochastic and core.sld_injection.
solve_transient_injection_stochastic step for step, including the order in
which random numbers are drawn, so a given seed gives the same trajectory.
The kernels keep only one period of history at a time, so a 1500-pulse
21-mode run needs a few MB instead of ~1 GB.  `validate` checks the kernels
against core on a short run before anything else is trusted.

Phase estimators reported for every configuration (all r1 = |<exp(i dphi)>|):
  pw_raw     core.phase_estimators.r1_power_weighted on E at the total-power
             peak, exactly as gain_switch_fp returns it (the headline number).
  top_raw    same, for the mode with the highest mean peak power only.
  pw_demod   E_peak_j * exp(-i*2*pi*dnu_j*t_peak): removes the deterministic
             mode-offset rotation that gain_switch_fp leaves in E_j, so that
             peak-time jitter no longer turns into phase noise for dnu_j != 0.
  pw_ovl     per-mode fixed-delay overlap X_nj = sum_t E_j(t) E_j*(t - T) over
             one period: what an AMZI with delay T and a slow detector sees per
             line.  No peak sampling at all.
Each finished configuration is appended to images/fp_randomisation/diag_model.csv.
"""

import argparse
import csv
import os
import sys
import time

import numba as nb
import numpy as np

sys.path.insert(0, '.')

from core.dfb_laser import DFBLaserParams
from core.fp_laser import FPLaserParams, q
from core.phase_estimators import r1_power_weighted

OUT_DIR = 'images/fp_randomisation'
CSV_PATH = f'{OUT_DIR}/diag_model.csv'
DT = 0.5e-12
F_REP = 2e9
DUTY = 0.30
WARMUP = 20


# ── kernels ───────────────────────────────────────────────────────────────────

@nb.njit(cache=True)
def _fp_chunk(N, E, I, noise, dt, qV, a, N_tr, eps, Gamma, v_g, tau_p, alpha_H,
              A_nr, B, C_aug, beta_sp, G_j, rot, spont_gw, cns, fwm_scale, alpha_nl,
              E_hist, N_hist, S_hist):
    """One chunk of core.fp_laser.solve_fp_stochastic (S_inj = 0).

    E_hist[k] is the state before update k, i.e. core's E_arr[i0 + k].

    fwm_scale > 0 adds intermode four-wave mixing from fast (SHB / carrier
    heating) gain compression, which core lacks.  Expanding the instantaneous
    gain g_L (1 - eps_f |u|^2) of the total field u = sum_j E_j e^{i m_j Omega t}
    gives, for mode j, -(1/2)(1 + i alpha_nl) Gamma v_g g_j eps_f C_j with
    C_j = [|u|^2 u]_j = sum_{k-l+m=j} E_k E_l^* E_m.  The phase-insensitive part
    (2 S_total - |E_j|^2) E_j is dropped because core already has saturation;
    only the phase-sensitive remainder is added.  eps_f = fwm_scale * eps.
    The response of SHB (~0.1 ps) and CH (~0.5 ps) is taken as 1 at 135 GHz and
    carrier-density pulsation is neglected (Omega tau_c >> 1).  Mode spacing is
    exactly uniform in this model, so there is no dispersion to fight the
    locking: this is an upper bound.  C_j is evaluated exactly on a P = 4M
    point grid over one round trip (no aliasing for P >= 3M - 2).
    """
    n, M = E_hist.shape
    sqrt_half_dt = np.sqrt(dt / 2)
    Eold = np.empty(M, dtype=np.complex128)
    F = np.zeros(M, dtype=np.complex128)
    P = 4 * M
    ph = np.empty((M, P), dtype=np.complex128)
    for j in range(M):
        for p_ in range(P):
            ph[j, p_] = np.exp(1j * 2 * np.pi * (j - M // 2) * p_ / P)
    u = np.empty(P, dtype=np.complex128)
    for k in range(n):
        S_total = 0.0
        for j in range(M):
            S_total += E[j].real * E[j].real + E[j].imag * E[j].imag
            E_hist[k, j] = E[j]
            Eold[j] = E[j]
        N_hist[k] = N
        S_hist[k] = S_total
        if fwm_scale > 0.0:
            for p_ in range(P):
                acc = 0j
                for j in range(M):
                    acc += Eold[j] * ph[j, p_]
                u[p_] = (acc.real * acc.real + acc.imag * acc.imag) * acc
            for j in range(M):
                acc = 0j
                for p_ in range(P):
                    acc += u[p_] * np.conj(ph[j, p_])
                s_j = Eold[j].real * Eold[j].real + Eold[j].imag * Eold[j].imag
                F[j] = acc / P - (2 * S_total - s_j) * Eold[j]
        g_mat = a * (N - N_tr) / (1 + eps * S_total)
        R_sp = A_nr * N + B * N**2 + C_aug * N**3
        R_mode = beta_sp * B * N**2
        stim = 0.0
        for j in range(M):
            s_j = Eold[j].real * Eold[j].real + Eold[j].imag * Eold[j].imag
            g_j = g_mat * G_j[j]
            stim += g_j * s_j
            ng = 0.5 * (1 + 1j * alpha_H) * (Gamma * v_g * g_j - 1 / tau_p)
            e = Eold[j] * np.exp(ng * dt)
            e = e * rot[j]
            if fwm_scale > 0.0:
                e -= 0.5 * (1 + 1j * alpha_nl) * Gamma * v_g * g_j * fwm_scale * eps * F[j] * dt
            r_m = R_mode * G_j[j] if spont_gw else R_mode
            e += np.sqrt(max(r_m, 0.0)) * sqrt_half_dt * (noise[k, j] + 1j * noise[k, M + j])
            E[j] = e
        dN = (I[k] / qV - R_sp - Gamma * v_g * stim) * dt
        F_N = cns * np.sqrt(2 * max(R_sp, 0.0) * dt) * noise[k, 2 * M]
        N = N + dN + F_N
    return N


@nb.njit(cache=True)
def _dfb_chunk(N, E, I, noise, dt, qV, a, N_tr, eps, Gamma, v_g, tau_p, alpha_H,
               A, B, C, beta_sp, R_SLD, E_hist, N_hist, S_hist):
    """One chunk of core.sld_injection.solve_transient_injection_stochastic."""
    n = E_hist.shape[0]
    sqrt_half_dt = np.sqrt(dt / 2)
    for k in range(n):
        Sk = max(E.real * E.real + E.imag * E.imag, 1e-10)
        E_hist[k, 0] = E
        N_hist[k] = N
        S_hist[k] = Sk
        g = a * (N - N_tr) / (1 + eps * Sk)
        R_sp = A * N + B * N**2 + C * N**3
        R_mode = beta_sp * B * N**2
        ng = 0.5 * (1 + 1j * alpha_H) * (Gamma * v_g * g - 1 / tau_p)
        E = E * np.exp(ng * dt)
        E += np.sqrt(R_mode) * sqrt_half_dt * (noise[k, 0] + 1j * noise[k, 1])
        E += np.sqrt(R_SLD) * sqrt_half_dt * (noise[k, 2] + 1j * noise[k, 3])
        dN = (I[k] / qV - R_sp - Gamma * v_g * g * Sk) * dt
        F_N = np.sqrt(2 * R_sp * dt) * noise[k, 4]
        N = N + dN + F_N
    return N, E


# ── drive and runner ──────────────────────────────────────────────────────────

def raised_cosine(t, T, I_off, I_on, duty=DUTY):
    """Vectorised copy of the I_func in core.fp_laser.gain_switch_fp."""
    t_on = duty * T
    t_rise = min(20e-12, t_on / 4)
    ph = t % T
    out = np.full_like(t, I_off)
    m = ph < t_rise
    out[m] = I_off + (I_on - I_off) * 0.5 * (1 - np.cos(np.pi * ph[m] / t_rise))
    m = (ph >= t_rise) & (ph < t_on - t_rise)
    out[m] = I_on
    m = (ph >= t_on - t_rise) & (ph < t_on)
    out[m] = I_off + (I_on - I_off) * 0.5 * (1 + np.cos(np.pi * (ph[m] - t_on + t_rise) / t_rise))
    return out


def simulate(kind, p, n_pulses, seed, f_rep=F_REP, I_off_rel=0.9, I_on_rel=3.0,
             warmup=WARMUP, dt=DT, spont_gw=False, fwm_scale=0.0, alpha_nl=0.0,
             keep_periods=0):
    """Gain-switch one laser and return per-pulse quantities.

    kind 'fp' uses FPLaserParams, 'dfb' uses DFBLaserParams.
    keep_periods > 0 also returns the last few full periods for plotting.
    """
    I_th = p.threshold_current()
    T = 1 / f_rep
    spp = int(T / dt)
    t_eval = np.arange(0, (warmup + n_pulses) * T, dt)
    n_steps = len(t_eval)
    I_all = raised_cosine(t_eval, T, I_off_rel * I_th, I_on_rel * I_th)
    rng = np.random.default_rng(seed)
    qV = q * p.V

    if kind == 'fp':
        M = p.n_modes
        dlam = p.mode_wavelengths()
        G_j = p.gain_profile(dlam)
        dnu = p.mode_frequencies()
        rot = np.exp(1j * 2 * np.pi * dnu * dt)
        N = p.N_tr
        E = np.sqrt(1e10) * np.exp(1j * rng.uniform(0, 2 * np.pi, M))
        n_noise = 2 * M + 1
    else:
        M = 1
        dnu = np.zeros(1)
        N = p.N_tr
        E = complex(np.sqrt(1e10))
        n_noise = 5

    E_hist = np.zeros((spp, M), dtype=np.complex128)
    E_prev = np.zeros((spp, M), dtype=np.complex128)
    N_hist = np.zeros(spp)
    S_hist = np.zeros(spp)

    E_peak = np.zeros((n_pulses, M), dtype=complex)
    X_ovl = np.zeros((n_pulses, M), dtype=complex)
    W_pulse = np.zeros((n_pulses, M))
    S_turnon = np.zeros((n_pulses, M))
    t_peak = np.zeros(n_pulses)
    N_peak = np.zeros(n_pulses)
    N_turnon = np.zeros(n_pulses)
    S_peak_tot = np.zeros(n_pulses)
    kept = []

    for per in range(warmup + n_pulses):
        i0 = per * spp
        i1 = min(i0 + spp, n_steps)
        n = i1 - i0
        noise = rng.standard_normal((n, n_noise))
        Eh, Nh, Sh = E_hist[:n], N_hist[:n], S_hist[:n]
        if kind == 'fp':
            N = _fp_chunk(N, E, I_all[i0:i1], noise, dt, qV, p.a, p.N_tr, p.epsilon,
                          p.Gamma, p.v_g, p.tau_p, p.alpha_H, p.A_nr, p.B, p.C_aug,
                          p.beta_sp, G_j, rot, spont_gw, 1.0, fwm_scale, alpha_nl,
                          Eh, Nh, Sh)
        else:
            N, E = _dfb_chunk(N, E, I_all[i0:i1], noise, dt, qV, p.a, p.N_tr, p.epsilon,
                              p.Gamma, p.v_g, p.tau_p, p.alpha_H, p.A, p.B, p.C,
                              p.beta_sp, 0.0, Eh, Nh, Sh)
        k = per - warmup
        if k >= 0 and n == spp:
            ipk = int(np.argmax(Sh))
            E_peak[k] = Eh[ipk]
            t_peak[k] = t_eval[i0 + ipk]
            N_peak[k] = Nh[ipk]
            S_peak_tot[k] = Sh[ipk]
            N_turnon[k] = Nh[0]
            S_turnon[k] = np.abs(Eh[0])**2
            W_pulse[k] = np.sum(np.abs(Eh)**2, axis=0)
            X_ovl[k] = np.sum(Eh * np.conj(E_prev), axis=0)
            if per >= warmup + n_pulses - keep_periods:
                kept.append((t_eval[i0:i1].copy(), Eh.copy(), Nh.copy(), I_all[i0:i1].copy()))
        E_prev[:] = E_hist
    return dict(E_peak=E_peak, X_ovl=X_ovl, W_pulse=W_pulse, S_turnon=S_turnon,
                t_peak=t_peak, N_peak=N_peak, N_turnon=N_turnon, S_peak_tot=S_peak_tot,
                dnu=dnu, kept=kept, I_th=I_th, M=M)


# ── metrics ───────────────────────────────────────────────────────────────────

def per_mode_r1_from_phasors(Z):
    """|<exp(i arg Z)>| per column, for Z already a phase difference phasor."""
    u = Z / np.maximum(np.abs(Z), 1e-300)
    return np.abs(np.mean(u, axis=0))


def per_mode_r1(E):
    return np.abs(np.mean(np.exp(1j * np.diff(np.angle(E), axis=0)), axis=0))


def metrics(res, p, kind):
    E = res['E_peak']
    w = np.mean(np.abs(E)**2, axis=0)
    top = int(np.argmax(w))
    E_dm = E * np.exp(-1j * 2 * np.pi * res['dnu'][None, :] * res['t_peak'][:, None])
    r_raw = per_mode_r1(E)
    r_dm = per_mode_r1(E_dm)
    r_ov = per_mode_r1_from_phasors(res['X_ovl'])
    Wm = np.mean(res['W_pulse'], axis=0)
    dom = np.argmax(np.abs(E)**2, axis=1)
    spont_rate = p.beta_sp * p.B * np.mean(res['N_turnon'])**2
    ntr = p.N_tr
    g_th = (p.alpha_i + p.alpha_m) / p.Gamma
    N_th = ntr + g_th / p.a
    return {
        'r1_pw_raw': r1_power_weighted(E),
        'r1_top_raw': float(r_raw[top]),
        'r1_pw_demod': float(np.sum(r_dm * w) / np.sum(w)),
        'r1_top_demod': float(r_dm[top]),
        'r1_pw_ovl': float(np.sum(r_ov * Wm) / np.sum(Wm)),
        'r1_top_ovl': float(r_ov[int(np.argmax(Wm))]),
        'top_mode_idx': top,
        'top_power_frac': float(w[top] / np.sum(w)),
        'M_eff': float(np.sum(w)**2 / np.sum(w**2)),
        'dom_switch_frac': float(np.mean(np.diff(dom) != 0)) if E.shape[1] > 1 else 0.0,
        'tpk_jitter_ps': float(np.std(res['t_peak'] - np.round(res['t_peak'] * F_REP) / F_REP) * 1e12),
        'S_peak_tot': float(np.mean(res['S_peak_tot'])),
        'S_turnon_tot': float(np.mean(np.sum(res['S_turnon'], axis=1))),
        'S_turnon_top': float(np.mean(res['S_turnon'][:, top])),
        'N_turnon_over_Nth': float(np.mean(res['N_turnon']) / N_th),
        'spont_per_mode_per_tau_p': float(spont_rate * p.tau_p),
    }


def params_row(label, kind, p, n, seed, extra=None):
    row = {'label': label, 'kind': kind, 'n_pulses': n, 'seed': seed,
           'n_modes': getattr(p, 'n_modes', 1), 'R2': p.R2, 'beta_sp': p.beta_sp,
           'tau_p_ps': p.tau_p * 1e12, 'I_th_mA': p.threshold_current() * 1e3,
           'floor_random': 0.886 / np.sqrt(n)}
    if extra:
        row.update(extra)
    return row


def done_labels(path=CSV_PATH):
    if not os.path.exists(path):
        return set()
    with open(path, newline='') as f:
        return {r['label'] for r in csv.DictReader(f)}


def append_row(row, path=CSV_PATH):
    """Append one row; if the row brings new columns, rewrite the file with them."""
    if not os.path.exists(path):
        with open(path, 'w', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=list(row))
            wr.writeheader()
            wr.writerow(row)
        return
    with open(path, newline='') as f:
        rd = csv.DictReader(f)
        fields = list(rd.fieldnames)
        rows = list(rd)
    missing = [k for k in row if k not in fields]
    if missing:
        fields += missing
        with open(path, 'w', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=fields, restval='')
            wr.writeheader()
            wr.writerows(rows)
            wr.writerow(row)
        return
    with open(path, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=fields, restval='').writerow(row)


def run_config(label, kind, p, n=1500, seed=42, spont_gw=False, fwm_scale=0.0,
               alpha_nl=0.0, extra=None, skip=None):
    if skip is not None and label in skip:
        print(f'  {label}: done, skipping', flush=True)
        return
    t0 = time.time()
    res = simulate(kind, p, n, seed, spont_gw=spont_gw, fwm_scale=fwm_scale, alpha_nl=alpha_nl)
    row = params_row(label, kind, p, n, seed, extra)
    row.update({'spont_gw': spont_gw, 'fwm_scale': fwm_scale, 'alpha_nl': alpha_nl})
    row.update(metrics(res, p, kind))
    append_row(row)
    print(f"  {label:28s} pw_raw={row['r1_pw_raw']:.4f} top_raw={row['r1_top_raw']:.4f} "
          f"pw_dm={row['r1_pw_demod']:.4f} top_dm={row['r1_top_demod']:.4f} "
          f"pw_ovl={row['r1_pw_ovl']:.4f} top_ovl={row['r1_top_ovl']:.4f} "
          f"Meff={row['M_eff']:.2f} jit={row['tpk_jitter_ps']:.2f}ps "
          f"[{time.time()-t0:.0f}s]", flush=True)


# ── studies ───────────────────────────────────────────────────────────────────

def validate():
    """Kernels vs core solvers on a short run with the same seed."""
    from core.fp_laser import gain_switch_fp
    from core.sld_injection import solve_transient_injection_stochastic
    n = 12
    for M in (1, 5):
        fp = FPLaserParams(n_modes=M)
        I_th = fp.threshold_current()
        ref = gain_switch_fp(fp, F_REP, n, 0.9 * I_th, 3.0 * I_th, duty=DUTY, dt=DT,
                             seed=42, warmup_pulses=WARMUP)
        mine = simulate('fp', fp, n, 42)
        d = np.max(np.abs(ref['E_peak'] - mine['E_peak'])) / np.max(np.abs(ref['E_peak']))
        dt_pk = np.max(np.abs(ref['t_peak'] - mine['t_peak']))
        print(f'FP M={M}: max rel |dE_peak| = {d:.2e}, max |dt_peak| = {dt_pk:.1e} s, '
              f'r1_pw core {r1_power_weighted(ref["E_peak"]):.4f} kernel '
              f'{r1_power_weighted(mine["E_peak"]):.4f}')
    las = DFBLaserParams()
    I_th = las.threshold_current()
    T = 1 / F_REP
    t_total = (WARMUP + n) * T
    t_eval = np.arange(0, t_total, DT)
    I_off, I_on = 0.9 * I_th, 3.0 * I_th
    t_on = DUTY * T
    t_rise = min(20e-12, t_on / 4)

    def I_func(t):  # verbatim from studies/fp_validated_sweeps.dfb_point
        ph = t % T
        if ph < t_rise:
            return I_off + (I_on - I_off) * 0.5 * (1 - np.cos(np.pi * ph / t_rise))
        if ph < t_on - t_rise:
            return I_on
        if ph < t_on:
            return I_off + (I_on - I_off) * 0.5 * (
                1 + np.cos(np.pi * (ph - t_on + t_rise) / t_rise))
        return I_off

    I_arr = raised_cosine(t_eval, T, I_off, I_on)
    print('drive max |dI| (vectorised vs scalar) =',
          np.max(np.abs(I_arr - np.array([I_func(t) for t in t_eval]))))
    sol = solve_transient_injection_stochastic(las, I_func, inj=None, S_inj=0.0,
                                               sld_tau_coh=None, t_span=[0, t_total],
                                               t_eval=t_eval, seed=42)
    S, phi = sol.y[1], sol.y[2]
    steps = int(T / DT)
    E_ref = []
    for k in range(n):
        i0 = (WARMUP + k) * steps
        j = np.argmax(S[i0:i0 + steps]) + i0
        E_ref.append(np.sqrt(S[j]) * np.exp(1j * phi[j]))
    E_ref = np.array(E_ref)
    mine = simulate('dfb', las, n, 42)
    d = np.max(np.abs(E_ref - mine['E_peak'][:, 0])) / np.max(np.abs(E_ref))
    print(f'DFB: max rel |dE_peak| = {d:.2e}, r1 core {r1_power_weighted(E_ref):.4f} '
          f'kernel {r1_power_weighted(mine["E_peak"]):.4f}')


def decomp():
    skip = done_labels()
    run_config('dfb', 'dfb', DFBLaserParams(), skip=skip)
    for R2 in (0.32, 0.95):
        for M in (1, 3, 7, 21, 37):
            run_config(f'fp_M{M}_R2{R2}', 'fp', FPLaserParams(n_modes=M, R2=R2), skip=skip)
            if M > 1:
                run_config(f'fp_M{M}_R2{R2}_betaM', 'fp',
                           FPLaserParams(n_modes=M, R2=R2, beta_sp=1e-4 / M), skip=skip)
    for M in (21, 37):
        run_config(f'fp_M{M}_R20.32_spontgw', 'fp', FPLaserParams(n_modes=M),
                   spont_gw=True, skip=skip)


def seeds():
    skip = done_labels()
    for s in (7, 1234, 2025, 99):
        run_config(f'dfb_s{s}', 'dfb', DFBLaserParams(), seed=s, skip=skip)
        for M, R2 in ((1, 0.95), (1, 0.32), (21, 0.95), (21, 0.32)):
            run_config(f'fp_M{M}_R2{R2}_s{s}', 'fp', FPLaserParams(n_modes=M, R2=R2),
                       seed=s, skip=skip)
        for R2 in (0.95, 0.32):
            run_config(f'fp_M21_R2{R2}_betaM_s{s}', 'fp',
                       FPLaserParams(n_modes=21, R2=R2, beta_sp=1e-4 / 21), seed=s, skip=skip)


def fwm():
    skip = done_labels()
    for scale, a_nl in ((0.3, 0.0), (1.0, 0.0), (3.0, 0.0), (1.0, 3.0), (3.0, 3.0)):
        run_config(f'fp_M21_R20.32_fwm{scale}_anl{a_nl}', 'fp', FPLaserParams(),
                   fwm_scale=scale, alpha_nl=a_nl, skip=skip)


STUDIES = {'validate': validate, 'decomp': decomp, 'seeds': seeds, 'fwm': fwm}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('study', choices=STUDIES)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    STUDIES[args.study]()
