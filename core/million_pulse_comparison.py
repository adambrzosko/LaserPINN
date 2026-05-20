"""
1M-pulse correlation analysis: free-running vs SLD-injected, 1-10 GHz.

Uses numba JIT-compiled Euler-Maruyama solver for the complex field
E = sqrt(S)*exp(i*phi). Extracts per-pulse peak phase, peak power, and
sampled power without storing the full time series.

Produces:
  1. Phase coherence (r1 and std) vs repetition rate
  2. Phase autocorrelation vs pulse lag at selected frequencies
  3. Peak intensity distributions for each frequency
"""
import numpy as np
import time as _time
from numba import njit

from core.dfb_laser import DFBLaserParams, q, h, c
from core.sld_injection import (
    SLDParams, InjectionParams,
    solve_sld_steady_state, sld_to_injection_field,
)
from gsdfb import compute_r1, setup_plotting, save_fig


# ── Numba-jitted solver ─────────────────────────────────────────────────────

@njit
def simulate_pulses(n_pulses, n_discard, pts_period, dt,
                    I_off, I_on, t_on, t_rise,
                    V, Gamma, v_g, a_gain, N_tr, eps,
                    A, B, C_aug, tau_p, beta_sp, alpha_H, q_e,
                    S_inj, seed):
    np.random.seed(seed)
    sqrt_half_dt = np.sqrt(dt / 2.0)

    n_out = n_pulses - n_discard
    peak_phi = np.empty(n_out)
    peak_S = np.empty(n_out)
    samp_S = np.empty(n_out)
    peak_k = np.empty(n_out, dtype=np.int64)

    N = N_tr * 1.2
    Er = np.sqrt(1e10)
    Ei = 0.0

    R_SLD = S_inj / tau_p
    amp_sld = np.sqrt(R_SLD) * sqrt_half_dt if R_SLD > 0.0 else 0.0

    for p in range(n_pulses):
        best_S = 0.0
        best_phi = 0.0
        best_k = 0
        samp = 0.0
        sampled = False
        t_samp = t_on * 0.5

        for k in range(pts_period):
            t = k * dt

            if t < t_rise:
                f = 0.5 * (1.0 - np.cos(np.pi * t / t_rise))
                I_cur = I_off + (I_on - I_off) * f
            elif t < t_on - t_rise:
                I_cur = I_on
            elif t < t_on:
                f = 0.5 * (1.0 - np.cos(np.pi * (t_on - t) / t_rise))
                I_cur = I_off + (I_on - I_off) * f
            else:
                I_cur = I_off

            S = Er * Er + Ei * Ei
            if S < 1e-10:
                S = 1e-10

            g = a_gain * (N - N_tr) / (1.0 + eps * S)
            R_sp = A * N + B * N * N + C_aug * N * N * N
            R_sp_mode = beta_sp * B * N * N

            gamma = 0.5 * (Gamma * v_g * g - 1.0 / tau_p)

            Er_new = (1.0 + gamma * dt) * Er - gamma * alpha_H * dt * Ei
            Ei_new = (1.0 + gamma * dt) * Ei + gamma * alpha_H * dt * Er
            Er = Er_new
            Ei = Ei_new

            amp_sp = np.sqrt(R_sp_mode) * sqrt_half_dt
            Er += amp_sp * np.random.randn()
            Ei += amp_sp * np.random.randn()

            if amp_sld > 0.0:
                Er += amp_sld * np.random.randn()
                Ei += amp_sld * np.random.randn()

            dN = (I_cur / (q_e * V) - R_sp - Gamma * v_g * g * S) * dt
            FN = np.sqrt(2.0 * R_sp * dt) * np.random.randn()
            N += dN + FN

            S_now = Er * Er + Ei * Ei
            if S_now > best_S:
                best_S = S_now
                best_phi = np.arctan2(Ei, Er)
                best_k = k

            if not sampled and t >= t_samp:
                samp = S_now
                sampled = True

        if p >= n_discard:
            idx = p - n_discard
            peak_phi[idx] = best_phi
            peak_S[idx] = best_S
            samp_S[idx] = samp
            peak_k[idx] = best_k

    return peak_phi, peak_S, samp_S, peak_k


# ── Solver with custom waveform ──────────────────────────────────────────────

@njit
def simulate_pulses_waveform(n_pulses, n_discard, pts_period, dt,
                              I_waveform,
                              V, Gamma, v_g, a_gain, N_tr, eps,
                              A, B, C_aug, tau_p, beta_sp, alpha_H, q_e,
                              S_inj, seed):
    """Same physics as simulate_pulses but with an arbitrary current waveform.

    Parameters
    ----------
    I_waveform : 1-D array of length pts_period
        Current (A) at each time step within one period.  Replayed
        identically for every pulse.  Must be a contiguous float64 array.

    All other parameters identical to simulate_pulses.
    """
    np.random.seed(seed)
    sqrt_half_dt = np.sqrt(dt / 2.0)

    n_out = n_pulses - n_discard
    peak_phi = np.empty(n_out)
    peak_S = np.empty(n_out)
    samp_S = np.empty(n_out)
    peak_k = np.empty(n_out, dtype=np.int64)

    N = N_tr * 1.2
    Er = np.sqrt(1e10)
    Ei = 0.0

    R_SLD = S_inj / tau_p
    amp_sld = np.sqrt(R_SLD) * sqrt_half_dt if R_SLD > 0.0 else 0.0

    half_pts = pts_period // 2

    for p in range(n_pulses):
        best_S = 0.0
        best_phi = 0.0
        best_k = 0
        samp = 0.0
        sampled = False

        for k in range(pts_period):
            I_cur = I_waveform[k]

            S = Er * Er + Ei * Ei
            if S < 1e-10:
                S = 1e-10

            g = a_gain * (N - N_tr) / (1.0 + eps * S)
            R_sp = A * N + B * N * N + C_aug * N * N * N
            R_sp_mode = beta_sp * B * N * N

            gamma = 0.5 * (Gamma * v_g * g - 1.0 / tau_p)

            Er_new = (1.0 + gamma * dt) * Er - gamma * alpha_H * dt * Ei
            Ei_new = (1.0 + gamma * dt) * Ei + gamma * alpha_H * dt * Er
            Er = Er_new
            Ei = Ei_new

            amp_sp = np.sqrt(R_sp_mode) * sqrt_half_dt
            Er += amp_sp * np.random.randn()
            Ei += amp_sp * np.random.randn()

            if amp_sld > 0.0:
                Er += amp_sld * np.random.randn()
                Ei += amp_sld * np.random.randn()

            dN = (I_cur / (q_e * V) - R_sp - Gamma * v_g * g * S) * dt
            FN = np.sqrt(2.0 * R_sp * dt) * np.random.randn()
            N += dN + FN

            S_now = Er * Er + Ei * Ei
            if S_now > best_S:
                best_S = S_now
                best_phi = np.arctan2(Ei, Er)
                best_k = k

            if not sampled and k >= half_pts:
                samp = S_now
                sampled = True

        if p >= n_discard:
            idx = p - n_discard
            peak_phi[idx] = best_phi
            peak_S[idx] = best_S
            samp_S[idx] = samp
            peak_k[idx] = best_k

    return peak_phi, peak_S, samp_S, peak_k


# ── Waveform builders ────────────────────────────────────────────────────────

def build_raised_cosine(pts_period, dt, I_off, I_on, t_on, t_rise):
    """Original raised-cosine waveform (default)."""
    waveform = np.full(pts_period, I_off)
    for k in range(pts_period):
        t = k * dt
        if t < t_rise:
            f = 0.5 * (1.0 - np.cos(np.pi * t / t_rise))
            waveform[k] = I_off + (I_on - I_off) * f
        elif t < t_on - t_rise:
            waveform[k] = I_on
        elif t < t_on:
            f = 0.5 * (1.0 - np.cos(np.pi * (t_on - t) / t_rise))
            waveform[k] = I_off + (I_on - I_off) * f
    return waveform


def build_square(pts_period, dt, I_off, I_on, t_on):
    """Ideal square pulse (zero rise time)."""
    waveform = np.full(pts_period, I_off)
    for k in range(pts_period):
        if k * dt < t_on:
            waveform[k] = I_on
    return waveform


def build_gaussian(pts_period, dt, I_off, I_on, t_center, sigma):
    """Gaussian current pulse."""
    waveform = np.full(pts_period, I_off)
    for k in range(pts_period):
        t = k * dt
        waveform[k] = I_off + (I_on - I_off) * np.exp(
            -0.5 * ((t - t_center) / sigma)**2)
    return waveform


def build_fourier(pts_period, dt, I_off, I_on, t_on, coeffs):
    """Fourier-parameterised waveform.

    coeffs : array of shape (n_harmonics, 2)
        coeffs[h] = (a_h, b_h) for harmonics h=1,2,...
        The waveform is:
            I(t) = I_off + (I_on - I_off) * rect(t/t_on) * envelope(t)
        where:
            envelope(t) = clip[0.5 + sum_h a_h*cos(2*pi*h*t/t_on)
                                      + b_h*sin(2*pi*h*t/t_on), 0, 1]

    This allows smooth shaping of the "on" portion while keeping
    the off-state clean.
    """
    T_rep = pts_period * dt
    waveform = np.full(pts_period, I_off)
    for k in range(pts_period):
        t = k * dt
        if t < t_on:
            env = 0.5
            for h in range(len(coeffs)):
                freq_h = 2.0 * np.pi * (h + 1) * t / t_on
                env += coeffs[h, 0] * np.cos(freq_h)
                env += coeffs[h, 1] * np.sin(freq_h)
            env = max(0.0, min(1.0, env))
            waveform[k] = I_off + (I_on - I_off) * env
    return waveform


def build_trapezoid(pts_period, dt, I_off, I_on, t_on, t_rise, t_fall):
    """Trapezoid with independent rise and fall times."""
    waveform = np.full(pts_period, I_off)
    for k in range(pts_period):
        t = k * dt
        if t < t_rise:
            waveform[k] = I_off + (I_on - I_off) * (t / t_rise)
        elif t < t_on - t_fall:
            waveform[k] = I_on
        elif t < t_on:
            waveform[k] = I_off + (I_on - I_off) * ((t_on - t) / t_fall)
    return waveform


# ── Analysis helpers ─────────────────────────────────────────────────────────

def phase_autocorrelation(phi, max_lag=200):
    E = np.exp(1j * phi)
    n = len(E)
    n_pad = 2 * n
    Ef = np.fft.fft(E, n=n_pad)
    acf = np.fft.ifft(np.abs(Ef)**2)
    return np.abs(acf[:max_lag].real) / n


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    setup_plotting()

    print("=" * 70)
    print("  1M-Pulse Correlation: Free-Running vs SLD-Injected, 1-10 GHz")
    print("=" * 70)

    laser = DFBLaserParams()
    I_th = laser.threshold_current()
    sld = SLDParams()

    sld_result = solve_sld_steady_state(sld, 150e-3)
    P_sld = sld_result['P_out']
    inj = InjectionParams(eta_coupling=0.10)
    inj.compute_derived(laser)
    S_INJ, _ = sld_to_injection_field(sld, P_sld, laser, inj,
                                       acceptance_bandwidth=500e9)

    print(f"  Laser: I_th = {I_th*1e3:.1f} mA, tau_p = {laser.tau_p*1e12:.1f} ps")
    print(f"  SLD: P = {P_sld*1e3:.1f} mW, eta = 0.10, acc_bw = 500 GHz")
    print(f"  S_inj = {S_INJ:.2e} m^-3  (R_SLD/R_sp = "
          f"{S_INJ/laser.tau_p / (laser.beta_sp*laser.B*laser.N_tr**2):.1f}x)")

    N_PULSES = 1_000_000
    N_DISCARD = 1000
    DT = 1.0e-12
    DUTY = 0.30
    I_BIAS_FACTOR = 0.9
    I_PEAK_FACTOR = 5.0
    MAX_LAG = 200

    freqs = np.arange(1, 11) * 1e9
    I_off = I_BIAS_FACTOR * I_th
    I_on = I_PEAK_FACTOR * I_th

    # Warm up numba (first call compiles)
    print("\n  Compiling JIT solver...")
    t0 = _time.time()
    _warmup = simulate_pulses(
        100, 10, 100, 1e-12,
        I_off, I_on, 30e-12, 7e-12,
        laser.V, laser.Gamma, laser.v_g, laser.a, laser.N_tr, laser.epsilon,
        laser.A, laser.B, laser.C, laser.tau_p, laser.beta_sp, laser.alpha_H, q,
        0.0, 0)
    del _warmup
    print(f"  Compiled in {_time.time()-t0:.1f}s")

    all_results = {}
    total_t0 = _time.time()

    for f_rep in freqs:
        T_rep = 1.0 / f_rep
        t_on = DUTY * T_rep
        pts_period = max(int(round(T_rep / DT)), 50)
        dt_actual = T_rep / pts_period
        t_rise = min(20e-12, t_on / 4.0)

        for case, s_inj in [('free', 0.0), ('sld', S_INJ)]:
            label = f"{f_rep*1e-9:.0f}GHz_{case}"
            seed = 42 + int(f_rep * 1e-9) * 100 + (0 if case == 'free' else 1)

            print(f"\n  {label}: {N_PULSES/1e6:.0f}M pulses, "
                  f"{pts_period} pts/period ...", end="", flush=True)

            t0 = _time.time()
            phi, pk_S, sm_S, pk_k = simulate_pulses(
                N_PULSES + N_DISCARD, N_DISCARD, pts_period, dt_actual,
                I_off, I_on, t_on, t_rise,
                laser.V, laser.Gamma, laser.v_g, laser.a,
                laser.N_tr, laser.epsilon,
                laser.A, laser.B, laser.C,
                laser.tau_p, laser.beta_sp, laser.alpha_H, q,
                s_inj, seed)
            elapsed = _time.time() - t0

            pk_P = laser.output_power(np.maximum(pk_S, 0))
            sm_P = laser.output_power(np.maximum(sm_S, 0))

            r1, dphi = compute_r1(phi)
            std_dphi = np.std(dphi)

            acf = phase_autocorrelation(phi, MAX_LAG)

            # Coherence length: number of pulses before acf < 1/e
            coh_pulses = MAX_LAG
            for m in range(1, MAX_LAG):
                if acf[m] < acf[0] / np.e:
                    coh_pulses = m
                    break

            all_results[label] = dict(
                f_rep=f_rep, case=case,
                r1=r1, std_dphi=std_dphi, acf=acf,
                coh_pulses=coh_pulses,
                pk_P=pk_P, sm_P=sm_P,
                elapsed=elapsed,
            )

            print(f" {elapsed:.1f}s  r1={r1:.4f}  std={std_dphi:.3f}  "
                  f"coh={coh_pulses} pulses  <P>={np.mean(pk_P)*1e3:.2f}mW")

    total_elapsed = _time.time() - total_t0
    print(f"\n  Total simulation time: {total_elapsed:.0f}s "
          f"({total_elapsed/60:.1f} min)")

    # ── Figure 1: Phase coherence vs frequency ───────────────────────────

    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))
    fig1.suptitle(
        'Pulse-to-Pulse Phase Coherence — 1M Pulses per Point\n'
        f'I$_{{bias}}$={I_BIAS_FACTOR}×I$_{{th}}$, '
        f'I$_{{peak}}$={I_PEAK_FACTOR}×I$_{{th}}$, '
        f'duty={DUTY*100:.0f}%, '
        f'S$_{{inj}}$={S_INJ:.1e} m$^{{-3}}$',
        fontsize=12)

    f_ghz = freqs * 1e-9

    for case, color, marker, label in [
        ('free', 'C0', 'o', 'Free-running'),
        ('sld', 'C3', 's', 'SLD-injected'),
    ]:
        r1_vals = [all_results[f"{f:.0f}GHz_{case}"]['r1'] for f in f_ghz]
        std_vals = [all_results[f"{f:.0f}GHz_{case}"]['std_dphi'] for f in f_ghz]
        coh_vals = [all_results[f"{f:.0f}GHz_{case}"]['coh_pulses'] for f in f_ghz]

        axes1[0].plot(f_ghz, r1_vals, f'{marker}-', color=color, lw=2, ms=7,
                      label=label)
        axes1[1].plot(f_ghz, std_vals, f'{marker}-', color=color, lw=2, ms=7,
                      label=label)
        axes1[2].plot(f_ghz, coh_vals, f'{marker}-', color=color, lw=2, ms=7,
                      label=label)

    axes1[0].set_ylabel('Phase correlation |$\\langle e^{i\\Delta\\phi}\\rangle$|')
    axes1[0].set_ylim(-0.05, 1.05)
    axes1[0].axhline(0, color='gray', ls=':', alpha=0.5)

    axes1[1].set_ylabel('Phase std $\\sigma_{\\Delta\\phi}$ (rad)')
    axes1[1].axhline(np.pi / np.sqrt(3), color='gray', ls=':', alpha=0.5,
                     label=f'Uniform: {np.pi/np.sqrt(3):.2f}')
    axes1[1].set_ylim(0, 2.2)

    axes1[2].set_ylabel('Coherence length (pulses)')
    axes1[2].set_yscale('log')

    for ax in axes1:
        ax.set_xlabel('Repetition rate (GHz)')
        ax.set_xticks(f_ghz)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig1.savefig('images/1M_pulse_comparison/1M_phase_coherence_vs_freq.png', dpi=150)
    print("  Saved: images/1M_pulse_comparison/1M_phase_coherence_vs_freq.png")

    # ── Figure 2: Phase autocorrelation at selected frequencies ──────────

    sel_freqs = [1, 3, 5, 7, 10]
    fig2, axes2 = plt.subplots(2, 5, figsize=(22, 8), sharey='row')
    fig2.suptitle(
        'Pulse-to-Pulse Phase Autocorrelation — '
        '$|\\langle E^*_n E_{n+m}\\rangle|$ / $\\langle|E|^2\\rangle$',
        fontsize=13)

    lags = np.arange(MAX_LAG)
    for col, fg in enumerate(sel_freqs):
        for row, (case, color, label) in enumerate([
            ('free', 'C0', 'Free-running'),
            ('sld', 'C3', 'SLD-injected'),
        ]):
            key = f"{fg}GHz_{case}"
            acf = all_results[key]['acf']
            coh = all_results[key]['coh_pulses']

            axes2[row, col].plot(lags, acf, color=color, lw=1.2)
            axes2[row, col].axhline(acf[0] / np.e, color='gray', ls='--',
                                     alpha=0.5)
            if coh < MAX_LAG:
                axes2[row, col].axvline(coh, color='orange', ls=':', alpha=0.7)
            axes2[row, col].set_title(
                f'{fg} GHz — {label}\ncoh = {coh} pulses',
                fontsize=10)
            axes2[row, col].grid(True, alpha=0.3)
            axes2[row, col].set_ylim(-0.05, 1.05)

        axes2[1, col].set_xlabel('Lag (pulses)')

    axes2[0, 0].set_ylabel('|g$^{(1)}_{pulse}$(m)|')
    axes2[1, 0].set_ylabel('|g$^{(1)}_{pulse}$(m)|')

    plt.tight_layout()
    fig2.savefig('images/1M_pulse_comparison/1M_phase_autocorrelation.png', dpi=150)
    print("  Saved: images/1M_pulse_comparison/1M_phase_autocorrelation.png")

    # ── Figure 3: Peak intensity distributions ───────────────────────────

    fig3, axes3 = plt.subplots(2, 5, figsize=(22, 8))
    fig3.suptitle(
        'Peak Pulse Intensity Distributions — 1M Pulses\n'
        'Top: Free-running    Bottom: SLD-injected',
        fontsize=13)

    for col, fg in enumerate(sel_freqs):
        for row, (case, color, label) in enumerate([
            ('free', 'C0', 'Free-running'),
            ('sld', 'C3', 'SLD-injected'),
        ]):
            key = f"{fg}GHz_{case}"
            pk_P = all_results[key]['pk_P']

            mean_P = np.mean(pk_P)
            std_P = np.std(pk_P)
            cv = std_P / mean_P

            ax = axes3[row, col]
            ax.hist(pk_P * 1e3, bins=200, density=True, color=color, alpha=0.7,
                    edgecolor='none')
            ax.axvline(mean_P * 1e3, color='k', ls='--', lw=1, alpha=0.7)
            ax.set_title(
                f'{fg} GHz — {label}\n'
                f'$\\langle P\\rangle$={mean_P*1e3:.2f} mW, CV={cv:.4f}',
                fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Peak power (mW)')

    axes3[0, 0].set_ylabel('Probability density')
    axes3[1, 0].set_ylabel('Probability density')

    plt.tight_layout()
    fig3.savefig('images/1M_pulse_comparison/1M_intensity_distributions.png', dpi=150)
    print("  Saved: images/1M_pulse_comparison/1M_intensity_distributions.png")

    # ── Figure 4: Sampled intensity distributions (mid-pulse) ────────────

    fig4, axes4 = plt.subplots(2, 5, figsize=(22, 8))
    fig4.suptitle(
        'Mid-Pulse Sampled Intensity Distributions — 1M Pulses\n'
        'Sampled at $t = t_{on}/2$ within each period',
        fontsize=13)

    for col, fg in enumerate(sel_freqs):
        for row, (case, color, label) in enumerate([
            ('free', 'C0', 'Free-running'),
            ('sld', 'C3', 'SLD-injected'),
        ]):
            key = f"{fg}GHz_{case}"
            sm_P = all_results[key]['sm_P']

            sm_P_pos = sm_P[sm_P > 0]
            if len(sm_P_pos) == 0:
                continue
            mean_P = np.mean(sm_P_pos)
            std_P = np.std(sm_P_pos)
            cv = std_P / mean_P

            ax = axes4[row, col]
            ax.hist(sm_P_pos * 1e3, bins=200, density=True, color=color,
                    alpha=0.7, edgecolor='none')
            ax.axvline(mean_P * 1e3, color='k', ls='--', lw=1, alpha=0.7)
            ax.set_title(
                f'{fg} GHz — {label}\n'
                f'$\\langle P\\rangle$={mean_P*1e3:.2f} mW, CV={cv:.4f}',
                fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Sampled power (mW)')

    axes4[0, 0].set_ylabel('Probability density')
    axes4[1, 0].set_ylabel('Probability density')

    plt.tight_layout()
    fig4.savefig('images/1M_pulse_comparison/1M_sampled_distributions.png', dpi=150)
    print("  Saved: images/1M_pulse_comparison/1M_sampled_distributions.png")

    # ── Summary table ────────────────────────────────────────────────────

    plt.close('all')

    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"  {'f_rep':>5s}  {'':10s}  {'r1':>6s}  {'std_dphi':>8s}  "
          f"{'coh_len':>7s}  {'<P_pk>':>8s}  {'CV_pk':>7s}")
    print(f"  {'(GHz)':>5s}  {'case':10s}  {'':>6s}  {'(rad)':>8s}  "
          f"{'(pulse)':>7s}  {'(mW)':>8s}  {'':>7s}")
    print("  " + "-" * 66)

    for f in f_ghz:
        for case in ['free', 'sld']:
            key = f"{f:.0f}GHz_{case}"
            r = all_results[key]
            pk_P = r['pk_P']
            cv = np.std(pk_P) / np.mean(pk_P)
            print(f"  {f:5.0f}  {case:10s}  {r['r1']:6.4f}  "
                  f"{r['std_dphi']:8.3f}  {r['coh_pulses']:7d}  "
                  f"{np.mean(pk_P)*1e3:8.2f}  {cv:7.4f}")

    print("\n" + "=" * 70)
    print(f"  Done. Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print("=" * 70)
