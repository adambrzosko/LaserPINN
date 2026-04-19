"""
Pulse-to-pulse statistics of a gain-switched DFB laser with CW SLD injection.

Simulates ~100,000 consecutive gain-switched pulses from a 1550 nm DFB laser
(1 GHz rep rate by default) that receives coherent CW seed light from a
superluminescent diode (Lang-Kobayashi injection terms), with Langevin noise
on the carrier, photon and phase rate equations (Euler-Maruyama integration).

For every pulse the photon density is sampled at a single time offset
(the pulse peak, found from a deterministic burn-in), and we then plot
    (a) the probability distribution of the sampled intensities (histogram)
    (b) the pulse-to-pulse intensity autocorrelation  r(lag) =
        < (I_n - <I>)(I_{n+lag} - <I>) > / Var(I)

Performance note
----------------
100,000 pulses × 500 Euler-Maruyama steps per pulse = 5·10^7 scalar SDE
steps — prohibitive in pure Python.  We exploit numpy vectorisation across
K parallel independent trains (K × n_per_train = 100,000 total pulses).
With K=200, N_per_train=500 this runs in ~30-60 s.  Because the Langevin
noise is delta-correlated, the trains are statistically independent and
their sample-point histograms / autocorrelations can be safely pooled.

Usage
-----
Default (100k pulses, 1 GHz, 5× threshold peak, moderate injection):
    python gs_injected_statistics.py

Custom parameters (all optional):
    python gs_injected_statistics.py \\
        --n_pulses 100000 --n_trains 200 \\
        --f_rep 1e9 --duty 0.30 \\
        --I_bias_factor 0.9 --I_peak_factor 5.0 \\
        --I_sld 150e-3 --eta_coupling 0.10 --delta_nu 0.0 \\
        --sample_mode peak --seed 0 \\
        --prefix gs_inj_stats

Requires: numpy, scipy, matplotlib
"""

import argparse
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dfb_laser import DFBLaserParams, solve_transient, q, h, c
from sld_injection import (
    SLDParams, InjectionParams,
    solve_sld_steady_state, sld_to_injection_field,
)
from gain_switched_interference import (
    GainSwitchParams, make_gain_switch_current, simulate_pulse_train,
)


# ── Vectorised gain-switch current waveform ─────────────────────────────────

def gain_switch_current_vec(t, gs, I_th):
    """Same raised-cosine waveform as make_gain_switch_current, vectorised
    over an array of t (seconds).  Returns I(t) array of same shape."""
    I_off = gs.I_bias_factor * I_th
    I_on = gs.I_peak_factor * I_th
    T = gs.T_rep
    t_on = gs.t_on
    t_r = gs.t_rise

    phase = np.mod(t, T)
    I = np.full_like(phase, I_off, dtype=float)

    # Rising edge
    m = phase < t_r
    I[m] = I_off + (I_on - I_off) * 0.5 * (1 - np.cos(np.pi * phase[m] / t_r))

    # ON plateau
    m = (phase >= t_r) & (phase < t_on - t_r)
    I[m] = I_on

    # Falling edge
    m = (phase >= t_on - t_r) & (phase < t_on)
    I[m] = I_off + (I_on - I_off) * 0.5 * (
        1 - np.cos(np.pi * (t_on - phase[m]) / t_r)
    )

    return I


# ── Locate peak within a pulse period (deterministic burn-in) ──────────────

def find_peak_offset(laser, gs):
    """Run a short deterministic simulation (no noise, no injection) and
    return the time index inside one period at which S is maximal.

    Returns (idx_peak_in_period, pts_per_period).
    """
    # Use a dense deterministic run to find the peak reliably
    pts_per_period = 2000
    gs_short = GainSwitchParams(
        f_rep=gs.f_rep, duty=gs.duty,
        I_bias_factor=gs.I_bias_factor, I_peak_factor=gs.I_peak_factor,
        t_rise=gs.t_rise, n_periods=15, n_discard=10,
    )
    data = simulate_pulse_train(laser, gs_short, pts_per_period=pts_per_period)
    # Look at one steady-state period worth of data
    S_one = data['S'][:pts_per_period]
    idx_peak = int(np.argmax(S_one))
    return idx_peak, pts_per_period


# ── Vectorised Euler-Maruyama across K parallel trains ─────────────────────

def simulate_gs_injected_trains(
    laser, gs, kappa, S_inj, delta_omega,
    n_trains=200, n_pulses_per_train=500,
    dt=2e-12, sample_idx=None, seed=0,
):
    """Run K=n_trains independent gain-switched pulse trains in parallel
    using vectorised Euler-Maruyama with Lang-Kobayashi injection terms.

    Parameters
    ----------
    laser : DFBLaserParams
    gs : GainSwitchParams
    kappa : float           injection rate (s^-1)
    S_inj : float           injected intracavity photon density (m^-3)
    delta_omega : float     angular frequency detuning (rad/s)
    n_trains : int          K parallel independent realisations
    n_pulses_per_train : int  number of pulses each train emits
    dt : float              Euler-Maruyama step (s); 2 ps << tau_p (~3 ps)
    sample_idx : int or None  time-step index within each pulse to sample.
                            If None, sampled at the pulse peak (from burn-in).
    seed : int

    Returns
    -------
    dict with
        S_samples : ndarray (n_trains, n_pulses_per_train)
                    photon density at the sample offset in every pulse
        sample_idx : int  step index within a period where samples are taken
        runtime : float   seconds
    """
    I_th = laser.threshold_current()

    # Number of Euler-Maruyama steps per pulse period
    n_per_pulse = int(round(gs.T_rep / dt))
    if sample_idx is None:
        # Find peak offset from deterministic burn-in and map onto our dt grid
        idx_peak_dense, pts_dense = find_peak_offset(laser, gs)
        frac = idx_peak_dense / pts_dense  # position within period (0..1)
        sample_idx = int(round(frac * n_per_pulse))
        sample_idx = min(max(sample_idx, 0), n_per_pulse - 1)
    print(f"    sampling at step {sample_idx}/{n_per_pulse} "
          f"(t = {sample_idx*dt*1e12:.1f} ps within each {gs.T_rep*1e9:.1f} ns period)")

    # Pre-compute the current waveform over one full period (same each pulse)
    t_period = np.arange(n_per_pulse) * dt
    I_period = gain_switch_current_vec(t_period, gs, I_th)  # (n_per_pulse,)

    # Initial conditions: small jitter across trains so they don't all start
    # exactly on the deterministic attractor.  Burn-in still makes the noise
    # build-up from the SDE itself the dominant source of train-to-train
    # variation at the sample point.
    rng = np.random.default_rng(seed)
    N = 1.05 * laser.N_tr + 0.02 * laser.N_tr * rng.standard_normal(n_trains)
    S = 1e10 * np.exp(0.5 * rng.standard_normal(n_trains))
    phi = 2 * np.pi * rng.random(n_trains)

    # Pre-compute constants
    inv_qV = 1.0 / (q * laser.V)
    inv_tau_p = 1.0 / laser.tau_p
    Gvg = laser.Gamma * laser.v_g
    Gvga = Gvg * laser.a
    aH_half = 0.5 * laser.alpha_H
    sqrt_Sinj = np.sqrt(max(S_inj, 0.0))

    sqrt_dt = np.sqrt(dt)
    S_samples = np.empty((n_trains, n_pulses_per_train), dtype=float)

    # ── Burn-in: discard the first few pulses so the pulse train locks into
    # its periodic limit cycle before we start sampling.
    n_burn = 10
    total_pulses = n_burn + n_pulses_per_train

    t_start = time.time()
    for p in range(total_pulses):
        for k in range(n_per_pulse):
            # Brownian increments for each of the K trains
            dW_N = rng.standard_normal(n_trains) * sqrt_dt
            dW_S = rng.standard_normal(n_trains) * sqrt_dt
            dW_phi = rng.standard_normal(n_trains) * sqrt_dt

            Sk = np.maximum(S, 1.0)  # prevent division by zero / negative S
            Nk = N

            gk = laser.a * (Nk - laser.N_tr) / (1.0 + laser.epsilon * Sk)
            R_sp = (laser.A * Nk
                    + laser.B * Nk * Nk
                    + laser.C * Nk * Nk * Nk)
            R_st = Gvg * gk * Sk
            Bsp_N2 = laser.beta_sp * laser.B * Nk * Nk

            # Langevin diffusion coefficients (Gardiner)
            sig_N = np.sqrt(np.maximum(2.0 * R_sp, 0.0))
            sig_S = np.sqrt(np.maximum(2.0 * Bsp_N2, 0.0))
            sig_phi = np.sqrt(np.maximum(Bsp_N2 / (2.0 * Sk), 0.0))

            # Injection terms (Lang-Kobayashi, CW seed at phi_inj = 0)
            if sqrt_Sinj > 0.0:
                sqrt_S = np.sqrt(Sk)
                inj_S = 2.0 * kappa * sqrt_S * sqrt_Sinj * np.cos(phi)
                inj_phi = -delta_omega + kappa * (sqrt_Sinj / sqrt_S) * np.sin(phi)
            else:
                inj_S = 0.0
                inj_phi = -delta_omega

            I_val = I_period[k]

            dN = (I_val * inv_qV - R_sp - R_st) * dt + sig_N * dW_N
            dS = ((R_st - Sk * inv_tau_p) + Bsp_N2 + inj_S) * dt \
                 + sig_S * dW_S
            dphi = (aH_half * (Gvga * (Nk - laser.N_tr) - inv_tau_p)
                    + inj_phi) * dt + sig_phi * dW_phi

            N = np.maximum(N + dN, 0.0)
            S = np.maximum(Sk + dS, 1.0)
            phi = phi + dphi

            # Sample exactly once per pulse at the chosen offset
            if (p >= n_burn) and (k == sample_idx):
                S_samples[:, p - n_burn] = S

        if p == n_burn - 1:
            print(f"    burn-in ({n_burn} pulses) done, starting sampling...")
        if (p >= n_burn) and ((p - n_burn + 1) % max(1, n_pulses_per_train // 10) == 0):
            frac = (p - n_burn + 1) / n_pulses_per_train
            elapsed = time.time() - t_start
            print(f"    pulse {p - n_burn + 1}/{n_pulses_per_train} "
                  f"({100*frac:.0f}%)   elapsed {elapsed:.1f} s")

    runtime = time.time() - t_start
    return {
        'S_samples': S_samples,
        'sample_idx': sample_idx,
        'n_per_pulse': n_per_pulse,
        'dt': dt,
        'runtime': runtime,
    }


# ── Autocorrelation (FFT-based) ─────────────────────────────────────────────

def pulse_autocorrelation(S_samples, max_lag=None):
    """Unbiased pulse-to-pulse intensity autocorrelation,
    averaged over independent trains.

    Parameters
    ----------
    S_samples : ndarray (n_trains, n_pulses)
    max_lag : int or None   maximum lag to return (default n_pulses // 4)

    Returns
    -------
    lags : ndarray (max_lag+1,)
    r : ndarray (max_lag+1,)     normalised autocorrelation, r(0) = 1
    """
    K, Np = S_samples.shape
    if max_lag is None:
        max_lag = Np // 4

    # Zero-mean each train individually
    S0 = S_samples - S_samples.mean(axis=1, keepdims=True)
    var = S0.var(axis=1, keepdims=True)

    # FFT-based autocorrelation per train
    N_fft = int(2 ** np.ceil(np.log2(2 * Np)))
    X = np.fft.rfft(S0, n=N_fft, axis=1)
    acf = np.fft.irfft(X * np.conj(X), n=N_fft, axis=1).real[:, :Np]

    # Unbiased normalisation: divide by (Np - lag)
    counts = Np - np.arange(Np)
    acf_unbiased = acf / counts[None, :]

    # Normalise so r(0) = 1 per train, then average over trains
    acf_norm = acf_unbiased / np.maximum(var, 1e-300)
    r = acf_norm.mean(axis=0)

    return np.arange(max_lag + 1), r[: max_lag + 1]


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_distribution(S_free, S_inj, laser, gs, title_extra='',
                      prefix='gs_inj_stats', n_bins=120):
    """Intensity-distribution histogram: free-running vs injected.

    The two distributions typically have vastly different widths
    (free-running σ comes only from Langevin noise, which is strongly
    suppressed at a gain-clamped pulse peak; injected σ comes from
    Lang-Kobayashi coherent beating which converts phase diffusion into
    amplitude noise).  We therefore put each in its own panel with its
    own x-axis and y-axis scale, then add a third panel plotting both
    histograms together (each normalised to unit area in its own band)
    so the shapes can be compared directly.
    """
    P_free = laser.output_power(np.maximum(S_free, 0)).ravel() * 1e3  # mW
    P_inj = laser.output_power(np.maximum(S_inj, 0)).ravel() * 1e3

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        f'Gain-switched pulse-peak intensity distribution   '
        f'(N = {P_inj.size:,} pulses,  f$_{{rep}}$ = {gs.f_rep*1e-9:.1f} GHz)'
        + ('\n' + title_extra if title_extra else ''),
        fontsize=11,
    )

    for ax, P, label, color in [
        (axes[0], P_free, 'Free-running (Langevin only)', 'C0'),
        (axes[1], P_inj, 'With SLD injection (Lang-Kobayashi)', 'C3'),
    ]:
        if P.size == 0 or not np.isfinite(P).any():
            continue
        mean = np.mean(P)
        std = np.std(P)
        ax.hist(P, bins=n_bins, density=True, color=color, alpha=0.75,
                edgecolor='k', linewidth=0.3)
        ax.axvline(mean, color='k', ls='--', lw=1.0,
                   label=f'⟨P⟩ = {mean:.4f} mW')
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel('Peak output power (mW)')
        ax.set_ylabel('Probability density (mW$^{-1}$)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        ax.text(0.02, 0.95,
                f'σ / ⟨P⟩ = {std/mean*100:.3g} %\n'
                f'⟨P⟩ = {mean:.4f} mW\nσ = {std:.3g} mW',
                transform=ax.transAxes, va='top', ha='left',
                fontsize=9, bbox=dict(facecolor='white', alpha=0.8,
                                       edgecolor='gray'))

    # Third panel: centred & rescaled comparison  (x = (P - <P>) / σ)
    ax = axes[2]
    for P, label, color in [
        (P_free, 'Free-running',  'C0'),
        (P_inj,  'SLD injected',  'C3'),
    ]:
        mean = np.mean(P); std = np.std(P)
        if std <= 0 or not np.isfinite(std):
            continue
        z = (P - mean) / std
        ax.hist(z, bins=n_bins, density=True, color=color, alpha=0.45,
                edgecolor='k', linewidth=0.3, label=label)
    # Overlay unit-variance Gaussian reference
    zz = np.linspace(-4, 4, 500)
    ax.plot(zz, np.exp(-0.5 * zz**2) / np.sqrt(2 * np.pi),
            'k--', lw=1.0, label='N(0,1) Gaussian reference')
    ax.set_title('Centred & normalised shape comparison', fontsize=11,
                 fontweight='bold')
    ax.set_xlabel('(P − ⟨P⟩) / σ')
    ax.set_ylabel('Probability density')
    ax.set_xlim(-4.5, 4.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    fname = f'{prefix}_distribution.png'
    fig.savefig(fname, dpi=130)
    plt.close(fig)
    return fname


def plot_autocorrelation(S_free, S_inj, gs, title_extra='',
                         prefix='gs_inj_stats', max_lag=None):
    """Pulse-to-pulse autocorrelation: free-running vs injected.

    Two columns of panels: left zoomed to short lags (0..~20 pulses) where
    the physical correlation lives; right the full lag range on log-y to
    expose the statistical noise floor.
    """
    lags_f, r_f = pulse_autocorrelation(S_free, max_lag=max_lag)
    lags_i, r_i = pulse_autocorrelation(S_inj, max_lag=max_lag)
    lag_max_short = min(30, len(lags_f) - 1)

    # 1-sigma statistical noise floor for N_eff ≈ K * N_p independent samples
    K, Np = S_free.shape
    N_eff = K * Np
    sigma_floor = 1.0 / np.sqrt(N_eff)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(
        f'Pulse-to-pulse intensity autocorrelation   '
        f'r(Δ) = ⟨(I$_n$−⟨I⟩)(I$_{{n+Δ}}$−⟨I⟩)⟩ / Var(I)   '
        f'(averaged over {K} trains × {Np:,} pulses each,  '
        f'≈ {N_eff:,} samples)'
        + ('\n' + title_extra if title_extra else ''),
        fontsize=11,
    )

    # ── Left column: short lag zoom (physical correlation) ──
    ax = axes[0, 0]
    ax.plot(lags_f[:lag_max_short+1], r_f[:lag_max_short+1],
            'o-', color='C0', lw=1.4, ms=4, label='Free-running')
    ax.plot(lags_i[:lag_max_short+1], r_i[:lag_max_short+1],
            's-', color='C3', lw=1.4, ms=4, label='SLD injected')
    ax.axhline(0, color='k', ls=':', lw=0.7)
    ax.axhspan(-2*sigma_floor, 2*sigma_floor, color='gray', alpha=0.15,
               label=f'±2σ noise floor (±{2*sigma_floor:.1e})')
    ax.set_xlim(-0.5, lag_max_short + 0.5)
    ax.set_xlabel('Pulse lag Δ')
    ax.set_ylabel('r(Δ)')
    ax.set_title('Short-lag zoom (linear)', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)

    ax = axes[1, 0]
    ax.semilogy(lags_f[:lag_max_short+1], np.abs(r_f[:lag_max_short+1]) + 1e-12,
                'o-', color='C0', lw=1.4, ms=4, label='Free-running')
    ax.semilogy(lags_i[:lag_max_short+1], np.abs(r_i[:lag_max_short+1]) + 1e-12,
                's-', color='C3', lw=1.4, ms=4, label='SLD injected')
    ax.axhline(sigma_floor, color='gray', ls='--', lw=1.0,
               label=f'1σ noise floor ({sigma_floor:.1e})')
    ax.set_xlim(-0.5, lag_max_short + 0.5)
    ax.set_ylim(1e-6, 2)
    ax.set_xlabel('Pulse lag Δ')
    ax.set_ylabel('|r(Δ)|')
    ax.set_title('Short-lag zoom (log |r|)', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper right', fontsize=9)

    # ── Right column: full range (statistical noise floor visible) ──
    ax = axes[0, 1]
    ax.plot(lags_f, r_f, color='C0', lw=0.8, alpha=0.9, label='Free-running')
    ax.plot(lags_i, r_i, color='C3', lw=0.8, alpha=0.9, label='SLD injected')
    ax.axhline(0, color='k', ls=':', lw=0.7)
    ax.axhspan(-2*sigma_floor, 2*sigma_floor, color='gray', alpha=0.15)
    ax.set_xlabel(f'Pulse lag Δ   (1 unit = {gs.T_rep*1e9:.2f} ns)')
    ax.set_ylabel('r(Δ)')
    ax.set_title('Full range (linear)', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)

    ax = axes[1, 1]
    ax.semilogy(lags_f, np.abs(r_f) + 1e-12,
                color='C0', lw=0.8, alpha=0.9, label='Free-running')
    ax.semilogy(lags_i, np.abs(r_i) + 1e-12,
                color='C3', lw=0.8, alpha=0.9, label='SLD injected')
    ax.axhline(sigma_floor, color='gray', ls='--', lw=1.0,
               label=f'1σ noise floor ({sigma_floor:.1e})')
    ax.set_ylim(1e-6, 2)
    ax.set_xlabel(f'Pulse lag Δ   (1 unit = {gs.T_rep*1e9:.2f} ns)')
    ax.set_ylabel('|r(Δ)|')
    ax.set_title('Full range (log |r|)', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    fname = f'{prefix}_autocorrelation.png'
    fig.savefig(fname, dpi=130)
    plt.close(fig)
    return fname


# ── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Pulse-to-pulse statistics of SLD-injected gain-switched DFB.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Statistics
    p.add_argument('--n_pulses', type=int, default=100000,
                   help='total pulses (distributed across parallel trains)')
    p.add_argument('--n_trains', type=int, default=200,
                   help='parallel independent trains (vectorisation axis)')
    p.add_argument('--dt', type=float, default=2e-12,
                   help='Euler-Maruyama step (s); must be << tau_p')
    # Gain switching
    p.add_argument('--f_rep', type=float, default=1e9,
                   help='pulse repetition rate (Hz)')
    p.add_argument('--duty', type=float, default=0.30,
                   help='ON-state duty cycle (0..1)')
    p.add_argument('--I_bias_factor', type=float, default=0.9,
                   help='bias current / I_th during OFF state')
    p.add_argument('--I_peak_factor', type=float, default=5.0,
                   help='peak current / I_th during ON state')
    p.add_argument('--t_rise', type=float, default=50e-12,
                   help='current rise/fall time (s)')
    # SLD / injection
    p.add_argument('--I_sld', type=float, default=150e-3,
                   help='SLD drive current (A)')
    p.add_argument('--eta_coupling', type=float, default=0.10,
                   help='SLD → DFB coupling efficiency')
    p.add_argument('--delta_nu', type=float, default=0.0,
                   help='frequency detuning (Hz)')
    # Misc
    p.add_argument('--sample_mode', type=str, default='peak',
                   choices=['peak', 'center'],
                   help='where inside each pulse to sample: peak or center')
    p.add_argument('--seed', type=int, default=0, help='RNG seed')
    p.add_argument('--prefix', type=str, default='gs_inj_stats',
                   help='output filename prefix')
    return p.parse_args()


def main():
    args = parse_args()

    # n_trains × n_per_train = n_pulses  (round n_per_train up to integer)
    n_per_train = max(1, int(np.ceil(args.n_pulses / args.n_trains)))
    n_total = args.n_trains * n_per_train
    print(f'▪ Target pulses: {args.n_pulses:,}  → '
          f'{args.n_trains} trains × {n_per_train} pulses = {n_total:,}')

    # ── Laser & pulse parameters ──
    laser = DFBLaserParams()
    I_th = laser.threshold_current()
    print(f'▪ Laser: lambda = {laser.lambda0*1e9:.0f} nm,  '
          f'I_th = {I_th*1e3:.2f} mA,  tau_p = {laser.tau_p*1e12:.2f} ps')

    gs = GainSwitchParams(
        f_rep=args.f_rep, duty=args.duty,
        I_bias_factor=args.I_bias_factor, I_peak_factor=args.I_peak_factor,
        t_rise=args.t_rise,
    )
    print(f'▪ Gain switching: {gs.f_rep*1e-9:.1f} GHz,  duty {gs.duty:.0%},  '
          f'I_bias={gs.I_bias_factor:.2f} I_th, I_peak={gs.I_peak_factor:.1f} I_th')

    # ── SLD steady state + intracavity injection level ──
    sld = SLDParams()
    print(f'▪ SLD: lambda = {sld.lambda0*1e9:.0f} nm,  '
          f'I_sld = {args.I_sld*1e3:.0f} mA')
    sld_res = solve_sld_steady_state(sld, args.I_sld)
    P_sld = sld_res['P_out']
    print(f'    SLD output P = {P_sld*1e3:.2f} mW  '
          f'({"conv" if sld_res["converged"] else "NOT conv"} '
          f'in {sld_res["n_iter"]} iter)')

    inj = InjectionParams(eta_coupling=args.eta_coupling,
                          delta_nu=args.delta_nu)
    inj.compute_derived(laser)
    S_inj, _ = sld_to_injection_field(sld, P_sld, laser, inj)
    print(f'    kappa = {inj.kappa:.3e} s^-1,  '
          f'Delta omega = {inj.delta_omega:.3e} rad/s,  '
          f'S_inj = {S_inj:.3e} m^-3')

    # ── Sample offset within a pulse period ──
    if args.sample_mode == 'peak':
        sample_idx = None  # auto-detect from burn-in
    else:  # 'center'
        n_per_pulse = int(round(gs.T_rep / args.dt))
        sample_idx = int(round((gs.duty / 2.0) * n_per_pulse))

    # ── Run free-running (no injection) ──
    print('\n── Simulating FREE-RUNNING pulse train (Langevin only) ──')
    out_free = simulate_gs_injected_trains(
        laser, gs,
        kappa=inj.kappa, S_inj=0.0, delta_omega=0.0,
        n_trains=args.n_trains, n_pulses_per_train=n_per_train,
        dt=args.dt, sample_idx=sample_idx, seed=args.seed,
    )
    print(f'  free-running done in {out_free["runtime"]:.1f} s')
    sample_idx = out_free['sample_idx']  # reuse the same offset

    # ── Run SLD-injected ──
    print('\n── Simulating INJECTED pulse train (Langevin + Lang-Kobayashi) ──')
    out_inj = simulate_gs_injected_trains(
        laser, gs,
        kappa=inj.kappa, S_inj=S_inj, delta_omega=inj.delta_omega,
        n_trains=args.n_trains, n_pulses_per_train=n_per_train,
        dt=args.dt, sample_idx=sample_idx, seed=args.seed + 1,
    )
    print(f'  injection   done in {out_inj["runtime"]:.1f} s')

    # ── Plot ──
    title_extra = (
        f'I$_{{sld}}$ = {args.I_sld*1e3:.0f} mA,  '
        f'η = {args.eta_coupling:.2f},  '
        f'κ = {inj.kappa:.2e} s$^{{-1}}$,  '
        f'S$_{{inj}}$ = {S_inj:.2e} m$^{{-3}}$,  '
        f'Δν = {args.delta_nu*1e-9:.1f} GHz'
    )

    fname_dist = plot_distribution(
        out_free['S_samples'], out_inj['S_samples'],
        laser, gs, title_extra=title_extra, prefix=args.prefix,
    )
    print(f'\n✓ Wrote {fname_dist}')

    fname_acf = plot_autocorrelation(
        out_free['S_samples'], out_inj['S_samples'], gs,
        title_extra=title_extra, prefix=args.prefix,
        max_lag=min(500, n_per_train // 2),
    )
    print(f'✓ Wrote {fname_acf}')

    # Quick numerical summary
    def stats(tag, S):
        P = laser.output_power(np.maximum(S, 0)).ravel() * 1e3
        print(f'    {tag:<20}  ⟨P⟩ = {P.mean():.4f} mW   '
              f'σ = {P.std():.4f} mW   σ/⟨P⟩ = {P.std()/P.mean()*100:.3f} %')

    print('\n── Pulse-peak statistics ──')
    stats('Free-running', out_free['S_samples'])
    stats('SLD injected',  out_inj['S_samples'])


if __name__ == '__main__':
    main()
