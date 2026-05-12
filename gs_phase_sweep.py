"""
Phase correlation analysis: gain-switched DFB + CW SLD injection, 1–10 GHz.

For each repetition rate f_rep = 1, 2, ..., 10 GHz:
  1. Simulate 1,000,000 gain-switched pulses via vectorised Euler-Maruyama
     across K=200 parallel independent realisations (5,000 pulses each).
  2. Sample photon density S and accumulated phase φ at the pulse-intensity
     peak for each pulse — producing arrays of shape (200, 5000).
  3. Compute:
       • Intensity distribution  P(S)
       • Pulse-to-pulse intensity autocorrelation  r(Δ) = Cov(Iₙ,Iₙ₊Δ)/Var(I)
       • Pulse-to-pulse phase coherence  |g⁽¹⁾(Δ)| = |⟨e^{i(φₙ−φₙ₊Δ)}⟩|
  4. Collect per-frequency summary statistics.

Two conditions are compared at every frequency:
  • Free-running   (Langevin spontaneous-emission noise only)
  • CW SLD injected (Lang-Kobayashi + Langevin)

Output: gs_phase_sweep_report.pdf  (12 pages: title + 10 frequency pages + summary)

Runtime: ~6-10 minutes on a modern CPU.
"""

import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

from dfb_laser import DFBLaserParams, q, h, c
from sld_injection import (
    SLDParams, InjectionParams,
    solve_sld_steady_state, sld_to_injection_field,
)
from gain_switched_interference import GainSwitchParams, simulate_pulse_train

# ── matplotlib style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'font.size': 9,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'figure.dpi': 120,
})

COLOR_FREE = '#2166AC'   # blue
COLOR_INJ  = '#D6604D'   # red-orange


# ── Vectorised gain-switch current waveform ───────────────────────────────────

def gs_current_vec(t, gs, I_th):
    """Raised-cosine gain-switch current I(t), vectorised over t array."""
    I_off = gs.I_bias_factor * I_th
    I_on  = gs.I_peak_factor * I_th
    T, t_on, t_r = gs.T_rep, gs.t_on, gs.t_rise
    phase = np.mod(t, T)
    I = np.full_like(phase, I_off, dtype=float)
    m = phase < t_r
    I[m] = I_off + (I_on - I_off) * 0.5 * (1 - np.cos(np.pi * phase[m] / t_r))
    m = (phase >= t_r) & (phase < t_on - t_r)
    I[m] = I_on
    m = (phase >= t_on - t_r) & (phase < t_on)
    I[m] = I_off + (I_on - I_off) * 0.5 * (1 - np.cos(np.pi * (t_on - phase[m]) / t_r))
    return I


def find_peak_offset(laser, gs, n_periods=15, n_discard=10, pts=2000):
    """Deterministic burn-in to locate the pulse-peak time index
    (fraction within one period), so all frequencies sample the same
    physical point (intensity maximum).  Returns sample_idx in [0, pts-1]."""
    gs_short = GainSwitchParams(
        f_rep=gs.f_rep, duty=gs.duty,
        I_bias_factor=gs.I_bias_factor, I_peak_factor=gs.I_peak_factor,
        t_rise=gs.t_rise, n_periods=n_periods, n_discard=n_discard,
    )
    data = simulate_pulse_train(laser, gs_short, pts_per_period=pts)
    S_one = data['S'][:pts]
    return int(np.argmax(S_one))


# ── Euler-Maruyama simulation ────────────────────────────────────────────────

def run_simulation(laser, gs, kappa, S_inj, delta_omega,
                   n_trains, n_per_train, dt, sample_idx_frac, seed,
                   verbose_label=''):
    """
    Run K=n_trains parallel gain-switched pulse trains with Langevin noise
    and optional Lang-Kobayashi injection.

    Parameters
    ----------
    laser : DFBLaserParams
    gs : GainSwitchParams
    kappa : float           injection rate (s⁻¹)
    S_inj : float           injected photon density (m⁻³)
    delta_omega : float     frequency detuning (rad s⁻¹)
    n_trains : int          K parallel realisations
    n_per_train : int       pulses to sample per train
    dt : float              Euler-Maruyama step (s)
    sample_idx_frac : float fraction within period [0,1) to sample
    seed : int

    Returns
    -------
    S_samples   (n_trains, n_per_train)  photon density at sample point
    phi_samples (n_trains, n_per_train)  accumulated optical phase at sample
    sample_t_ps float                   sample time in ps within period
    """
    I_th = laser.threshold_current()
    n_per_pulse = max(1, int(round(gs.T_rep / dt)))
    sample_idx = min(int(round(sample_idx_frac * n_per_pulse)), n_per_pulse - 1)
    sample_t_ps = sample_idx * dt * 1e12

    # Pre-compute current waveform for one period (same every pulse)
    t_period = np.arange(n_per_pulse) * dt
    I_period = gs_current_vec(t_period, gs, I_th)

    rng = np.random.default_rng(seed)
    # Slightly jittered initial conditions (they converge to limit cycle during burn-in)
    N   = 1.05 * laser.N_tr + 0.02 * laser.N_tr * rng.standard_normal(n_trains)
    S   = 1e10 * np.exp(0.5 * rng.standard_normal(n_trains))
    phi = 2 * np.pi * rng.random(n_trains)

    # Pre-compute constants
    inv_qV   = 1.0 / (q * laser.V)
    inv_tau  = 1.0 / laser.tau_p
    Gvg      = laser.Gamma * laser.v_g
    Gvga     = Gvg * laser.a
    aH2      = 0.5 * laser.alpha_H
    sqrt_Si  = np.sqrt(max(S_inj, 0.0))
    sqrt_dt  = np.sqrt(dt)

    S_samples   = np.empty((n_trains, n_per_train), dtype=np.float64)
    phi_samples = np.empty((n_trains, n_per_train), dtype=np.float64)

    n_burn = 15          # discard first N_burn pulses (transient)
    total  = n_burn + n_per_train
    t0 = time.time()

    for p in range(total):
        # Pre-allocate noise for all steps in this pulse at once (faster than
        # repeated small allocations)
        dW = rng.standard_normal((3, n_per_pulse, n_trains)) * sqrt_dt

        for k in range(n_per_pulse):
            Sk = np.maximum(S, 1.0)
            Nk = N
            gk    = laser.a * (Nk - laser.N_tr) / (1.0 + laser.epsilon * Sk)
            R_sp  = laser.A * Nk + laser.B * Nk**2 + laser.C * Nk**3
            R_st  = Gvg * gk * Sk
            BN2   = laser.beta_sp * laser.B * Nk**2

            sig_N   = np.sqrt(np.maximum(2.0 * R_sp, 0.0))
            sig_S   = np.sqrt(np.maximum(2.0 * BN2, 0.0))
            sig_phi = np.sqrt(np.maximum(BN2 / (2.0 * Sk), 0.0))

            if sqrt_Si > 0.0:
                sqS   = np.sqrt(Sk)
                inj_S = 2.0 * kappa * sqS * sqrt_Si * np.cos(phi)
                inj_p = -delta_omega + kappa * (sqrt_Si / sqS) * np.sin(phi)
            else:
                inj_S = 0.0
                inj_p = -delta_omega

            I_val = I_period[k]
            dN  = (I_val * inv_qV - R_sp - R_st) * dt  + sig_N   * dW[0, k]
            dS  = ((R_st - Sk * inv_tau) + BN2 + inj_S) * dt + sig_S   * dW[1, k]
            dphi = (aH2 * (Gvga * (Nk - laser.N_tr) - inv_tau) + inj_p) * dt \
                   + sig_phi * dW[2, k]

            N   = np.maximum(N + dN, 0.0)
            S   = np.maximum(Sk + dS, 1.0)
            phi = phi + dphi

            if (p >= n_burn) and (k == sample_idx):
                idx = p - n_burn
                S_samples[:, idx]   = S
                phi_samples[:, idx] = phi

        if verbose_label and (p == n_burn - 1):
            print(f'    [{verbose_label}] burn-in done, sampling...')
        if verbose_label and (p >= n_burn):
            done = p - n_burn + 1
            if done % max(1, n_per_train // 5) == 0:
                pct = 100 * done / n_per_train
                print(f'    [{verbose_label}] {done}/{n_per_train} '
                      f'pulses ({pct:.0f}%)  {time.time()-t0:.1f}s')

    return S_samples, phi_samples, sample_t_ps


# ── Statistical functions ─────────────────────────────────────────────────────

def intensity_autocorr(S, max_lag=100):
    """Unbiased FFT intensity autocorrelation, averaged over trains.
    Returns (lags, r) with r(0)=1."""
    K, Np = S.shape
    S0  = S - S.mean(axis=1, keepdims=True)
    var = S0.var(axis=1, keepdims=True).mean()      # mean variance
    N2  = int(2**np.ceil(np.log2(2 * Np)))
    X   = np.fft.rfft(S0, n=N2, axis=1)
    acf = np.fft.irfft(X * np.conj(X), n=N2, axis=1).real[:, :Np]
    cnt = Np - np.arange(Np)
    r   = (acf / cnt[None, :]).mean(axis=0) / var
    ml  = min(max_lag, Np - 1)
    return np.arange(ml + 1), r[:ml + 1]


def phase_coherence(phi, max_lag=100):
    """|g⁽¹⁾(Δ)| = |⟨exp(i(φₙ−φₙ₊Δ))⟩| averaged over trains and pulse index n.
    Computed as the normalised complex autocorrelation of the phasor field."""
    K, Np = phi.shape
    Z  = np.exp(1j * phi)          # (K, Np) complex, bounded ∈ unit circle
    N2 = int(2**np.ceil(np.log2(2 * Np)))
    X  = np.fft.fft(Z, n=N2, axis=1)
    # ifft of |X|^2 gives complex autocorrelation of Z
    acf = np.fft.ifft(np.abs(X)**2, axis=1)[:, :Np]
    cnt = Np - np.arange(Np)
    g1  = (acf / cnt[None, :]).mean(axis=0)   # complex, averaged over trains
    g1  = np.abs(g1)
    g1  /= g1[0]                               # normalise: g1(0)=1
    ml  = min(max_lag, Np - 1)
    return np.arange(ml + 1), g1[:ml + 1]


def coherence_length(lags, g1, threshold=1.0 / np.e):
    """First lag where |g⁽¹⁾| drops below threshold (1/e by default)."""
    idx = np.argmax(g1 < threshold)
    return int(lags[idx]) if idx > 0 else int(lags[-1])


# ── PDF page builders ──────────────────────────────────────────────────────────

def _despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _add_stat_box(ax, text, loc='upper left'):
    ax.text(0.97 if 'right' in loc else 0.03,
            0.97,
            text,
            transform=ax.transAxes,
            va='top', ha='right' if 'right' in loc else 'left',
            fontsize=7,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      alpha=0.85, edgecolor='#aaaaaa', linewidth=0.6))


def build_title_page(laser, sld, inj, S_inj, n_trains, n_per_train, dt):
    """A text-only title/parameter page."""
    fig = plt.figure(figsize=(11.69, 8.27))   # A4 landscape
    fig.patch.set_facecolor('white')
    ax  = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    # Title block
    ax.text(0.5, 0.93, 'Phase Correlation Analysis of a Gain-Switched DFB Laser',
            ha='center', va='top', fontsize=17, fontweight='bold')
    ax.text(0.5, 0.87,
            'Effect of repetition rate on pulse-to-pulse intensity and phase coherence\n'
            'with and without CW SLD injection  (1 – 10 GHz, step 1 GHz)',
            ha='center', va='top', fontsize=12, color='#444444')

    # Divider line
    ax.axhline(0.83, xmin=0.08, xmax=0.92, color='#888888', linewidth=0.8)

    # Two-column parameter tables
    col_x = [0.10, 0.55]
    y0 = 0.78

    def tbl(title, rows, cx, y_start):
        ax.text(cx, y_start, title, fontsize=10, fontweight='bold',
                color='#1a1a6e', va='top')
        for i, (lbl, val) in enumerate(rows):
            y = y_start - 0.045 - i * 0.038
            ax.text(cx + 0.01,    y, lbl, va='top', fontsize=8.5,
                    color='#333333')
            ax.text(cx + 0.19, y, val, va='top', fontsize=8.5,
                    color='#000000', fontweight='bold')

    I_th = laser.threshold_current()
    tbl('DFB Laser  (1550 nm, InGaAsP/InP)', [
        ('Wavelength',         f'{laser.lambda0*1e9:.0f} nm'),
        ('Cavity length',      f'{laser.L*1e6:.0f} µm'),
        ('Threshold current',  f'{I_th*1e3:.2f} mA'),
        ('Photon lifetime τₚ', f'{laser.tau_p*1e12:.2f} ps'),
        ('Henry α factor',     f'{laser.alpha_H:.1f}'),
        ('Diff. gain  a',      f'{laser.a:.2e} m²'),
        ('Gain compression ε', f'{laser.epsilon:.2e} m³'),
        ('Spont. coupling β',  f'{laser.beta_sp:.1e}'),
    ], col_x[0], y0)

    sld_res = solve_sld_steady_state(sld, 150e-3)
    tbl('SLD Injection  (CW, broadband)', [
        ('SLD wavelength',        f'{sld.lambda0*1e9:.0f} nm'),
        ('SLD current',           '150 mA'),
        ('SLD output power',      f'{sld_res["P_out"]*1e3:.2f} mW'),
        ('Coupling efficiency η', f'{inj.eta_coupling:.2f}'),
        ('Injection rate κ',      f'{inj.kappa:.3e} s⁻¹'),
        ('Injected density S_inj',f'{S_inj:.3e} m⁻³'),
        ('Frequency detuning Δν', '0 GHz'),
    ], col_x[1], y0)

    tbl('Simulation Parameters', [
        ('Integration method',   'Euler-Maruyama (SDE)'),
        ('Time step dt',         f'{dt*1e12:.0f} ps'),
        ('Parallel trains K',    f'{n_trains}'),
        ('Pulses per train',     f'{n_per_train:,}'),
        ('Total pulses',         f'{n_trains*n_per_train:,}'),
        ('Bias  I_bias',         '0.9 × I_th'),
        ('Peak  I_peak',         '5.0 × I_th'),
        ('Rise time t_rise',     '5% of T_rep'),
    ], col_x[0], y0 - 0.43)

    tbl('Noise (Langevin / Gardiner)', [
        ('Carrier noise σ_N',    '√(2 R_sp)  per √s'),
        ('Photon noise σ_S',     '√(2 β B N²)  per √s'),
        ('Phase noise σ_φ',      '√(β B N² / 2S)  per √s'),
        ('Injection (free-run)', 'κ = 0,  S_inj = 0'),
        ('Injection (seeded)',   'κ as above,  S_inj as above'),
        ('Lang-Kobayashi term',  '2κ√(S·S_inj) cos φ  in dS/dt'),
    ], col_x[1], y0 - 0.43)

    ax.axhline(0.06, xmin=0.08, xmax=0.92, color='#cccccc', linewidth=0.5)
    ax.text(0.5, 0.035,
            'Sampling: photon density S and accumulated phase φ are recorded at '
            'the pulse-intensity peak (located via a deterministic burn-in) '
            'for each of the 1,000,000 simulated pulses.',
            ha='center', va='bottom', fontsize=8, color='#555555', style='italic')
    return fig


def build_frequency_page(f_rep_GHz, laser,
                         S_f, phi_f, r_f_lags, r_f, g1_f_lags, g1_f,
                         S_i, phi_i, r_i_lags, r_i, g1_i_lags, g1_i,
                         sample_t_ps, n_trains, n_per_train):
    """One A4-landscape page per repetition rate."""
    T_rep_ns  = 1.0 / (f_rep_GHz * 1e9) * 1e9
    n_total   = n_trains * n_per_train

    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor('white')

    # Title bar
    fig.text(0.5, 0.97,
             f'Repetition rate  f_rep = {f_rep_GHz:.0f} GHz    '
             f'(T_rep = {T_rep_ns:.2f} ns)    '
             f'N = {n_total:,} pulses    '
             f'sampled at t = {sample_t_ps:.1f} ps into each pulse',
             ha='center', va='top', fontsize=11, fontweight='bold')

    gs_grid = GridSpec(2, 3, figure=fig,
                       left=0.07, right=0.97, top=0.90, bottom=0.08,
                       hspace=0.50, wspace=0.38)

    P_f = laser.output_power(np.maximum(S_f.ravel(), 0)) * 1e3   # mW
    P_i = laser.output_power(np.maximum(S_i.ravel(), 0)) * 1e3

    # ── Column 0: Intensity distributions ────────────────────────────────────
    for row, (P, lbl, col, Sarr) in enumerate([
        (P_f, f'Free-running  (Langevin only)', COLOR_FREE, S_f),
        (P_i, f'SLD injected  (Lang-Kobayashi)', COLOR_INJ,  S_i),
    ]):
        ax = fig.add_subplot(gs_grid[row, 0])
        mu  = P.mean();  sig = P.std()
        # If relative spread is very small, plot centred deviations in nW
        # to avoid matplotlib's confusing offset notation.
        rel = sig / mu * 100 if mu > 0 else 0
        if rel < 0.01:           # spread < 0.01 % → plot in nW deviation
            Pp   = (P - mu) * 1e6   # nW deviations
            unit = 'nW'
            x_label = f'Peak power deviation from {mu:.3f} mW  (nW)'
            pdf_unit = 'nW'
        else:
            Pp       = P
            x_label  = 'Peak power (mW)'
            pdf_unit = 'mW'
        ax.hist(Pp, bins=100, density=True, color=col, alpha=0.72,
                linewidth=0.2, edgecolor='k')
        ax.axvline(Pp.mean(), color='k', ls='--', lw=0.9)
        ax.set_title(lbl, fontsize=8, fontweight='bold', pad=3)
        ax.set_xlabel(x_label, labelpad=2)
        ax.set_ylabel(f'Prob. density ({pdf_unit}⁻¹)', labelpad=2)
        ax.xaxis.set_major_formatter(
            plt.matplotlib.ticker.ScalarFormatter(useOffset=False))
        _despine(ax)
        _add_stat_box(ax,
            f'⟨P⟩ = {mu:.3f} mW\n'
            f'σ   = {sig:.3g} mW\n'
            f'σ/⟨P⟩ = {rel:.3g} %')

    # ── Column 1: Intensity autocorrelation ───────────────────────────────────
    # Merge rows with a single axis spanning both rows in column 1
    ax = fig.add_subplot(gs_grid[:, 1])
    lmax = min(len(r_f_lags), len(r_i_lags)) - 1
    ax.plot(r_f_lags[:lmax+1], r_f[:lmax+1], '-', color=COLOR_FREE,
            lw=1.2, label='Free-running')
    ax.plot(r_i_lags[:lmax+1], r_i[:lmax+1], '-', color=COLOR_INJ,
            lw=1.2, label='SLD injected')
    ax.axhline(0, color='k', ls=':', lw=0.7)
    # 2-sigma statistical noise floor
    N_eff   = n_trains * n_per_train
    sig_fl  = 2.0 / np.sqrt(N_eff)
    ax.axhspan(-sig_fl, sig_fl, color='gray', alpha=0.13, label=f'±2σ floor')
    ax.set_xlabel(f'Pulse lag  Δ  (1 unit = {T_rep_ns:.2f} ns)', labelpad=2)
    ax.set_ylabel(r'$r(\Delta) = \mathrm{Cov}(I_n,I_{n+\Delta}) / \mathrm{Var}(I)$',
                  labelpad=3)
    ax.set_title('Pulse-to-pulse intensity autocorrelation', fontsize=9,
                 fontweight='bold', pad=4)
    ax.legend(loc='upper right', framealpha=0.85)
    _despine(ax)
    _add_stat_box(ax,
        f'r(1): free = {r_f[1]:.3f}\n'
        f'r(1): inj  = {r_i[1]:.3f}', loc='lower right')

    # ── Column 2: Phase coherence |g⁽¹⁾| ─────────────────────────────────────
    ax2 = fig.add_subplot(gs_grid[:, 2])
    lmax_g = min(len(g1_f_lags), len(g1_i_lags)) - 1
    ax2.semilogy(g1_f_lags[:lmax_g+1], g1_f[:lmax_g+1] + 1e-9,
                 '-', color=COLOR_FREE, lw=1.2, label='Free-running')
    ax2.semilogy(g1_i_lags[:lmax_g+1], g1_i[:lmax_g+1] + 1e-9,
                 '-', color=COLOR_INJ,  lw=1.2, label='SLD injected')
    ax2.axhline(1.0/np.e, color='k', ls=':', lw=0.8, label='1/e level')
    ax2.set_ylim(1e-4, 2)
    ax2.set_xlabel(f'Pulse lag  Δ  (1 unit = {T_rep_ns:.2f} ns)', labelpad=2)
    ax2.set_ylabel(r'$|g^{(1)}(\Delta)|$', labelpad=3)
    ax2.set_title('Pulse-to-pulse phase coherence', fontsize=9,
                  fontweight='bold', pad=4)
    ax2.legend(loc='upper right', framealpha=0.85)
    _despine(ax2)
    lc_f = coherence_length(g1_f_lags, g1_f)
    lc_i = coherence_length(g1_i_lags, g1_i)
    _add_stat_box(ax2,
        f'|g⁽¹⁾(1)|: free = {g1_f[1]:.3f}\n'
        f'|g⁽¹⁾(1)|: inj  = {g1_i[1]:.3f}\n'
        f'L_c(1/e): free = {lc_f} pulses\n'
        f'L_c(1/e): inj  = {lc_i} pulses', loc='lower right')

    return fig


def build_summary_page(results, laser, n_trains, n_per_train):
    """Summary: key metrics vs repetition rate (1 page)."""
    freqs  = np.array([r['f_rep_GHz'] for r in results])
    mean_f = np.array([r['mean_P_f'] for r in results])
    mean_i = np.array([r['mean_P_i'] for r in results])
    sig_f  = np.array([r['sigma_frac_f'] for r in results])
    sig_i  = np.array([r['sigma_frac_i'] for r in results])
    g1_1_f = np.array([r['g1_1_f'] for r in results])
    g1_1_i = np.array([r['g1_1_i'] for r in results])
    lc_f   = np.array([r['Lc_f'] for r in results])
    lc_i   = np.array([r['Lc_i'] for r in results])
    r1_f   = np.array([r['r1_f'] for r in results])
    r1_i   = np.array([r['r1_i'] for r in results])

    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor('white')
    fig.text(0.5, 0.97,
             'Summary: Pulse-to-Pulse Statistics vs Repetition Rate  (1 – 10 GHz)',
             ha='center', va='top', fontsize=13, fontweight='bold')

    gs_grid = GridSpec(2, 3, figure=fig,
                       left=0.08, right=0.97, top=0.90, bottom=0.10,
                       hspace=0.50, wspace=0.42)

    def _ax(row, col, title, ylabel, xlabel='Repetition rate (GHz)'):
        a = fig.add_subplot(gs_grid[row, col])
        a.set_title(title, fontsize=9, fontweight='bold', pad=3)
        a.set_xlabel(xlabel, labelpad=2)
        a.set_ylabel(ylabel, labelpad=2)
        a.set_xticks(freqs)
        _despine(a)
        return a

    mk = dict(marker='o', markersize=5, lw=1.4)

    # (0,0) Mean pulse peak power
    ax = _ax(0, 0, 'Mean pulse-peak power', '⟨P⟩  (mW)')
    ax.plot(freqs, mean_f, color=COLOR_FREE, label='Free-running', **mk)
    ax.plot(freqs, mean_i, color=COLOR_INJ,  label='SLD injected', **mk)
    ax.legend()

    # (0,1) Relative intensity noise
    ax = _ax(0, 1, 'Relative intensity fluctuation', 'σ / ⟨P⟩  (%)')
    ax.semilogy(freqs, sig_f * 100, color=COLOR_FREE, label='Free-running', **mk)
    ax.semilogy(freqs, sig_i * 100, color=COLOR_INJ,  label='SLD injected', **mk)
    ax.legend()

    # (0,2) Intensity autocorr at lag=1
    ax = _ax(0, 2, 'Intensity autocorrelation at lag 1', 'r(1)')
    ax.plot(freqs, r1_f, color=COLOR_FREE, label='Free-running', **mk)
    ax.plot(freqs, r1_i, color=COLOR_INJ,  label='SLD injected', **mk)
    ax.axhline(0, color='k', ls=':', lw=0.7)
    ax.legend()

    # (1,0) Phase coherence at lag=1
    ax = _ax(1, 0, 'Phase coherence at lag 1', '|g⁽¹⁾(1)|')
    ax.plot(freqs, g1_1_f, color=COLOR_FREE, label='Free-running', **mk)
    ax.plot(freqs, g1_1_i, color=COLOR_INJ,  label='SLD injected', **mk)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0, color='k', ls=':', lw=0.7)
    ax.axhline(1, color='k', ls=':', lw=0.7)
    ax.legend()

    # (1,1) Coherence length (1/e lag)
    ax = _ax(1, 1, '1/e coherence length', 'L_c  (pulses)')
    ax.plot(freqs, lc_f, color=COLOR_FREE, label='Free-running', **mk)
    ax.plot(freqs, lc_i, color=COLOR_INJ,  label='SLD injected', **mk)
    ax.legend()

    # (1,2) Text summary table
    ax = fig.add_subplot(gs_grid[1, 2])
    ax.axis('off')
    col_labels = ['f (GHz)', '|g¹(1)| free', '|g¹(1)| inj', 'L_c free', 'L_c inj']
    cell_data  = [
        [f'{int(r["f_rep_GHz"])}',
         f'{r["g1_1_f"]:.3f}',
         f'{r["g1_1_i"]:.3f}',
         f'{r["Lc_f"]}',
         f'{r["Lc_i"]}']
        for r in results
    ]
    tbl = ax.table(cellText=cell_data, colLabels=col_labels,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.25)
    for (i, j), cell in tbl.get_celld().items():
        if i == 0:
            cell.set_facecolor('#2166AC')
            cell.set_text_props(color='white', fontweight='bold')
        elif i % 2 == 0:
            cell.set_facecolor('#e8f0f8')
    ax.set_title('Phase coherence summary', fontsize=9, fontweight='bold',
                 pad=4, y=0.97)

    return fig


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_trains',    type=int,   default=200)
    ap.add_argument('--n_per_train', type=int,   default=5000,
                    help='Pulses per train; 200 x 5000 = 1,000,000')
    ap.add_argument('--dt',          type=float, default=2e-12)
    ap.add_argument('--max_lag',     type=int,   default=100)
    ap.add_argument('--seed',        type=int,   default=42)
    ap.add_argument('--output',      type=str,   default='gs_phase_sweep_report.pdf')
    args = ap.parse_args()

    n_total = args.n_trains * args.n_per_train
    print(f'>> {n_total:,} pulses per condition  '
          f'({args.n_trains} trains × {args.n_per_train:,} pulses)')

    # ── Laser & SLD setup ──────────────────────────────────────────────────
    laser = DFBLaserParams()
    I_th  = laser.threshold_current()
    sld   = SLDParams()
    sld_res = solve_sld_steady_state(sld, 150e-3)
    inj   = InjectionParams(eta_coupling=0.10, delta_nu=0.0)
    inj.compute_derived(laser)
    S_inj, _ = sld_to_injection_field(sld, sld_res['P_out'], laser, inj)

    print(f'I_th = {I_th*1e3:.2f} mA,  tau_p = {laser.tau_p*1e12:.2f} ps')
    print(f'kappa = {inj.kappa:.3e} s^-1,  S_inj = {S_inj:.3e} m^-3')

    results = []
    t_wall = time.time()

    with PdfPages(args.output) as pdf:
        # Page 1: title
        fig_title = build_title_page(laser, sld, inj, S_inj,
                                     args.n_trains, args.n_per_train, args.dt)
        pdf.savefig(fig_title); plt.close(fig_title)

        for f_GHz in range(1, 11):
            f_rep = f_GHz * 1e9
            T_rep = 1.0 / f_rep
            t_rise = min(50e-12, 0.05 * T_rep)     # 5% of period, cap 50 ps

            gs = GainSwitchParams(
                f_rep=f_rep, duty=0.30,
                I_bias_factor=0.9, I_peak_factor=5.0,
                t_rise=t_rise,
            )

            # Find pulse-peak offset (fraction of period)
            print(f'\n── {f_GHz} GHz: finding peak offset...')
            peak_pts = find_peak_offset(laser, gs)
            frac = peak_pts / 2000          # 2000 is default pts_per_period in find_peak_offset

            # Free-running
            print(f'── {f_GHz} GHz: simulating free-running...')
            S_f, phi_f, spt = run_simulation(
                laser, gs, 0.0, 0.0, 0.0,
                args.n_trains, args.n_per_train, args.dt, frac,
                seed=args.seed,
                verbose_label=f'{f_GHz}GHz-free',
            )

            # SLD injected
            print(f'── {f_GHz} GHz: simulating injected...')
            S_i, phi_i, _ = run_simulation(
                laser, gs, inj.kappa, S_inj, inj.delta_omega,
                args.n_trains, args.n_per_train, args.dt, frac,
                seed=args.seed + 1,
                verbose_label=f'{f_GHz}GHz-inj',
            )

            # Statistics
            ml = args.max_lag
            r_f_lags, r_f  = intensity_autocorr(S_f,  max_lag=ml)
            r_i_lags, r_i  = intensity_autocorr(S_i,  max_lag=ml)
            g_f_lags, g1_f = phase_coherence(phi_f,   max_lag=ml)
            g_i_lags, g1_i = phase_coherence(phi_i,   max_lag=ml)

            P_f = laser.output_power(np.maximum(S_f.ravel(), 0)) * 1e3
            P_i = laser.output_power(np.maximum(S_i.ravel(), 0)) * 1e3

            rec = {
                'f_rep_GHz'   : f_GHz,
                'mean_P_f'    : float(P_f.mean()),
                'mean_P_i'    : float(P_i.mean()),
                'sigma_frac_f': float(P_f.std() / P_f.mean()) if P_f.mean() > 0 else 0,
                'sigma_frac_i': float(P_i.std() / P_i.mean()) if P_i.mean() > 0 else 0,
                'g1_1_f'      : float(g1_f[1]) if len(g1_f) > 1 else 0,
                'g1_1_i'      : float(g1_i[1]) if len(g1_i) > 1 else 0,
                'Lc_f'        : coherence_length(g_f_lags, g1_f),
                'Lc_i'        : coherence_length(g_i_lags, g1_i),
                'r1_f'        : float(r_f[1]) if len(r_f) > 1 else 0,
                'r1_i'        : float(r_i[1]) if len(r_i) > 1 else 0,
            }
            results.append(rec)
            elapsed = time.time() - t_wall
            print(f'   {f_GHz} GHz done — '
                  f'free σ/P={rec["sigma_frac_f"]*100:.3f}%  '
                  f'|g1(1)|_free={rec["g1_1_f"]:.3f}  '
                  f'|g1(1)|_inj={rec["g1_1_i"]:.3f}  '
                  f'[{elapsed:.0f}s total]')

            # Build page
            fig_page = build_frequency_page(
                f_GHz, laser,
                S_f, phi_f, r_f_lags, r_f, g_f_lags, g1_f,
                S_i, phi_i, r_i_lags, r_i, g_i_lags, g1_i,
                spt, args.n_trains, args.n_per_train,
            )
            pdf.savefig(fig_page); plt.close(fig_page)

            # Free memory — we only need summary stats for the final page
            del S_f, phi_f, S_i, phi_i

        # Final summary page
        fig_sum = build_summary_page(results, laser, args.n_trains, args.n_per_train)
        pdf.savefig(fig_sum); plt.close(fig_sum)

        # PDF metadata
        d = pdf.infodict()
        d['Title']   = 'Phase Correlation Analysis of Gain-Switched DFB Laser (1-10 GHz)'
        d['Subject'] = '1 GHz – 10 GHz gain-switched pulse statistics with SLD injection'
        d['Keywords']= 'DFB laser, gain switching, phase coherence, Lang-Kobayashi, SLD injection'

    total = time.time() - t_wall
    print(f'\nDone in {total:.1f} s  →  {args.output}')


if __name__ == '__main__':
    main()
