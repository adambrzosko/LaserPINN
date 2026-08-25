"""
Numerical simulation of 10 GHz phase-randomised gain-switched DFB + SLD.

Reproduces the experiment in:
  Y. S. Lo et al., "Phase-Randomized Laser Pulse Generation at 10 GHz
  for Quantum Photonic Applications," arXiv:2601.04031 (2025).

Addresses reviewer concerns:
  #2  Physical model — stochastic rate equations with Langevin + ASE noise
  #3  Quantitative phase randomisation — r1, KL, KS, arcsine fits, CI
  #5  QKD trade-offs — timing jitter vs randomisation quality
  #6  Optical spectra — filtered & unfiltered comb structure
  #7  Injection-power dependence — systematic sweep of all metrics

Experimental parameters:
  DC bias:  45.5 mA (laser), 550 mA (SLD)
  RF:       3.7 V amplitude sinusoidal into 50 Ω
  Filter:   0.3 nm bandpass on output (detection) path

Outputs:  images/paper_10ghz/
"""
import numpy as np
import time as _time
from scipy.stats import kstest

from core.dfb_laser import make_laser, q, h, c
from core.million_pulse_comparison import (
    simulate_pulses, simulate_pulses_waveform, build_raised_cosine,
)
from studies.multimode_analysis import simulate_singlemode, mode_setup
from gsdfb import compute_r1, setup_plotting, save_fig
from gsdfb.analysis import (
    phase_randomisation_quality, absolute_jitter, amzi_outputs,
)
from gsdfb.plotting import img_dir


# ── Device parameters matching the paper ────────────────────────────────

def make_paper_laser():
    """1547 nm DFB matching arXiv:2601.04031.

    Paper reports:
      - Centre wavelength: 1547 nm
      - Modulation bandwidth: 18 GHz
      - Cavity linewidth: ~159 GHz → tau_p ~ 1 ps
    We use a 150 um cavity (shorter than the default 300 um) to achieve
    a photon lifetime closer to the paper's value.
    """
    return make_laser('dfb', lambda0=1547e-9, L=150e-6)


# ── SLD power to S_inj conversion ──────────────────────────────────────

def sld_power_to_sinj(P_sld_mW, laser, sld_bw_nm=33.0, acceptance_bw_nm=8.0):
    """Convert total SLD output power (mW) to internal photon density S_inj.

    The SLD (centred 1550 nm, 33 nm 3-dB bandwidth) is broadband.
    Only the fraction within the DFB acceptance bandwidth couples to
    the lasing mode.

    Parameters
    ----------
    P_sld_mW : float
        Total SLD output power in milliwatts.
    laser : DFBLaserParams
    sld_bw_nm : float
        SLD 3-dB bandwidth in nm (default 33.0 nm, matching arXiv:2601.04031).
    acceptance_bw_nm : float
        Effective DFB acceptance bandwidth in nm (default 8.0 nm).
        This is wider than the passive grating stopband (~1.5 nm) because
        the SLD also injects noise through broadband carrier depletion:
        SLD light outside the mode bandwidth is absorbed by the gain medium,
        creating carrier density fluctuations that couple into the phase
        via the linewidth enhancement factor alpha_H. The effective
        acceptance bandwidth accounts for both direct mode coupling
        (~1.5 nm, Bragg stopband) and the broader carrier-noise channel.
    """
    # Spectral coupling: fraction of SLD within acceptance bandwidth
    spectral_frac = acceptance_bw_nm / sld_bw_nm
    # Additional coupling loss (spatial mode mismatch, polarisation, circulator)
    coupling_loss = 0.5   # ~3 dB
    total_coupling = spectral_frac * coupling_loss

    P_coupled_W = total_coupling * P_sld_mW * 1e-3
    nu0 = c / laser.lambda0
    S_inj = P_coupled_W * laser.tau_p / (h * nu0 * laser.V)
    return S_inj


# ── Drive parameters from the paper ───────────────────────────────────
#   Sinusoidal modulation: I(t) = I_DC + I_RF·sin(2π f_rep t)

DT_TARGET = 0.5e-12   # target time step (ps)
I_DC = 45.5e-3         # laser DC bias (A)
V_RF_AMP = 3.7         # RF modulation voltage amplitude (V)
Z_MATCH = 50.0         # matched impedance (Ω)
I_RF = V_RF_AMP / Z_MATCH   # peak RF current amplitude (A)

# Output bandpass filter (on detection path, removes broadband SLD leakage)
FILTER_BW_NM = 0.3     # filter 3-dB bandwidth in nm


def build_sine_waveform(pts_period, dt, f_rep, I_dc, I_rf):
    """Sinusoidal gain-switching waveform: I(t) = I_dc + I_rf·sin(2πft)."""
    t = np.arange(pts_period) * dt
    waveform = I_dc + I_rf * np.sin(2.0 * np.pi * f_rep * t)
    return np.maximum(waveform, 0.0).astype(np.float64)


def run_config(laser, f_rep, S_inj, n_pulses, n_discard=200, seed=42):
    """Run gain-switched simulation with sinusoidal modulation."""
    T_rep = 1.0 / f_rep
    pts = max(int(round(T_rep / DT_TARGET)), 100)
    dt = T_rep / pts

    waveform = build_sine_waveform(pts, dt, f_rep, I_DC, I_RF)

    pk_phi, pk_S, samp_S, pk_k = simulate_pulses_waveform(
        n_pulses + n_discard, n_discard, pts, dt,
        waveform,
        laser.V, laser.Gamma, laser.v_g, laser.a, laser.N_tr, laser.epsilon,
        laser.A, laser.B, laser.C, laser.tau_p, laser.beta_sp, laser.alpha_H, q,
        S_inj, seed)

    # Phase metrics
    r1, dphi = compute_r1(pk_phi)
    pq = phase_randomisation_quality(pk_phi)

    # Timing jitter
    sigma_t, t_peak = absolute_jitter(pk_k, dt)

    # Power stats
    pk_P = laser.output_power(np.maximum(pk_S, 0))
    mean_P = float(np.mean(pk_P))
    cv_P = float(np.std(pk_P) / mean_P) if mean_P > 0 else 0.0

    # AMZI
    I_A, I_B, eta = amzi_outputs(pk_P, pk_phi)

    # Multi-lag autocorrelation
    max_lag = min(50, n_pulses // 10)
    autocorr = np.zeros(max_lag)
    for k in range(max_lag):
        if k == 0:
            autocorr[k] = 1.0
        else:
            autocorr[k] = float(np.abs(
                np.mean(np.exp(1j * (pk_phi[k:] - pk_phi[:-k])))))

    return dict(
        r1=r1, dphi=dphi, pq=pq,
        sigma_t=sigma_t, t_peak=t_peak,
        mean_P=mean_P, cv_P=cv_P,
        pk_phi=pk_phi, pk_S=pk_S, pk_P=pk_P, pk_k=pk_k,
        eta=eta, I_A=I_A, I_B=I_B,
        autocorr=autocorr, dt=dt, pts=pts,
        kl=pq['kl'], ks_stat=pq['ks_stat'],
    )


# ── Helper: confidence interval for autocorrelation under H0 ──────────

def autocorr_ci(n_pulses, confidence=0.99):
    """99% CI for |r_k| under H0: uniform iid phases.

    Under H0, Re and Im of mean(exp(i*dphi)) are each ~ N(0, 1/(2N)).
    |r_k|^2 ~ Exp(1/N) → |r_k| ~ Rayleigh(1/sqrt(2N)).
    The p-quantile of |r_k| is sqrt(-ln(1-p)/N).
    """
    return np.sqrt(-np.log(1.0 - confidence) / n_pulses)


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    setup_plotting()
    out = img_dir('paper_10ghz')

    laser = make_paper_laser()
    I_th = laser.threshold_current()

    print("=" * 70)
    print("  10 GHz Phase-Randomised GS-DFB + SLD Injection")
    print("  Numerical simulation matching arXiv:2601.04031")
    print("=" * 70)
    print(f"\n  Device: {laser.lambda0*1e9:.0f} nm DFB, "
          f"L = {laser.L*1e6:.0f} um, tau_p = {laser.tau_p*1e12:.2f} ps")

    I_min = max(I_DC - I_RF, 0)
    I_max = I_DC + I_RF
    # Below-threshold fraction of sinusoidal cycle
    sin_thr = (I_th - I_DC) / I_RF
    if -1 < sin_thr < 1:
        alpha_thr = np.arcsin(abs(sin_thr))
        frac_below = (np.pi - 2 * alpha_thr) / (2 * np.pi)
    elif sin_thr <= -1:
        frac_below = 0.0
    else:
        frac_below = 1.0

    print(f"  I_th = {I_th*1e3:.2f} mA")
    print(f"  Drive: I_DC = {I_DC*1e3:.1f} mA ({I_DC/I_th:.1f}x I_th), "
          f"V_RF = {V_RF_AMP:.1f} V amp into {Z_MATCH:.0f} Ω")
    print(f"         I(t) = {I_DC*1e3:.1f} ± {I_RF*1e3:.1f} mA sinusoidal")
    print(f"         I_min = {I_min*1e3:.1f} mA ({I_min/I_th:.2f}x I_th), "
          f"I_max = {I_max*1e3:.1f} mA ({I_max/I_th:.1f}x I_th)")
    print(f"         Below threshold for {frac_below*100:.1f}% of cycle")

    # SLD calibration
    P_sld_main = 19.0   # mW — paper's operating point (SLD at 550 mA)
    S_inj_main = sld_power_to_sinj(P_sld_main, laser)
    filter_bw_hz_info = c * FILTER_BW_NM * 1e-9 / (laser.lambda0**2)
    print(f"\n  SLD: {P_sld_main:.0f} mW total -> S_inj = {S_inj_main:.2e} m^-3")
    print(f"  Output filter: {FILTER_BW_NM} nm "
          f"({filter_bw_hz_info*1e-9:.1f} GHz) on detection path")

    # JIT warmup — compile both Numba solvers
    print("\n  Compiling JIT...", end="", flush=True)
    t0 = _time.time()
    _wf = build_sine_waveform(50, 1e-12, 10e9, I_DC, I_RF)
    simulate_pulses_waveform(
        100, 10, 50, 1e-12,
        _wf,
        laser.V, laser.Gamma, laser.v_g, laser.a, laser.N_tr, laser.epsilon,
        laser.A, laser.B, laser.C, laser.tau_p, laser.beta_sp, laser.alpha_H, q,
        0.0, 0)
    del _wf
    print(f" {_time.time()-t0:.1f}s")

    # ── Part 1: Frequency sweep — free-running vs SLD-injected ───────

    N_PULSES = 500_000
    freqs = [1e9, 5e9, 8e9, 10e9]
    freq_labels = ['1 GHz', '5 GHz', '8 GHz', '10 GHz']

    print(f"\n  Part 1: Frequency sweep ({N_PULSES//1000}k pulses)")

    results = {}
    for f_rep, flabel in zip(freqs, freq_labels):
        for inj_label, P_sld in [('free', 0.0), ('sld', P_sld_main)]:
            S_inj = sld_power_to_sinj(P_sld, laser)
            key = f'{flabel}_{inj_label}'
            seed = hash(key) % 100000

            print(f"    {key:20s} ...", end="", flush=True)
            t0 = _time.time()
            results[key] = run_config(laser, f_rep, S_inj, N_PULSES, seed=seed)
            elapsed = _time.time() - t0
            r = results[key]
            print(f" {elapsed:.1f}s  r1={r['r1']:.4f}  "
                  f"KL={r['kl']:.4f}  jitter={r['sigma_t']*1e12:.1f}ps")

    # ── Figure 1: AMZI histograms ────────────────────────────────────

    fig1, axes1 = plt.subplots(2, 4, figsize=(20, 8))
    fig1.suptitle(
        'AMZI Splitting-Ratio Histograms\n'
        'Top: free-running | Bottom: 19 mW SLD injection',
        fontsize=13)

    # Theoretical arcsine PDF for reference
    eta_th = np.linspace(0.001, 0.999, 500)
    arcsine_pdf = 1.0 / (np.pi * np.sqrt(eta_th * (1.0 - eta_th)))

    for col, (f_rep, flabel) in enumerate(zip(freqs, freq_labels)):
        for row, inj_label in enumerate(['free', 'sld']):
            key = f'{flabel}_{inj_label}'
            r = results[key]
            ax = axes1[row, col]

            ax.hist(r['eta'], bins=200, density=True, alpha=0.7,
                    color='C0' if row == 0 else 'C1',
                    label=f'Simulation (r$_1$={r["r1"]:.3f})')
            ax.plot(eta_th, arcsine_pdf, 'k--', lw=1.2, alpha=0.6,
                    label='Arcsine (ideal)')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 12)
            ax.set_xlabel('Splitting ratio $\\eta$')
            if col == 0:
                ax.set_ylabel('Probability density')
            ax.set_title(f'{flabel}' + (' — Free' if row == 0 else ' — SLD'))
            ax.legend(fontsize=7, loc='upper center')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig1, f'{out}/amzi_histograms.png')

    # ── Figure 2: Autocorrelation at 10 GHz ──────────────────────────

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    fig2.suptitle(
        'Phase Autocorrelation $|\\langle e^{i(\\phi_n - \\phi_{n-k})} \\rangle|$ '
        'vs Pulse Lag',
        fontsize=13)

    ci99 = autocorr_ci(N_PULSES, 0.99)
    ci999 = autocorr_ci(N_PULSES, 0.999)

    for ax, flabel, title in [
        (axes2[0], '1 GHz', '1 GHz — Free (reference)'),
        (axes2[1], '10 GHz', '10 GHz — Free'),
        (axes2[2], '10 GHz', '10 GHz — SLD injected'),
    ]:
        inj = 'free' if 'Free' in title or 'reference' in title else 'sld'
        key = f'{flabel}_{inj}'
        r = results[key]
        lags = np.arange(len(r['autocorr']))

        ax.bar(lags[1:], r['autocorr'][1:], width=0.8,
               color='C0' if inj == 'free' else 'C1', alpha=0.7)
        ax.axhline(ci99, color='r', ls='--', lw=1,
                    label=f'99% CI = {ci99:.4f}')
        ax.axhline(ci999, color='r', ls=':', lw=1,
                    label=f'99.9% CI = {ci999:.4f}')
        ax.set_xlabel('Lag $k$')
        ax.set_ylabel('$|r_k|$')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 50)

        # Auto-scale y: use 1.5× the max for free, or fixed for correlated
        ymax = max(0.01, 1.5 * np.max(r['autocorr'][1:]))
        ax.set_ylim(0, min(ymax, 1.0))

    plt.tight_layout()
    save_fig(fig2, f'{out}/autocorrelation.png')

    # ── Part 2: Injection-power sweep at 10 GHz ─────────────────────

    N_PULSES_SWEEP = 200_000
    P_sld_sweep = np.array([0, 1, 2, 3, 5, 8, 10, 13, 16, 19, 22, 25, 30])

    print(f"\n  Part 2: Injection-power sweep at 10 GHz "
          f"({N_PULSES_SWEEP//1000}k pulses × {len(P_sld_sweep)} powers)")

    sweep = {k: [] for k in ['P_mW', 'r1', 'kl', 'ks_stat',
                              'sigma_t_ps', 'cv_P', 'mean_P_mW']}

    for P_mW in P_sld_sweep:
        S_inj = sld_power_to_sinj(P_mW, laser)
        seed = 10000 + int(P_mW * 100)

        print(f"    P_SLD = {P_mW:5.1f} mW ...", end="", flush=True)
        t0 = _time.time()
        r = run_config(laser, 10e9, S_inj, N_PULSES_SWEEP, seed=seed)
        elapsed = _time.time() - t0
        print(f" {elapsed:.1f}s  r1={r['r1']:.4f}  "
              f"jitter={r['sigma_t']*1e12:.1f}ps")

        sweep['P_mW'].append(P_mW)
        sweep['r1'].append(r['r1'])
        sweep['kl'].append(r['kl'])
        sweep['ks_stat'].append(r['ks_stat'])
        sweep['sigma_t_ps'].append(r['sigma_t'] * 1e12)
        sweep['cv_P'].append(r['cv_P'])
        sweep['mean_P_mW'].append(r['mean_P'] * 1e3)

    for k in sweep:
        sweep[k] = np.array(sweep[k])

    # ── Figure 3: Injection-power dependence ─────────────────────────

    fig3, axes3 = plt.subplots(2, 3, figsize=(18, 10))
    fig3.suptitle(
        'Phase Randomisation vs SLD Injection Power — 10 GHz\n'
        'Addressing Reviewer Concern #7: systematic power study',
        fontsize=13)

    # r1 vs power
    ax = axes3[0, 0]
    ax.plot(sweep['P_mW'], sweep['r1'], 'o-', color='C0', lw=2, ms=6)
    ax.axhline(ci99, color='r', ls='--', lw=1, alpha=0.7,
               label=f'99% CI under H$_0$ = {ci99:.4f}')
    ax.set_ylabel('Phase correlation $r_1$')
    ax.set_title('(a) Inter-pulse phase correlation')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # KL divergence vs power
    ax = axes3[0, 1]
    ax.plot(sweep['P_mW'], sweep['kl'], 's-', color='C2', lw=2, ms=6)
    ax.set_ylabel('KL divergence from uniform (bits)')
    ax.set_title('(b) Phase distribution non-uniformity')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # KS statistic vs power
    ax = axes3[0, 2]
    ax.plot(sweep['P_mW'], sweep['ks_stat'], 'd-', color='C4', lw=2, ms=6)
    ax.axhline(1.63 / np.sqrt(min(N_PULSES_SWEEP, 10000)),
               color='r', ls='--', lw=1, alpha=0.7, label='KS 1% threshold')
    ax.set_ylabel('KS statistic vs uniform')
    ax.set_title('(c) Kolmogorov-Smirnov test')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Timing jitter vs power
    ax = axes3[1, 0]
    ax.plot(sweep['P_mW'], sweep['sigma_t_ps'], '^-', color='C1', lw=2, ms=6)
    ax.set_ylabel('RMS timing jitter (ps)')
    ax.set_title('(d) Timing jitter — QKD penalty')
    ax.grid(True, alpha=0.3)

    # Intensity CV vs power
    ax = axes3[1, 1]
    ax.plot(sweep['P_mW'], sweep['cv_P'], 'v-', color='C3', lw=2, ms=6)
    ax.set_ylabel('Intensity CV')
    ax.set_title('(e) Pulse intensity noise')
    ax.grid(True, alpha=0.3)

    # Mean power vs injection
    ax = axes3[1, 2]
    ax.plot(sweep['P_mW'], sweep['mean_P_mW'], 'p-', color='C5', lw=2, ms=6)
    ax.set_ylabel('Mean peak power (mW)')
    ax.set_title('(f) Output power')
    ax.grid(True, alpha=0.3)

    for ax in axes3.flat:
        ax.set_xlabel('SLD injection power (mW)')

    plt.tight_layout()
    save_fig(fig3, f'{out}/injection_power_sweep.png')

    # ── Part 3: Optical spectrum (comb lines) ────────────────────────

    print(f"\n  Part 3: Optical spectrum — comb line analysis")

    N_SPEC = 2000      # pulses for spectral analysis
    N_MODES_SPEC = 1   # single mode for spectrum

    # Use the single-mode solver which stores full E(t).
    # simulate_singlemode uses a pulsed waveform, so we map the sinusoidal
    # drive to approximate equivalents: I_off ≈ I_min, I_on ≈ I_max,
    # t_on ≈ T_rep/2, t_rise from sinusoidal quarter-period.
    from studies.multimode_analysis import simulate_singlemode

    # JIT warmup for singlemode
    simulate_singlemode(
        4, 50, 1e-12, 0.01, 0.08, 15e-12, 5e-12,
        laser.V, laser.Gamma, laser.v_g, laser.a, laser.N_tr, laser.epsilon,
        laser.A, laser.B, laser.C, laser.tau_p, laser.beta_sp, laser.alpha_H, q,
        0.0, 0)

    spec_data = {}
    for flabel, f_rep in [('1 GHz', 1e9), ('10 GHz', 10e9)]:
        for inj_label, P_sld in [('free', 0.0), ('sld', P_sld_main)]:
            S_inj = sld_power_to_sinj(P_sld, laser)
            key = f'{flabel}_{inj_label}'
            seed = hash(key + '_spec') % 100000

            T_rep = 1.0 / f_rep
            # Approximate sinusoidal as pulsed for spectrum solver
            t_on_approx = 0.5 * T_rep
            pts = max(int(round(T_rep / DT_TARGET)), 100)
            dt = T_rep / pts
            t_rise_approx = min(T_rep / 8.0, 20e-12)

            print(f"    Spectrum {key:20s} ...", end="", flush=True)
            t0 = _time.time()
            Er, Ei, _ = simulate_singlemode(
                N_SPEC, pts, dt,
                I_min, I_max, t_on_approx, t_rise_approx,
                laser.V, laser.Gamma, laser.v_g, laser.a,
                laser.N_tr, laser.epsilon,
                laser.A, laser.B, laser.C,
                laser.tau_p, laser.beta_sp, laser.alpha_H, q,
                S_inj, seed)
            elapsed = _time.time() - t0
            print(f" {elapsed:.1f}s")

            # Compute optical spectrum via FFT of complex field
            E_complex = Er + 1j * Ei
            total_pts = len(E_complex)

            # Windowed FFT
            window = np.hanning(total_pts)
            E_windowed = E_complex * window
            spectrum = np.abs(np.fft.fftshift(np.fft.fft(E_windowed)))**2
            freq_axis = np.fft.fftshift(np.fft.fftfreq(total_pts, dt))

            # Normalise to peak (unfiltered)
            spectrum /= np.max(spectrum)

            # Apply output bandpass filter (0.3 nm, detection path)
            # Convert filter bandwidth to frequency:
            #   delta_nu = c * delta_lambda / lambda^2
            filter_bw_hz = c * FILTER_BW_NM * 1e-9 / (laser.lambda0**2)
            # Super-Gaussian (order 2) for realistic flat-top filter shape
            spectrum_filtered = spectrum * np.exp(
                -np.log(2) * (2 * freq_axis / filter_bw_hz)**4)
            filt_max = np.max(spectrum_filtered)
            if filt_max > 0:
                spectrum_filtered /= filt_max

            spec_data[key] = dict(
                freq=freq_axis, spectrum=spectrum,
                spectrum_filtered=spectrum_filtered,
                filter_bw_hz=filter_bw_hz,
                dt=dt, total_pts=total_pts, f_rep=f_rep)

    # ── Figure 4: Optical spectra ────────────────────────────────────

    filter_bw_GHz = c * FILTER_BW_NM * 1e-9 / (laser.lambda0**2) * 1e-9
    print(f"    Output filter: {FILTER_BW_NM} nm = {filter_bw_GHz:.1f} GHz FWHM")

    fig4, axes4 = plt.subplots(1, 3, figsize=(18, 5))
    fig4.suptitle(
        f'Optical Spectrum — {FILTER_BW_NM} nm Output Filter '
        f'(Δν = {filter_bw_GHz:.0f} GHz)\n'
        'Addressing Reviewer Concern #6: comb line suppression',
        fontsize=13)

    spec_configs = [
        ('1 GHz_free', '1 GHz — Free (reference)', 'C0'),
        ('10 GHz_free', '10 GHz — Free (comb lines)', 'C3'),
        ('10 GHz_sld', '10 GHz — SLD injected', 'C1'),
    ]

    for ax, (key, title, color) in zip(axes4, spec_configs):
        sd = spec_data[key]
        f_GHz = sd['freq'] * 1e-9

        # Show ±60 GHz around DC (optical carrier)
        mask = np.abs(f_GHz) < 60

        # Unfiltered (faint)
        spec_dB_raw = 10 * np.log10(np.maximum(sd['spectrum'][mask], 1e-10))
        ax.plot(f_GHz[mask], spec_dB_raw, color=color, lw=0.3, alpha=0.25,
                label='Unfiltered')

        # Filtered with 0.3 nm bandpass (bold)
        spec_dB_filt = 10 * np.log10(
            np.maximum(sd['spectrum_filtered'][mask], 1e-10))
        ax.plot(f_GHz[mask], spec_dB_filt, color=color, lw=0.8, alpha=0.9,
                label=f'{FILTER_BW_NM} nm filtered')

        # Mark expected comb line positions
        f_rep_GHz = sd['f_rep'] * 1e-9
        for n in range(-6, 7):
            if n != 0:
                ax.axvline(n * f_rep_GHz, color='gray', ls=':',
                           lw=0.5, alpha=0.3)

        # Show filter passband edges
        ax.axvline(-filter_bw_GHz / 2, color='k', ls='--', lw=0.7,
                    alpha=0.4)
        ax.axvline(filter_bw_GHz / 2, color='k', ls='--', lw=0.7,
                    alpha=0.4)

        ax.set_xlabel('Frequency offset (GHz)')
        ax.set_ylabel('Power spectral density (dB)')
        ax.set_title(title)
        ax.set_ylim(-60, 5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='upper right')

        # Annotate comb spacing
        if 'free' in key and '10' in key:
            ax.annotate(f'{f_rep_GHz:.0f} GHz spacing',
                        xy=(f_rep_GHz, -10), fontsize=8, color='gray')

    plt.tight_layout()
    save_fig(fig4, f'{out}/optical_spectra.png')

    # ── Part 4: Mechanism decomposition ──────────────────────────────
    # Separate effect of:
    #   (a) Increased spontaneous emission noise (R_sp,eff = R_sp + R_ASE)
    #   (b) Random seeding of each pulse from ASE field
    #   (c) Timing jitter washing out residual correlation
    # In our model, (a) and (b) are both captured by the SLD noise term.
    # (c) follows naturally from the perturbed carrier dynamics.

    print(f"\n  Part 4: Mechanism decomposition at 10 GHz")

    # Run a fine sweep of SLD power to identify the threshold
    P_fine = np.linspace(0, 30, 30)
    r1_fine = []
    jitter_fine = []

    for P_mW in P_fine:
        S_inj = sld_power_to_sinj(P_mW, laser)
        seed = 50000 + int(P_mW * 100)
        r = run_config(laser, 10e9, S_inj, 100_000, seed=seed)
        r1_fine.append(r['r1'])
        jitter_fine.append(r['sigma_t'] * 1e12)

    r1_fine = np.array(r1_fine)
    jitter_fine = np.array(jitter_fine)

    # Find threshold: where r1 drops below 99% CI
    ci99_100k = autocorr_ci(100_000, 0.99)
    threshold_idx = np.where(r1_fine < ci99_100k)[0]
    P_threshold = P_fine[threshold_idx[0]] if len(threshold_idx) > 0 else P_fine[-1]

    print(f"    Phase randomisation threshold: P_SLD ~ {P_threshold:.1f} mW "
          f"(r1 < {ci99_100k:.4f})")

    # ── Figure 5: Mechanism decomposition ────────────────────────────

    fig5, axes5 = plt.subplots(1, 3, figsize=(18, 5))
    fig5.suptitle(
        'Physical Mechanism: ASE Injection → Phase Randomisation\n'
        'Addressing Reviewer Concern #2',
        fontsize=13)

    # Panel (a): r1 vs SLD power — threshold identification
    ax = axes5[0]
    ax.plot(P_fine, r1_fine, 'o-', color='C0', ms=4, lw=1.5)
    ax.axhline(ci99_100k, color='r', ls='--', lw=1,
               label=f'99% CI = {ci99_100k:.4f}')
    ax.axvspan(P_threshold, P_fine[-1], alpha=0.1, color='green',
               label='Phase-randomised regime')
    ax.axvline(P_threshold, color='green', ls=':', lw=1)
    ax.set_xlabel('SLD injection power (mW)')
    ax.set_ylabel('$r_1$')
    ax.set_title('(a) Coherence destruction threshold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Panel (b): jitter vs SLD power
    ax = axes5[1]
    ax.plot(P_fine, jitter_fine, 's-', color='C1', ms=4, lw=1.5)
    ax.axvline(P_threshold, color='green', ls=':', lw=1,
               label=f'Threshold = {P_threshold:.0f} mW')

    # Annotate paper's experimental values
    paper_P = [0, 5, 19, 24]
    paper_jitter = [4.4, 15.1, 21.9, 28.3]
    ax.scatter(paper_P, paper_jitter, marker='*', s=150, c='red',
               zorder=5, label='Experiment (paper)')

    ax.set_xlabel('SLD injection power (mW)')
    ax.set_ylabel('RMS timing jitter (ps)')
    ax.set_title('(b) Jitter trade-off')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel (c): r1 vs jitter — parametric plot
    ax = axes5[2]
    sc = ax.scatter(jitter_fine, r1_fine, c=P_fine, cmap='viridis',
                    s=40, edgecolors='k', lw=0.5, zorder=3)
    ax.axhline(ci99_100k, color='r', ls='--', lw=1)
    ax.set_xlabel('RMS timing jitter (ps)')
    ax.set_ylabel('$r_1$')
    ax.set_title('(c) Randomisation vs jitter trade-off')
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('SLD power (mW)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    save_fig(fig5, f'{out}/mechanism_decomposition.png')

    # ── Summary table ────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("  Summary — Simulation Results")
    print("=" * 70)

    print(f"\n  {'Config':>22s}  {'r1':>7s}  {'KL':>7s}  {'KS':>7s}  "
          f"{'jitter':>8s}  {'CV':>6s}  {'P_mean':>8s}")
    print("  " + "-" * 72)

    for f_rep, flabel in zip(freqs, freq_labels):
        for inj_label in ['free', 'sld']:
            key = f'{flabel}_{inj_label}'
            r = results[key]
            tag = f'{flabel} {"Free":>5s}' if inj_label == 'free' \
                else f'{flabel} {f"SLD({P_sld_main:.0f}mW)":>11s}'
            print(f"  {tag:>22s}  {r['r1']:7.4f}  {r['kl']:7.4f}  "
                  f"{r['ks_stat']:7.4f}  {r['sigma_t']*1e12:7.1f}ps  "
                  f"{r['cv_P']:6.3f}  {r['mean_P']*1e3:7.3f}mW")

    print(f"\n  Phase randomisation threshold at 10 GHz: "
          f"P_SLD > {P_threshold:.0f} mW")
    print(f"  99% CI for r1 under H0 (uniform): {ci99:.5f}")

    # Min-entropy estimate
    r_10g_sld = results['10 GHz_sld']
    # H_min from AMZI: for near-uniform eta, H_min ~ log2(n_bins) - log2(max_count/n)
    eta_vals = r_10g_sld['eta']
    hist_counts, _ = np.histogram(eta_vals, bins=256, range=(0, 1))
    p_max = np.max(hist_counts) / len(eta_vals)
    H_min = -np.log2(p_max) if p_max > 0 else 8.0
    print(f"\n  Min-entropy estimate (10 GHz + SLD, 8-bit): "
          f"H_min = {H_min:.3f} bit/sample")
    print(f"  → Indicative QRNG rate: {H_min * 10:.1f} Gbit/s "
          f"(upper bound, no extraction losses)")

    # Comparison with paper's experimental jitter values
    # Add electronic jitter floor in quadrature (RF source + detection)
    sigma_elec = 4.0  # ps — typical for 10 GHz RF drive chain
    print(f"\n  Timing jitter comparison (10 GHz):")
    print(f"    σ_electronic = {sigma_elec:.1f} ps "
          f"(added in quadrature to simulation)")
    print(f"    {'P_SLD (mW)':>12s}  {'Experiment':>12s}  "
          f"{'Sim (optical)':>14s}  {'Sim (total)':>12s}")
    print("    " + "-" * 56)
    for P_mW, exp_jitter in zip(paper_P, paper_jitter):
        # Find closest simulation point
        idx = np.argmin(np.abs(sweep['P_mW'] - P_mW))
        sim_opt = sweep['sigma_t_ps'][idx]
        sim_total = np.sqrt(sim_opt**2 + sigma_elec**2)
        print(f"    {P_mW:12.0f}  {exp_jitter:11.1f}ps  "
              f"{sim_opt:13.1f}ps  {sim_total:11.1f}ps")

    print("\n" + "=" * 70)
    print("  Done. Figures saved to images/paper_10ghz/")
    print("=" * 70)
