"""
Multi-mode competition under gain switching.

A DFB laser is nominally single-mode, but under strong gain switching
the side modes can lase transiently during the turn-on transient before
the main mode saturates the gain.  This produces mode partition noise
(MPN): anti-correlated intensity fluctuations between modes.

Model: N_modes longitudinal modes sharing a single carrier reservoir.
Each mode m has its own complex field E_m = Er_m + j*Ei_m, with:

  - A frequency detuning delta_nu_m from the gain peak
  - A gain offset: g_m(N) = a*(N - N_tr)/(1+eps*S_total) * G_shape(delta_nu_m)
  - The same carrier reservoir: all modes deplete N together
  - Independent spontaneous emission noise (uncorrelated between modes)
  - Optional SLD injection (broadband, enters all modes equally)

The gain spectrum is parabolic:
  G_shape(delta_nu) = 1 - (delta_nu / delta_nu_gain)^2

where delta_nu_gain ~ 3-5 THz is the gain bandwidth (half-width).

Key observables:
  - Side-mode suppression ratio (SMSR) vs time within each pulse
  - Mode partition noise: correlation between main and side mode power
  - Effect on total intensity stability and phase coherence
  - How SLD injection changes the mode competition dynamics
"""
import numpy as np
import time as _time
from numba import njit

from core.dfb_laser import DFBLaserParams, make_laser, q, h, c
from gsdfb import compute_r1, setup_plotting, save_fig


# ── Mode parameters ───────────────────────────────────────────────────

# DFB longitudinal mode spacing: delta_lambda = lambda^2 / (2*n_g*L)
# At 1550 nm, n_g=3.7, L=300 um: delta_lambda = 1.08 nm → delta_nu = 135 GHz
# DFB grating selects main mode; nearest side modes are at +/- 135 GHz

GAIN_BW_HZ = 4.0e12        # gain bandwidth half-width (Hz) — parabolic shape
N_MODES = 5                 # main mode + 2 side modes on each side


def mode_setup(laser, n_modes=N_MODES):
    """Set up mode frequencies and gain shape factors.

    Returns arrays of (detuning_Hz, gain_factor) for each mode.
    Mode 0 = main DFB mode (at gain peak).
    """
    # Mode spacing from cavity
    delta_nu = c / (2 * laser.n_g * laser.L)  # FSR in Hz

    detunings = np.zeros(n_modes)
    for m in range(n_modes):
        # Modes: 0 (main), 1 (+1), 2 (-1), 3 (+2), 4 (-2), ...
        if m == 0:
            detunings[m] = 0.0
        elif m % 2 == 1:
            detunings[m] = +((m + 1) // 2) * delta_nu
        else:
            detunings[m] = -((m) // 2) * delta_nu

    # Parabolic gain shape
    gain_factors = 1.0 - (detunings / GAIN_BW_HZ)**2
    gain_factors = np.maximum(gain_factors, 0.0)

    # DFB grating: side modes see higher mirror loss (weaker grating feedback).
    # FP cavity:   no grating — all modes compete equally within gain BW.
    grating_penalty = np.ones(n_modes)
    if getattr(laser, 'cavity_type', 'dfb') == 'dfb':
        for m in range(1, n_modes):
            order = (m + 1) // 2
            extra_loss = 300.0 * order   # m^-1 extra mirror loss
            g_th = (laser.alpha_i + laser.alpha_m) / laser.Gamma
            grating_penalty[m] = 1.0 - extra_loss / (laser.Gamma * g_th + extra_loss)
    # FP: grating_penalty stays all 1.0 — modes compete on gain shape alone

    effective_gain = gain_factors * grating_penalty

    return detunings, effective_gain, delta_nu


# ── Numba solver: multi-mode gain switching ───────────────────────────

@njit
def simulate_multimode(
        n_pulses, pts_period, dt,
        I_off, I_on, t_on, t_rise,
        V, Gamma, v_g, a_gain, N_tr, eps,
        A_nr, B_rad, C_aug, tau_p, beta_sp, alpha_H, q_e,
        n_modes, gain_eff,   # gain_eff[m] = gain_factor * grating_penalty
        S_inj, seed):
    """Multi-mode Euler-Maruyama solver.

    Each mode has independent E_r, E_i fields.
    All modes share carrier density N and deplete it via stimulated emission.

    Returns:
        Er_all, Ei_all: shape (n_modes, n_pulses * pts_period)
        N_all: shape (n_pulses * pts_period,)
    """
    np.random.seed(seed)
    sqrt_half_dt = np.sqrt(dt / 2.0)

    total_pts = n_pulses * pts_period

    # Output arrays
    Er_all = np.zeros((n_modes, total_pts))
    Ei_all = np.zeros((n_modes, total_pts))
    N_all = np.zeros(total_pts)

    # Initial state
    N = N_tr * 1.2
    Er = np.zeros(n_modes)
    Ei = np.zeros(n_modes)
    # Small random seed for each mode
    for m in range(n_modes):
        Er[m] = np.sqrt(1e8) * (1.0 + 0.1 * np.random.randn())
        Ei[m] = np.sqrt(1e8) * 0.1 * np.random.randn()

    # SLD injection — broadband, enters all modes equally
    R_SLD = S_inj / tau_p
    amp_sld_per_mode = np.sqrt(R_SLD / n_modes) * sqrt_half_dt if R_SLD > 0.0 else 0.0

    idx = 0
    for p in range(n_pulses):
        for k in range(pts_period):
            t = k * dt

            # Current waveform
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

            # Total photon density (all modes)
            S_total = 0.0
            for m in range(n_modes):
                S_total += Er[m] * Er[m] + Ei[m] * Ei[m]
            if S_total < 1e-10:
                S_total = 1e-10

            # Carrier recombination
            R_sp = A_nr * N + B_rad * N * N + C_aug * N * N * N

            # Update each mode's field
            dN_stim = 0.0
            for m in range(n_modes):
                S_m = Er[m] * Er[m] + Ei[m] * Ei[m]
                if S_m < 1e-10:
                    S_m = 1e-10

                # Mode-dependent gain (gain compression uses total S)
                g_m = gain_eff[m] * a_gain * (N - N_tr) / (1.0 + eps * S_total)

                # Net gain rate for this mode
                gamma_m = 0.5 * (Gamma * v_g * g_m - 1.0 / tau_p)

                # Field evolution
                Er_new = (1.0 + gamma_m * dt) * Er[m] - gamma_m * alpha_H * dt * Ei[m]
                Ei_new = (1.0 + gamma_m * dt) * Ei[m] + gamma_m * alpha_H * dt * Er[m]
                Er[m] = Er_new
                Ei[m] = Ei_new

                # Spontaneous emission noise (independent per mode)
                R_sp_mode = beta_sp * B_rad * N * N / n_modes
                amp_sp = np.sqrt(max(R_sp_mode, 0.0)) * sqrt_half_dt
                Er[m] += amp_sp * np.random.randn()
                Ei[m] += amp_sp * np.random.randn()

                # SLD injection noise (independent per mode, broadband)
                if amp_sld_per_mode > 0.0:
                    Er[m] += amp_sld_per_mode * np.random.randn()
                    Ei[m] += amp_sld_per_mode * np.random.randn()

                # Accumulate stimulated emission for carrier equation
                dN_stim += Gamma * v_g * g_m * S_m

            # Carrier dynamics (shared reservoir)
            dN = (I_cur / (q_e * V) - R_sp - dN_stim) * dt
            FN = np.sqrt(2.0 * max(R_sp, 0.0) * dt) * np.random.randn()
            N += dN + FN
            if N < 0:
                N = 0.0

            # Store
            for m in range(n_modes):
                Er_all[m, idx] = Er[m]
                Ei_all[m, idx] = Ei[m]
            N_all[idx] = N
            idx += 1

    return Er_all, Ei_all, N_all


# ── Single-mode solver for comparison ─────────────────────────────────

@njit
def simulate_singlemode(
        n_pulses, pts_period, dt,
        I_off, I_on, t_on, t_rise,
        V, Gamma, v_g, a_gain, N_tr, eps,
        A_nr, B_rad, C_aug, tau_p, beta_sp, alpha_H, q_e,
        S_inj, seed):
    """Standard single-mode solver for comparison."""
    np.random.seed(seed)
    sqrt_half_dt = np.sqrt(dt / 2.0)

    total_pts = n_pulses * pts_period
    Er_out = np.zeros(total_pts)
    Ei_out = np.zeros(total_pts)
    N_out = np.zeros(total_pts)

    N = N_tr * 1.2
    Er = np.sqrt(1e10)
    Ei = 0.0

    R_SLD = S_inj / tau_p
    amp_sld = np.sqrt(R_SLD) * sqrt_half_dt if R_SLD > 0.0 else 0.0

    idx = 0
    for p in range(n_pulses):
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
            R_sp = A_nr * N + B_rad * N * N + C_aug * N * N * N
            R_sp_mode = beta_sp * B_rad * N * N

            gamma = 0.5 * (Gamma * v_g * g - 1.0 / tau_p)
            Er_new = (1.0 + gamma * dt) * Er - gamma * alpha_H * dt * Ei
            Ei_new = (1.0 + gamma * dt) * Ei + gamma * alpha_H * dt * Er
            Er = Er_new
            Ei = Ei_new

            amp_sp = np.sqrt(max(R_sp_mode, 0.0)) * sqrt_half_dt
            Er += amp_sp * np.random.randn()
            Ei += amp_sp * np.random.randn()

            if amp_sld > 0.0:
                Er += amp_sld * np.random.randn()
                Ei += amp_sld * np.random.randn()

            dN = (I_cur / (q_e * V) - R_sp - Gamma * v_g * g * S) * dt
            FN = np.sqrt(2.0 * max(R_sp, 0.0) * dt) * np.random.randn()
            N += dN + FN
            if N < 0:
                N = 0.0

            Er_out[idx] = Er
            Ei_out[idx] = Ei
            N_out[idx] = N
            idx += 1

    return Er_out, Ei_out, N_out


# ── Analysis functions ────────────────────────────────────────────────

def compute_smsr_time(S_modes, dt, pts_period, pulse_idx):
    """Compute SMSR(t) within a single pulse period.

    S_modes: shape (n_modes, total_pts)
    Returns time array and SMSR in dB.
    """
    start = pulse_idx * pts_period
    end = (pulse_idx + 1) * pts_period

    S_main = S_modes[0, start:end]
    S_side_max = np.max(S_modes[1:, start:end], axis=0)

    smsr = 10 * np.log10(np.maximum(S_main, 1e-30) /
                          np.maximum(S_side_max, 1e-30))
    t = np.arange(pts_period) * dt
    return t, smsr


def per_pulse_metrics(Er_all, Ei_all, pts_period, n_discard):
    """Extract per-pulse peak phase, peak power, and mode powers."""
    n_modes = Er_all.shape[0]
    total_pts = Er_all.shape[1]
    n_total_pulses = total_pts // pts_period
    n_out = n_total_pulses - n_discard

    # Per-mode peak power per pulse
    peak_S = np.zeros((n_modes, n_out))
    # Main mode peak phase
    peak_phi = np.zeros(n_out)
    # Total peak power
    peak_total = np.zeros(n_out)

    for p in range(n_discard, n_total_pulses):
        start = p * pts_period
        end = (p + 1) * pts_period
        out_idx = p - n_discard

        for m in range(n_modes):
            S_m = Er_all[m, start:end]**2 + Ei_all[m, start:end]**2
            peak_S[m, out_idx] = np.max(S_m)

        # Total intensity
        S_tot = np.zeros(end - start)
        for m in range(n_modes):
            S_tot += Er_all[m, start:end]**2 + Ei_all[m, start:end]**2

        pk_idx = np.argmax(S_tot)
        peak_total[out_idx] = S_tot[pk_idx]

        # Phase from main mode at total peak
        peak_phi[out_idx] = np.arctan2(
            Ei_all[0, start + pk_idx], Er_all[0, start + pk_idx])

    return peak_phi, peak_S, peak_total


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    setup_plotting()

    print("=" * 70)
    print("  Multi-Mode Competition Under Gain Switching")
    print("=" * 70)

    laser = DFBLaserParams()
    I_th = laser.threshold_current()

    DT = 1.0e-12
    DUTY = 0.30
    I_BIAS_FACTOR = 0.9
    I_PEAK_FACTOR = 5.0
    I_off = I_BIAS_FACTOR * I_th
    I_on = I_PEAK_FACTOR * I_th

    detunings, gain_eff, delta_nu = mode_setup(laser, N_MODES)

    print(f"\n  Laser: lambda = {laser.lambda0*1e9:.0f} nm, "
          f"alpha_H = {laser.alpha_H}, tau_p = {laser.tau_p*1e12:.2f} ps")
    print(f"  Mode spacing: {delta_nu*1e-9:.1f} GHz  "
          f"({delta_nu * laser.lambda0**2 / c * 1e9:.2f} nm)")
    print(f"  Gain bandwidth: {GAIN_BW_HZ*1e-12:.1f} THz (half-width)")
    print(f"\n  Mode setup ({N_MODES} modes):")
    for m in range(N_MODES):
        print(f"    Mode {m}: detuning = {detunings[m]*1e-9:+7.1f} GHz, "
              f"gain factor = {gain_eff[m]:.4f}")

    # Compile
    print("\n  Compiling JIT solvers...")
    t0 = _time.time()
    _w1 = simulate_multimode(
        4, 50, 1e-12, I_off, I_on, 15e-12, 5e-12,
        laser.V, laser.Gamma, laser.v_g, laser.a, laser.N_tr, laser.epsilon,
        laser.A, laser.B, laser.C, laser.tau_p, laser.beta_sp, laser.alpha_H, q,
        N_MODES, gain_eff, 0.0, 0)
    _w2 = simulate_singlemode(
        4, 50, 1e-12, I_off, I_on, 15e-12, 5e-12,
        laser.V, laser.Gamma, laser.v_g, laser.a, laser.N_tr, laser.epsilon,
        laser.A, laser.B, laser.C, laser.tau_p, laser.beta_sp, laser.alpha_H, q,
        0.0, 0)
    del _w1, _w2
    print(f"  Compiled in {_time.time()-t0:.1f}s")

    # ── Part 1: Time-resolved mode dynamics (few pulses) ──────────────

    freqs = [2e9, 5e9, 10e9]
    N_PULSES_WAVEFORM = 20
    PULSE_SHOW = 10        # which pulse to show in time-resolved plots

    print(f"\n  Part 1: Time-resolved waveforms ({N_PULSES_WAVEFORM} pulses)")

    waveform_data = {}

    for f_rep in freqs:
        T_rep = 1.0 / f_rep
        t_on = DUTY * T_rep
        pts = max(int(round(T_rep / DT)), 100)
        dt = T_rep / pts
        t_rise = min(20e-12, t_on / 4.0)
        fg = f_rep * 1e-9

        for mode_label, S_inj_val in [('free', 0.0), ('sld', 3e19)]:
            seed = 42 + int(fg) * 100 + (500 if mode_label == 'sld' else 0)
            key = f'{fg:.0f}GHz_{mode_label}'

            print(f"\n    {fg:.0f} GHz, {mode_label}: ", end="", flush=True)
            t0 = _time.time()

            Er, Ei, N_carr = simulate_multimode(
                N_PULSES_WAVEFORM, pts, dt,
                I_off, I_on, t_on, t_rise,
                laser.V, laser.Gamma, laser.v_g, laser.a,
                laser.N_tr, laser.epsilon,
                laser.A, laser.B, laser.C,
                laser.tau_p, laser.beta_sp, laser.alpha_H, q,
                N_MODES, gain_eff, S_inj_val, seed)

            elapsed = _time.time() - t0
            print(f"{elapsed:.2f}s")

            S_modes = Er**2 + Ei**2
            waveform_data[key] = dict(
                Er=Er, Ei=Ei, N=N_carr, S_modes=S_modes,
                dt=dt, pts=pts, t_on=t_on)

    # ── Part 2: Statistical analysis (many pulses) ────────────────────

    N_PULSES_STAT = 50_000
    N_DISCARD_STAT = 100

    print(f"\n  Part 2: Statistical analysis ({N_PULSES_STAT//1000}k pulses)")

    stat_data = {}

    for f_rep in freqs:
        T_rep = 1.0 / f_rep
        t_on = DUTY * T_rep
        pts = max(int(round(T_rep / DT)), 100)
        dt = T_rep / pts
        t_rise = min(20e-12, t_on / 4.0)
        fg = f_rep * 1e-9

        for mode_label, S_inj_val in [('free', 0.0), ('sld', 3e19)]:
            seed = 42 + int(fg) * 100 + (500 if mode_label == 'sld' else 0)
            key = f'{fg:.0f}GHz_{mode_label}'

            # Multi-mode
            print(f"    {fg:.0f} GHz {mode_label} multi ...", end="", flush=True)
            t0 = _time.time()
            Er_m, Ei_m, _ = simulate_multimode(
                N_PULSES_STAT + N_DISCARD_STAT, pts, dt,
                I_off, I_on, t_on, t_rise,
                laser.V, laser.Gamma, laser.v_g, laser.a,
                laser.N_tr, laser.epsilon,
                laser.A, laser.B, laser.C,
                laser.tau_p, laser.beta_sp, laser.alpha_H, q,
                N_MODES, gain_eff, S_inj_val, seed)
            elapsed = _time.time() - t0

            phi_m, pkS_m, pkTot_m = per_pulse_metrics(
                Er_m, Ei_m, pts, N_DISCARD_STAT)
            print(f" {elapsed:.1f}s", flush=True)

            # Single-mode
            print(f"    {fg:.0f} GHz {mode_label} single ...", end="", flush=True)
            t0 = _time.time()
            Er_s, Ei_s, _ = simulate_singlemode(
                N_PULSES_STAT + N_DISCARD_STAT, pts, dt,
                I_off, I_on, t_on, t_rise,
                laser.V, laser.Gamma, laser.v_g, laser.a,
                laser.N_tr, laser.epsilon,
                laser.A, laser.B, laser.C,
                laser.tau_p, laser.beta_sp, laser.alpha_H, q,
                S_inj_val, seed)
            elapsed = _time.time() - t0

            # Extract single-mode per-pulse metrics
            S_single = Er_s**2 + Ei_s**2
            n_total = (N_PULSES_STAT + N_DISCARD_STAT)
            n_out = N_PULSES_STAT
            pk_phi_s = np.zeros(n_out)
            pk_S_s = np.zeros(n_out)
            for p in range(N_DISCARD_STAT, n_total):
                start = p * pts
                end = (p + 1) * pts
                seg = S_single[start:end]
                pk_idx = np.argmax(seg)
                pk_S_s[p - N_DISCARD_STAT] = seg[pk_idx]
                pk_phi_s[p - N_DISCARD_STAT] = np.arctan2(
                    Ei_s[start + pk_idx], Er_s[start + pk_idx])

            print(f" {elapsed:.1f}s", flush=True)

            # Phase correlation
            r1_multi, dphi_m = compute_r1(phi_m)
            r1_single, dphi_s = compute_r1(pk_phi_s)

            # Intensity CV
            P_total_m = laser.output_power(np.maximum(pkTot_m, 0))
            P_single = laser.output_power(np.maximum(pk_S_s, 0))
            cv_multi = float(np.std(P_total_m) / np.mean(P_total_m))
            cv_single = float(np.std(P_single) / np.mean(P_single))

            # SMSR statistics (from multi-mode per-pulse peaks)
            smsr_vals = 10 * np.log10(
                np.maximum(pkS_m[0], 1e-30) /
                np.maximum(np.max(pkS_m[1:], axis=0), 1e-30))

            # Mode partition coefficient k
            # k = sqrt(1 - var(P_main)/var(P_total))
            P_main = laser.output_power(np.maximum(pkS_m[0], 0))
            k_mpn = np.sqrt(max(0,
                1.0 - np.var(P_main) / max(np.var(P_total_m), 1e-30)))

            # Anti-correlation between main and side modes
            P_sides = laser.output_power(
                np.maximum(np.sum(pkS_m[1:], axis=0), 0))
            corr_main_side = float(np.corrcoef(P_main, P_sides)[0, 1])

            stat_data[key] = dict(
                r1_multi=r1_multi, r1_single=r1_single,
                cv_multi=cv_multi, cv_single=cv_single,
                smsr_mean=float(np.mean(smsr_vals)),
                smsr_std=float(np.std(smsr_vals)),
                smsr_min=float(np.min(smsr_vals)),
                k_mpn=k_mpn,
                corr_main_side=corr_main_side,
                P_mean_multi=float(np.mean(P_total_m)),
                P_mean_single=float(np.mean(P_single)),
                pkS_modes=pkS_m,
                smsr_vals=smsr_vals,
            )

            print(f"      r1: {r1_multi:.4f} (multi) vs {r1_single:.4f} (single)")
            print(f"      CV: {cv_multi:.4f} (multi) vs {cv_single:.4f} (single)")
            print(f"      SMSR: {np.mean(smsr_vals):.1f} +/- {np.std(smsr_vals):.1f} dB "
                  f"(min {np.min(smsr_vals):.1f} dB)")
            print(f"      MPN k = {k_mpn:.4f}, "
                  f"corr(main,side) = {corr_main_side:.3f}")

    # ── Figure 1: Time-resolved mode dynamics ─────────────────────────

    fig1, axes1 = plt.subplots(3, 3, figsize=(18, 14))
    fig1.suptitle(
        'Multi-Mode Gain-Switched Dynamics — Free-Running\n'
        f'{N_MODES} modes, gain BW = {GAIN_BW_HZ*1e-12:.0f} THz, '
        f'mode spacing = {delta_nu*1e-9:.0f} GHz',
        fontsize=13)

    mode_colors = ['C0', 'C1', 'C3', 'C4', 'C5']
    mode_labels = ['Main', '+1', '-1', '+2', '-2']

    for col, f_rep in enumerate(freqs):
        fg = f_rep * 1e-9
        key = f'{fg:.0f}GHz_free'
        d = waveform_data[key]
        dt_val = d['dt']
        pts_val = d['pts']
        t_on_val = d['t_on']
        pulse = PULSE_SHOW

        start = pulse * pts_val
        end = (pulse + 1) * pts_val
        t_ps = np.arange(pts_val) * dt_val * 1e12

        # Row 0: Mode powers
        ax = axes1[0, col]
        for m in range(N_MODES):
            P_m = laser.output_power(np.maximum(d['S_modes'][m, start:end], 0))
            ax.plot(t_ps, P_m * 1e3, color=mode_colors[m], lw=1.5,
                    label=mode_labels[m], alpha=0.9 if m == 0 else 0.7)
        ax.set_title(f'{fg:.0f} GHz')
        ax.set_ylabel('Power per mode (mW)')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        # Row 1: SMSR(t)
        ax = axes1[1, col]
        t_smsr, smsr = compute_smsr_time(d['S_modes'], dt_val, pts_val, pulse)
        ax.plot(t_smsr * 1e12, smsr, 'k-', lw=1.5)
        ax.axhline(30, color='r', ls='--', lw=1, alpha=0.7, label='30 dB target')
        ax.set_ylabel('SMSR (dB)')
        ax.set_ylim(-10, 60)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Row 2: Carrier density
        ax = axes1[2, col]
        ax.plot(t_ps, d['N'][start:end] * 1e-24, 'C2-', lw=1.5)
        ax.set_ylabel('Carrier density ($\\times 10^{24}$ m$^{-3}$)')
        ax.set_xlabel('Time (ps)')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig1.savefig('images/multimode/time_resolved_free.png', dpi=150)
    print("\n  Saved: images/multimode/time_resolved_free.png")

    # ── Figure 2: Time-resolved — SLD-injected ───────────────────────

    fig2, axes2 = plt.subplots(3, 3, figsize=(18, 14))
    fig2.suptitle(
        'Multi-Mode Gain-Switched Dynamics — SLD-Injected\n'
        f'{N_MODES} modes, $S_{{inj}}$ = 3$\\times 10^{{19}}$ m$^{{-3}}$',
        fontsize=13)

    for col, f_rep in enumerate(freqs):
        fg = f_rep * 1e-9
        key = f'{fg:.0f}GHz_sld'
        d = waveform_data[key]
        dt_val = d['dt']
        pts_val = d['pts']
        pulse = PULSE_SHOW

        start = pulse * pts_val
        end = (pulse + 1) * pts_val
        t_ps = np.arange(pts_val) * dt_val * 1e12

        ax = axes2[0, col]
        for m in range(N_MODES):
            P_m = laser.output_power(np.maximum(d['S_modes'][m, start:end], 0))
            ax.plot(t_ps, P_m * 1e3, color=mode_colors[m], lw=1.5,
                    label=mode_labels[m], alpha=0.9 if m == 0 else 0.7)
        ax.set_title(f'{fg:.0f} GHz')
        ax.set_ylabel('Power per mode (mW)')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        ax = axes2[1, col]
        t_smsr, smsr = compute_smsr_time(d['S_modes'], dt_val, pts_val, pulse)
        ax.plot(t_smsr * 1e12, smsr, 'k-', lw=1.5)
        ax.axhline(30, color='r', ls='--', lw=1, alpha=0.7, label='30 dB target')
        ax.set_ylabel('SMSR (dB)')
        ax.set_ylim(-10, 60)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes2[2, col]
        ax.plot(t_ps, d['N'][start:end] * 1e-24, 'C2-', lw=1.5)
        ax.set_ylabel('Carrier density ($\\times 10^{24}$ m$^{-3}$)')
        ax.set_xlabel('Time (ps)')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig2.savefig('images/multimode/time_resolved_sld.png', dpi=150)
    print("  Saved: images/multimode/time_resolved_sld.png")

    # ── Figure 3: SMSR distribution and mode partition noise ──────────

    fig3, axes3 = plt.subplots(2, 3, figsize=(18, 10))
    fig3.suptitle(
        'Mode Partition Noise Statistics — 100k Pulses\n'
        'Free-running (solid) vs SLD-injected (dashed)',
        fontsize=13)

    for col, f_rep in enumerate(freqs):
        fg = f_rep * 1e-9

        # SMSR histogram
        ax = axes3[0, col]
        for mode_label, ls, alpha_val in [('free', '-', 0.7), ('sld', '--', 0.5)]:
            key = f'{fg:.0f}GHz_{mode_label}'
            sd = stat_data[key]
            label_txt = f'{"Free" if mode_label == "free" else "SLD"}: ' \
                       f'{sd["smsr_mean"]:.1f}$\\pm${sd["smsr_std"]:.1f} dB'
            ax.hist(sd['smsr_vals'], bins=80, density=True,
                    alpha=alpha_val, label=label_txt,
                    histtype='stepfilled' if mode_label == 'free' else 'step',
                    lw=2)
        ax.axvline(30, color='r', ls='--', lw=1, alpha=0.7)
        ax.set_xlabel('SMSR (dB)')
        ax.set_ylabel('Probability density')
        ax.set_title(f'{fg:.0f} GHz')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Main vs side mode power scatter
        ax = axes3[1, col]
        for mode_label, marker, alpha_val in [('free', '.', 0.1), ('sld', '.', 0.1)]:
            key = f'{fg:.0f}GHz_{mode_label}'
            sd = stat_data[key]
            pkS = sd['pkS_modes']
            P_main = laser.output_power(np.maximum(pkS[0], 0)) * 1e3
            P_side = laser.output_power(
                np.maximum(np.sum(pkS[1:], axis=0), 0)) * 1e3

            # Subsample for scatter
            n_show = min(5000, len(P_main))
            idx = np.random.choice(len(P_main), n_show, replace=False)
            lbl = 'Free' if mode_label == 'free' else 'SLD'
            color = 'C0' if mode_label == 'free' else 'C1'
            ax.scatter(P_main[idx], P_side[idx], s=1, alpha=0.15,
                       color=color, label=lbl)

        ax.set_xlabel('Main mode peak power (mW)')
        ax.set_ylabel('Side modes peak power (mW)')
        ax.set_title(f'{fg:.0f} GHz — Anti-correlation')
        ax.legend(fontsize=8, markerscale=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig3.savefig('images/multimode/mpn_statistics.png', dpi=150)
    print("  Saved: images/multimode/mpn_statistics.png")

    # ── Figure 4: Multi-mode vs single-mode comparison ────────────────

    fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))
    fig4.suptitle(
        'Impact of Multi-Mode Competition on Pulse Statistics\n'
        'Single-mode (circles) vs 5-mode (squares)',
        fontsize=13)

    f_ghz = np.array([f * 1e-9 for f in freqs])

    for mode_label, ls, fill in [('free', '-', True), ('sld', '--', False)]:
        r1_m = [stat_data[f'{fg:.0f}GHz_{mode_label}']['r1_multi'] for fg in f_ghz]
        r1_s = [stat_data[f'{fg:.0f}GHz_{mode_label}']['r1_single'] for fg in f_ghz]
        cv_m = [stat_data[f'{fg:.0f}GHz_{mode_label}']['cv_multi'] for fg in f_ghz]
        cv_s = [stat_data[f'{fg:.0f}GHz_{mode_label}']['cv_single'] for fg in f_ghz]
        k_vals = [stat_data[f'{fg:.0f}GHz_{mode_label}']['k_mpn'] for fg in f_ghz]
        corr_vals = [stat_data[f'{fg:.0f}GHz_{mode_label}']['corr_main_side']
                     for fg in f_ghz]
        lbl = 'Free' if mode_label == 'free' else 'SLD'
        color = 'C0' if mode_label == 'free' else 'C1'

        axes4[0, 0].plot(f_ghz, r1_m, f's{ls}', color=color, lw=2, ms=7,
                         label=f'{lbl} multi', markerfacecolor=color if fill else 'none')
        axes4[0, 0].plot(f_ghz, r1_s, f'o{ls}', color=color, lw=2, ms=7,
                         label=f'{lbl} single', markerfacecolor=color if fill else 'none',
                         alpha=0.6)

        axes4[0, 1].plot(f_ghz, cv_m, f's{ls}', color=color, lw=2, ms=7,
                         label=f'{lbl} multi', markerfacecolor=color if fill else 'none')
        axes4[0, 1].plot(f_ghz, cv_s, f'o{ls}', color=color, lw=2, ms=7,
                         label=f'{lbl} single', markerfacecolor=color if fill else 'none',
                         alpha=0.6)

        axes4[1, 0].plot(f_ghz, k_vals, f's{ls}', color=color, lw=2, ms=7,
                         label=f'{lbl}', markerfacecolor=color if fill else 'none')

        axes4[1, 1].plot(f_ghz, corr_vals, f's{ls}', color=color, lw=2, ms=7,
                         label=f'{lbl}', markerfacecolor=color if fill else 'none')

    axes4[0, 0].set_ylabel('Phase correlation $r_1$')
    axes4[0, 0].set_title('Phase coherence')
    axes4[0, 1].set_ylabel('Intensity CV')
    axes4[0, 1].set_title('Intensity noise')
    axes4[1, 0].set_ylabel('MPN coefficient $k$')
    axes4[1, 0].set_title('Mode partition noise')
    axes4[1, 1].set_ylabel('Correlation(main, side)')
    axes4[1, 1].set_title('Main-side anti-correlation')
    axes4[1, 1].axhline(0, color='gray', ls=':', alpha=0.5)

    for ax in axes4.flat:
        ax.set_xlabel('Repetition rate (GHz)')
        ax.set_xticks(f_ghz)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig4.savefig('images/multimode/multimode_vs_singlemode.png', dpi=150)
    print("  Saved: images/multimode/multimode_vs_singlemode.png")

    # ── Figure 5: Pulse-train overlay — 5 consecutive pulses ─────────

    fig5, axes5 = plt.subplots(2, 3, figsize=(18, 10))
    fig5.suptitle(
        'Multi-Pulse Mode Dynamics — 5 Consecutive Pulses\n'
        'Showing pulse-to-pulse variation in mode partition',
        fontsize=13)

    for col, f_rep in enumerate(freqs):
        fg = f_rep * 1e-9

        for row, (mode_label, title) in enumerate([
            ('free', 'Free-running'), ('sld', 'SLD-injected')
        ]):
            key = f'{fg:.0f}GHz_{mode_label}'
            d = waveform_data[key]
            dt_val = d['dt']
            pts_val = d['pts']
            ax = axes5[row, col]

            # Show 5 pulses starting from PULSE_SHOW
            n_show = 5
            for p_offset in range(n_show):
                pulse = PULSE_SHOW + p_offset
                start = pulse * pts_val
                end = (pulse + 1) * pts_val
                t_ps = np.arange(pts_val) * dt_val * 1e12

                # Total power
                S_tot = np.sum(d['S_modes'][:, start:end], axis=0)
                P_tot = laser.output_power(np.maximum(S_tot, 0)) * 1e3

                # Main mode fraction
                S_main = d['S_modes'][0, start:end]
                frac_main = S_main / np.maximum(S_tot, 1e-30)

                # Color by main mode fraction: blue=100% main, red=side dominant
                ax.plot(t_ps + p_offset * pts_val * dt_val * 1e12,
                        P_tot, 'k-', lw=0.8, alpha=0.5)

                # Overlay main mode
                P_main = laser.output_power(
                    np.maximum(d['S_modes'][0, start:end], 0)) * 1e3
                ax.fill_between(
                    t_ps + p_offset * pts_val * dt_val * 1e12,
                    0, P_main, alpha=0.3, color='C0', label='Main' if p_offset == 0 else '')
                P_side_sum = P_tot - P_main
                ax.fill_between(
                    t_ps + p_offset * pts_val * dt_val * 1e12,
                    P_main, P_tot, alpha=0.3, color='C3',
                    label='Side modes' if p_offset == 0 else '')

            ax.set_title(f'{fg:.0f} GHz — {title}')
            ax.set_xlabel('Time (ps)')
            ax.set_ylabel('Power (mW)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig5.savefig('images/multimode/pulse_train_modes.png', dpi=150)
    print("  Saved: images/multimode/pulse_train_modes.png")

    plt.close('all')

    # ── Part 3: DFB vs Fabry-Perot cavity comparison ─────────────────

    print(f"\n  Part 3: DFB vs Fabry-Perot cavity comparison (5 GHz)")

    fp_laser = make_laser('fp')
    fp_I_th = fp_laser.threshold_current()
    fp_I_off = I_BIAS_FACTOR * fp_I_th
    fp_I_on = I_PEAK_FACTOR * fp_I_th

    detunings_fp, gain_eff_fp, delta_nu_fp = mode_setup(fp_laser, N_MODES)

    print(f"\n    DFB: tau_p = {laser.tau_p*1e12:.2f} ps, "
          f"I_th = {I_th*1e3:.1f} mA, alpha_m = {laser.alpha_m:.0f} m^-1")
    print(f"    FP:  tau_p = {fp_laser.tau_p*1e12:.2f} ps, "
          f"I_th = {fp_I_th*1e3:.1f} mA, alpha_m = {fp_laser.alpha_m:.0f} m^-1")
    print(f"\n    Mode gain factors (shape × grating):")
    for m in range(N_MODES):
        print(f"      Mode {m}: DFB = {gain_eff[m]:.4f}, FP = {gain_eff_fp[m]:.4f}")

    # Run FP at 5 GHz
    f_rep_fp = 5e9
    T_rep_fp = 1.0 / f_rep_fp
    t_on_fp = DUTY * T_rep_fp
    pts_fp = max(int(round(T_rep_fp / DT)), 100)
    dt_fp = T_rep_fp / pts_fp
    t_rise_fp = min(20e-12, t_on_fp / 4.0)

    fp_stat = {}
    for mode_label, S_inj_val in [('free', 0.0), ('sld', 3e19)]:
        seed = 7777 + (500 if mode_label == 'sld' else 0)

        print(f"    FP 5 GHz {mode_label} multi ...", end="", flush=True)
        t0 = _time.time()
        Er_fp, Ei_fp, _ = simulate_multimode(
            N_PULSES_STAT + N_DISCARD_STAT, pts_fp, dt_fp,
            fp_I_off, fp_I_on, t_on_fp, t_rise_fp,
            fp_laser.V, fp_laser.Gamma, fp_laser.v_g, fp_laser.a,
            fp_laser.N_tr, fp_laser.epsilon,
            fp_laser.A, fp_laser.B, fp_laser.C,
            fp_laser.tau_p, fp_laser.beta_sp, fp_laser.alpha_H, q,
            N_MODES, gain_eff_fp, S_inj_val, seed)
        elapsed = _time.time() - t0

        phi_fp, pkS_fp, pkTot_fp = per_pulse_metrics(
            Er_fp, Ei_fp, pts_fp, N_DISCARD_STAT)
        print(f" {elapsed:.1f}s", flush=True)

        r1_fp, _ = compute_r1(phi_fp)
        P_total_fp = fp_laser.output_power(np.maximum(pkTot_fp, 0))
        cv_fp = float(np.std(P_total_fp) / np.mean(P_total_fp))
        smsr_fp = 10 * np.log10(
            np.maximum(pkS_fp[0], 1e-30) /
            np.maximum(np.max(pkS_fp[1:], axis=0), 1e-30))
        P_main_fp = fp_laser.output_power(np.maximum(pkS_fp[0], 0))
        k_mpn_fp = np.sqrt(max(0,
            1.0 - np.var(P_main_fp) / max(np.var(P_total_fp), 1e-30)))

        fp_stat[mode_label] = dict(
            r1=r1_fp, cv=cv_fp,
            smsr_mean=float(np.mean(smsr_fp)),
            smsr_std=float(np.std(smsr_fp)),
            k_mpn=k_mpn_fp, smsr_vals=smsr_fp,
            pkS_modes=pkS_fp,
        )

        print(f"      r1 = {r1_fp:.4f}, CV = {cv_fp:.4f}, "
              f"SMSR = {np.mean(smsr_fp):.1f}+/-{np.std(smsr_fp):.1f} dB, "
              f"k = {k_mpn_fp:.4f}")

    # ── Figure 6: DFB vs FP comparison ───────────────────────────────

    fig6, axes6 = plt.subplots(2, 2, figsize=(14, 10))
    fig6.suptitle(
        'DFB vs Fabry-Perot Cavity — 5 GHz Gain Switching\n'
        f'DFB: R$_1$=0.32, R$_2$=0.95  |  FP: R$_1$=R$_2$=0.32  |  '
        f'{N_MODES} modes',
        fontsize=13)

    # Panel (0,0): SMSR distribution
    ax = axes6[0, 0]
    dfb_5g = stat_data['5GHz_free']
    ax.hist(dfb_5g['smsr_vals'], bins=80, density=True, alpha=0.6,
            color='C0', label=f'DFB: {dfb_5g["smsr_mean"]:.0f} dB')
    ax.hist(fp_stat['free']['smsr_vals'], bins=80, density=True, alpha=0.5,
            color='C1', histtype='step', lw=2,
            label=f'FP: {fp_stat["free"]["smsr_mean"]:.0f} dB')
    ax.axvline(30, color='r', ls='--', lw=1, alpha=0.7)
    ax.set_xlabel('SMSR (dB)')
    ax.set_ylabel('Probability density')
    ax.set_title('SMSR — Free-Running')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel (0,1): Mode power spectrum
    ax = axes6[0, 1]
    modes_x = np.arange(N_MODES)
    mode_names = ['Main', '+1', '-1', '+2', '-2'][:N_MODES]

    for cav_label, pkS_m, laser_ref, color, offset in [
        ('DFB', dfb_5g['pkS_modes'], laser, 'C0', -0.15),
        ('FP', fp_stat['free']['pkS_modes'], fp_laser, 'C1', 0.15),
    ]:
        mean_pk = np.mean(pkS_m, axis=1)
        P_modes = laser_ref.output_power(np.maximum(mean_pk, 0)) * 1e3
        ax.bar(modes_x + offset, P_modes, width=0.3, color=color,
               alpha=0.7, label=cav_label)

    ax.set_xticks(modes_x)
    ax.set_xticklabels(mode_names)
    ax.set_xlabel('Mode')
    ax.set_ylabel('Mean peak power (mW)')
    ax.set_title('Mode Spectrum — Free-Running')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel (1,0): Metrics comparison bar chart
    ax = axes6[1, 0]
    metric_names = ['$r_1$', 'Intensity CV', 'MPN $k$']
    x = np.arange(len(metric_names))
    width = 0.18
    bars = [
        ('DFB Free', [dfb_5g['r1_multi'], dfb_5g['cv_multi'],
                       dfb_5g['k_mpn']], 'C0', ''),
        ('DFB SLD', [stat_data['5GHz_sld']['r1_multi'],
                      stat_data['5GHz_sld']['cv_multi'],
                      stat_data['5GHz_sld']['k_mpn']], 'C0', '//'),
        ('FP Free', [fp_stat['free']['r1'], fp_stat['free']['cv'],
                      fp_stat['free']['k_mpn']], 'C1', ''),
        ('FP SLD', [fp_stat['sld']['r1'], fp_stat['sld']['cv'],
                     fp_stat['sld']['k_mpn']], 'C1', '//'),
    ]
    for i, (lbl, vals, col, hatch) in enumerate(bars):
        ax.bar(x + (i - 1.5) * width, vals, width, label=lbl,
               color=col, alpha=0.7 if not hatch else 0.4,
               hatch=hatch, edgecolor='k', lw=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel('Value')
    ax.set_title('Key Metrics — DFB vs FP')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel (1,1): Parameter summary text
    ax = axes6[1, 1]
    ax.axis('off')
    summary_lines = (
        f"DFB (R1=0.32, R2=0.95):\n"
        f"  alpha_m = {laser.alpha_m:.0f} m-1\n"
        f"  tau_p   = {laser.tau_p*1e12:.2f} ps\n"
        f"  I_th    = {I_th*1e3:.1f} mA\n"
        f"  frac_front = {(1-laser.R1)/((1-laser.R1)+(1-laser.R2)):.2f}\n\n"
        f"FP (R1=R2=0.32):\n"
        f"  alpha_m = {fp_laser.alpha_m:.0f} m-1\n"
        f"  tau_p   = {fp_laser.tau_p*1e12:.2f} ps\n"
        f"  I_th    = {fp_I_th*1e3:.1f} mA\n"
        f"  frac_front = {(1-fp_laser.R1)/((1-fp_laser.R1)+(1-fp_laser.R2)):.2f}\n\n"
        f"Key differences:\n"
        f"  * FP mirror loss {fp_laser.alpha_m/laser.alpha_m:.1f}x higher\n"
        f"  * No grating: modes compete equally\n"
        f"  * Lower SMSR, higher MPN\n"
        f"  * 50/50 front/rear output coupling"
    )
    ax.text(0.05, 0.95, summary_lines, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax.set_title('Parameter Comparison')

    plt.tight_layout()
    save_fig(fig6, 'images/multimode/dfb_vs_fp.png')

    plt.close('all')

    # ── Summary ──────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("  Summary: Multi-Mode Competition")
    print("=" * 70)

    print(f"\n  {'Config':>16s}  {'r1_M':>6s}  {'r1_S':>6s}  "
          f"{'CV_M':>6s}  {'CV_S':>6s}  "
          f"{'SMSR':>10s}  {'k_MPN':>6s}  {'corr':>6s}")
    print("  " + "-" * 72)

    for f_rep in freqs:
        fg = f_rep * 1e-9
        for mode_label in ['free', 'sld']:
            key = f'{fg:.0f}GHz_{mode_label}'
            sd = stat_data[key]
            tag = f'{fg:.0f}GHz {mode_label}'
            print(f"  {tag:>16s}  {sd['r1_multi']:6.4f}  {sd['r1_single']:6.4f}  "
                  f"{sd['cv_multi']:6.4f}  {sd['cv_single']:6.4f}  "
                  f"{sd['smsr_mean']:5.1f}+/-{sd['smsr_std']:3.1f}  "
                  f"{sd['k_mpn']:6.4f}  {sd['corr_main_side']:+6.3f}")

    print(f"\n  FP cavity comparison (5 GHz, {N_MODES} modes):")
    print("  " + "-" * 72)
    for mode_label in ['free', 'sld']:
        sd = fp_stat[mode_label]
        tag = f'FP 5GHz {mode_label}'
        print(f"  {tag:>16s}  {sd['r1']:6.4f}  {'   -  ':>6s}  "
              f"{sd['cv']:6.4f}  {'   -  ':>6s}  "
              f"{sd['smsr_mean']:5.1f}+/-{sd['smsr_std']:3.1f}  "
              f"{sd['k_mpn']:6.4f}  {'   -  ':>6s}")

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)
