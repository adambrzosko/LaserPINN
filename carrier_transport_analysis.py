"""
Carrier transport effects on gain-switched pulse dynamics.

At high repetition rates (>5 GHz), the carrier capture time into the
quantum well (tau_cap ~ 1-5 ps) becomes a significant fraction of the
modulation period.  This delays turn-on, reduces modulation depth, and
changes phase/timing jitter.

Extended rate equations:
  dN_b/dt = I/(q*V_b) - N_b/tau_cap + N_w/(chi*tau_esc) - N_b/tau_b
  dN_w/dt = chi*N_b/tau_cap - N_w/tau_esc - R(N_w) - v_g*g(N_w)*S
  dS/dt   = [Gamma*v_g*g(N_w) - 1/tau_p]*S + beta_sp*B*N_w^2
  dphi/dt = (alpha_H/2) * [Gamma*v_g*a*(N_w - N_tr) - 1/tau_p]

where chi = V_b/V_w (barrier-to-QW volume ratio) and the current
injects into the barrier, not the QW directly.

Compares to the standard model (tau_cap = 0) across 1-10 GHz.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time as _time
from numba import njit

from dfb_laser import DFBLaserParams, q, h, c


# ── Transport parameters ────────────────────────────────────────────────

# Volume ratio: SCH barrier vs QW active layer.
# Typical InGaAsP DFB: SCH width ~ 0.1-0.5 um on each side,
# QW thickness ~ 5-10 nm × N_wells, total d_QW ~ 20-40 nm.
# chi = V_barrier / V_QW ~ 5-15 (same area, different thickness).
CHI = 5.0

# Carrier capture time into QW (barrier -> QW).
# Measured values: 0.5-10 ps for InGaAsP/InP MQW structures.
# Dominant mechanism: thermionic + phonon-assisted capture.
TAU_CAP_DEFAULT = 3.0e-12   # 3 ps

# Carrier escape time (QW -> barrier).
# Related to capture by detailed balance: tau_esc = tau_cap * f(T, dE)
# For deep QWs at 300K with dE ~ 100 meV: tau_esc >> tau_cap.
TAU_ESC = 100e-12   # 100 ps (deep QW, negligible escape)

# Barrier non-radiative recombination time.
# Much slower than capture for good-quality material.
TAU_B = 1e-9   # 1 ns


# ── Numba solver with carrier transport ─────────────────────────────────

@njit
def simulate_pulses_transport(
        n_pulses, n_discard, pts_period, dt,
        I_off, I_on, t_on, t_rise,
        # QW / laser params (same as original)
        V_w, Gamma, v_g, a_gain, N_tr, eps,
        A, B, C_aug, tau_p, beta_sp, alpha_H, q_e,
        # Transport params
        chi, tau_cap, tau_esc, tau_b,
        # Injection
        S_inj, seed):
    """Euler-Maruyama solver with barrier carrier dynamics."""
    np.random.seed(seed)
    sqrt_half_dt = np.sqrt(dt / 2.0)

    V_b = chi * V_w
    n_out = n_pulses - n_discard
    peak_phi = np.empty(n_out)
    peak_S = np.empty(n_out)
    samp_S = np.empty(n_out)
    peak_k = np.empty(n_out, dtype=np.int64)

    # Initial conditions: guess steady state at I_off
    N_w = N_tr * 1.2
    N_b = I_off * tau_cap / (q_e * V_b) if tau_cap > 0 else 0.0
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

            # Current waveform (raised cosine edges)
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

            # --- QW photon / field dynamics ---
            S = Er * Er + Ei * Ei
            if S < 1e-10:
                S = 1e-10

            g = a_gain * (N_w - N_tr) / (1.0 + eps * S)
            R_sp_w = A * N_w + B * N_w * N_w + C_aug * N_w * N_w * N_w
            R_sp_mode = beta_sp * B * N_w * N_w

            gamma = 0.5 * (Gamma * v_g * g - 1.0 / tau_p)

            # Field evolution (same as original)
            Er_new = (1.0 + gamma * dt) * Er - gamma * alpha_H * dt * Ei
            Ei_new = (1.0 + gamma * dt) * Ei + gamma * alpha_H * dt * Er
            Er = Er_new
            Ei = Ei_new

            # Spontaneous emission noise
            amp_sp = np.sqrt(R_sp_mode) * sqrt_half_dt
            Er += amp_sp * np.random.randn()
            Ei += amp_sp * np.random.randn()

            # SLD injection noise
            if amp_sld > 0.0:
                Er += amp_sld * np.random.randn()
                Ei += amp_sld * np.random.randn()

            # --- Carrier transport ---
            # Barrier carriers: current injection, capture, escape, recombination
            capture_rate = N_b / tau_cap if tau_cap > 0 else 0.0
            escape_rate = N_w / tau_esc if tau_esc > 1e-30 else 0.0

            dN_b = (I_cur / (q_e * V_b)
                    - capture_rate
                    + escape_rate / chi
                    - N_b / tau_b) * dt

            # QW carriers: capture from barrier, escape, recombination, stimulated
            dN_w = (chi * capture_rate
                    - escape_rate
                    - R_sp_w
                    - Gamma * v_g * g * S) * dt

            # Carrier noise (Langevin)
            FN_w = np.sqrt(2.0 * max(R_sp_w, 0.0) * dt) * np.random.randn()
            N_b += dN_b
            N_w += dN_w + FN_w
            if N_b < 0:
                N_b = 0.0
            if N_w < 0:
                N_w = 0.0

            # Track peak
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


# ── Original solver (no transport) for comparison ───────────────────────

@njit
def simulate_pulses_notransport(
        n_pulses, n_discard, pts_period, dt,
        I_off, I_on, t_on, t_rise,
        V, Gamma, v_g, a_gain, N_tr, eps,
        A, B, C_aug, tau_p, beta_sp, alpha_H, q_e,
        S_inj, seed):
    """Original solver: current injects directly into active region."""
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


# ── Metrics ─────────────────────────────────────────────────────────────

def extract_metrics(phi, pk_S, pk_k, dt, T_rep, laser):
    dphi = np.angle(np.exp(1j * np.diff(phi)))
    r1 = float(np.abs(np.mean(np.exp(1j * dphi))))
    sig_phi = float(np.std(dphi))
    sig_t = float(np.std(pk_k.astype(np.float64) * dt))
    mean_t = float(np.mean(pk_k.astype(np.float64) * dt))
    pk_P = laser.output_power(np.maximum(pk_S, 0))
    mean_P = float(np.mean(pk_P))
    cv = float(np.std(pk_P) / mean_P) if mean_P > 0 else 0.0
    return dict(r1=r1, sig_phi=sig_phi, sig_t=sig_t, mean_t=mean_t,
                cv=cv, mean_P=mean_P)


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 70)
    print("  Carrier Transport Effects on Gain-Switched Dynamics")
    print("=" * 70)

    laser = DFBLaserParams()
    I_th = laser.threshold_current()

    N_PULSES = 500_000
    N_DISCARD = 500
    DT = 1.0e-12
    DUTY = 0.30
    I_BIAS_FACTOR = 0.9
    I_PEAK_FACTOR = 5.0

    I_off = I_BIAS_FACTOR * I_th
    I_on = I_PEAK_FACTOR * I_th

    freqs = np.arange(1, 11) * 1e9
    f_ghz = freqs * 1e-9

    tau_cap_values = [0.5e-12, 1.0e-12, 3.0e-12, 5.0e-12, 10.0e-12]
    tau_cap_labels = ['0.5 ps', '1 ps', '3 ps', '5 ps', '10 ps']

    print(f"  {N_PULSES/1e3:.0f}k pulses, tau_cap = {tau_cap_labels}")
    print(f"  chi (V_b/V_w) = {CHI}, tau_esc = {TAU_ESC*1e12:.0f} ps, "
          f"tau_b = {TAU_B*1e9:.0f} ns")
    print(f"  I_bias = {I_BIAS_FACTOR}×I_th, I_peak = {I_PEAK_FACTOR}×I_th")

    # Warmup both solvers
    print("\n  Compiling JIT solvers...")
    t0 = _time.time()
    _w1 = simulate_pulses_notransport(
        100, 10, 100, 1e-12,
        I_off, I_on, 30e-12, 7e-12,
        laser.V, laser.Gamma, laser.v_g, laser.a, laser.N_tr, laser.epsilon,
        laser.A, laser.B, laser.C, laser.tau_p, laser.beta_sp, laser.alpha_H, q,
        0.0, 0)
    _w2 = simulate_pulses_transport(
        100, 10, 100, 1e-12,
        I_off, I_on, 30e-12, 7e-12,
        laser.V, laser.Gamma, laser.v_g, laser.a, laser.N_tr, laser.epsilon,
        laser.A, laser.B, laser.C, laser.tau_p, laser.beta_sp, laser.alpha_H, q,
        CHI, 3e-12, TAU_ESC, TAU_B,
        0.0, 0)
    del _w1, _w2
    print(f"  Compiled in {_time.time()-t0:.1f}s")

    # ── Part 1: Frequency sweep — no transport vs transport ──────────────

    data = {}
    total_t0 = _time.time()

    # (a) No transport (original model)
    print(f"\n  --- No transport (original model) ---")
    for f_rep in freqs:
        T_rep = 1.0 / f_rep
        t_on = DUTY * T_rep
        pts = max(int(round(T_rep / DT)), 50)
        dt = T_rep / pts
        t_rise = min(20e-12, t_on / 4.0)
        fg = f_rep * 1e-9
        seed = 42 + int(fg) * 100

        print(f"    {fg:.0f} GHz ...", end="", flush=True)
        t0 = _time.time()
        phi, pk_S, _, pk_k = simulate_pulses_notransport(
            N_PULSES + N_DISCARD, N_DISCARD, pts, dt,
            I_off, I_on, t_on, t_rise,
            laser.V, laser.Gamma, laser.v_g, laser.a,
            laser.N_tr, laser.epsilon,
            laser.A, laser.B, laser.C,
            laser.tau_p, laser.beta_sp, laser.alpha_H, q,
            0.0, seed)
        elapsed = _time.time() - t0

        m = extract_metrics(phi, pk_S, pk_k, dt, T_rep, laser)
        data[f'{fg:.0f}GHz_notrans'] = m
        print(f" {elapsed:4.1f}s  r1={m['r1']:.4f}  sig_t={m['sig_t']*1e12:.2f}ps  "
              f"<t>={m['mean_t']*1e12:.1f}ps  P={m['mean_P']*1e3:.2f}mW")

    # (b) With transport at tau_cap = 3 ps (default)
    print(f"\n  --- With transport (tau_cap = {TAU_CAP_DEFAULT*1e12:.0f} ps) ---")
    for f_rep in freqs:
        T_rep = 1.0 / f_rep
        t_on = DUTY * T_rep
        pts = max(int(round(T_rep / DT)), 50)
        dt = T_rep / pts
        t_rise = min(20e-12, t_on / 4.0)
        fg = f_rep * 1e-9
        seed = 42 + int(fg) * 100

        print(f"    {fg:.0f} GHz ...", end="", flush=True)
        t0 = _time.time()
        phi, pk_S, _, pk_k = simulate_pulses_transport(
            N_PULSES + N_DISCARD, N_DISCARD, pts, dt,
            I_off, I_on, t_on, t_rise,
            laser.V, laser.Gamma, laser.v_g, laser.a,
            laser.N_tr, laser.epsilon,
            laser.A, laser.B, laser.C,
            laser.tau_p, laser.beta_sp, laser.alpha_H, q,
            CHI, TAU_CAP_DEFAULT, TAU_ESC, TAU_B,
            0.0, seed)
        elapsed = _time.time() - t0

        m = extract_metrics(phi, pk_S, pk_k, dt, T_rep, laser)
        data[f'{fg:.0f}GHz_trans3ps'] = m
        print(f" {elapsed:4.1f}s  r1={m['r1']:.4f}  sig_t={m['sig_t']*1e12:.2f}ps  "
              f"<t>={m['mean_t']*1e12:.1f}ps  P={m['mean_P']*1e3:.2f}mW")

    # ── Part 2: tau_cap sweep at 10 GHz ──────────────────────────────────

    print(f"\n  --- tau_cap sweep at 10 GHz ---")
    f10 = 10e9
    T10 = 1.0 / f10
    t_on10 = DUTY * T10
    pts10 = max(int(round(T10 / DT)), 50)
    dt10 = T10 / pts10
    t_rise10 = min(20e-12, t_on10 / 4.0)

    tau_sweep = {}
    for i, tc in enumerate(tau_cap_values):
        seed = 42 + 1000 + i
        print(f"    tau_cap = {tc*1e12:.1f} ps ...", end="", flush=True)
        t0 = _time.time()
        phi, pk_S, _, pk_k = simulate_pulses_transport(
            N_PULSES + N_DISCARD, N_DISCARD, pts10, dt10,
            I_off, I_on, t_on10, t_rise10,
            laser.V, laser.Gamma, laser.v_g, laser.a,
            laser.N_tr, laser.epsilon,
            laser.A, laser.B, laser.C,
            laser.tau_p, laser.beta_sp, laser.alpha_H, q,
            CHI, tc, TAU_ESC, TAU_B,
            0.0, seed)
        elapsed = _time.time() - t0

        m = extract_metrics(phi, pk_S, pk_k, dt10, T10, laser)
        tau_sweep[tc] = m
        print(f" {elapsed:4.1f}s  r1={m['r1']:.4f}  sig_t={m['sig_t']*1e12:.2f}ps  "
              f"<t>={m['mean_t']*1e12:.1f}ps  P={m['mean_P']*1e3:.2f}mW")

    total = _time.time() - total_t0
    print(f"\n  Total: {total:.0f}s ({total/60:.1f} min)")

    # ── Figure 1: Frequency comparison ───────────────────────────────────

    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle(
        'Carrier Transport Effects on Gain-Switched Dynamics\n'
        f'$\\chi$ = {CHI}, $\\tau_{{esc}}$ = {TAU_ESC*1e12:.0f} ps, '
        f'$\\tau_{{cap}}$ = {TAU_CAP_DEFAULT*1e12:.0f} ps  |  '
        f'500k pulses, free-running',
        fontsize=12)

    for model, color, marker, label in [
        ('notrans', 'C0', 'o', 'No transport (standard)'),
        ('trans3ps', 'C3', 's', f'With transport ($\\tau_{{cap}}$ = {TAU_CAP_DEFAULT*1e12:.0f} ps)'),
    ]:
        r1_vals = [data[f'{f:.0f}GHz_{model}']['r1'] for f in f_ghz]
        sig_phi_vals = [data[f'{f:.0f}GHz_{model}']['sig_phi'] for f in f_ghz]
        sig_t_vals = [data[f'{f:.0f}GHz_{model}']['sig_t'] * 1e12 for f in f_ghz]
        mean_t_vals = [data[f'{f:.0f}GHz_{model}']['mean_t'] * 1e12 for f in f_ghz]

        axes1[0, 0].plot(f_ghz, r1_vals, f'{marker}-', color=color,
                         lw=2, ms=6, label=label)
        axes1[0, 1].plot(f_ghz, sig_phi_vals, f'{marker}-', color=color,
                         lw=2, ms=6, label=label)
        axes1[1, 0].plot(f_ghz, sig_t_vals, f'{marker}-', color=color,
                         lw=2, ms=6, label=label)
        axes1[1, 1].plot(f_ghz, mean_t_vals, f'{marker}-', color=color,
                         lw=2, ms=6, label=label)

    axes1[0, 0].set_ylabel('Phase correlation $r_1$')
    axes1[0, 0].set_title('Phase memory')
    axes1[0, 1].set_ylabel('Phase jitter $\\sigma_{\\Delta\\phi}$ (rad)')
    axes1[0, 1].set_title('Phase jitter')
    axes1[0, 1].axhline(np.pi / np.sqrt(3), color='gray', ls=':', lw=1, alpha=0.5,
                         label='Uniform random')
    axes1[1, 0].set_ylabel('Timing jitter $\\sigma_t$ (ps)')
    axes1[1, 0].set_title('Peak arrival jitter')
    axes1[1, 1].set_ylabel('Mean peak time $\\langle t_{peak}\\rangle$ (ps)')
    axes1[1, 1].set_title('Turn-on delay (transport adds delay)')

    for ax in axes1.flat:
        ax.set_xlabel('Repetition rate (GHz)')
        ax.set_xticks(f_ghz)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig1.savefig('images/carrier_transport/freq_comparison.png', dpi=150)
    print("  Saved: images/carrier_transport/freq_comparison.png")

    # ── Figure 2: tau_cap sweep at 10 GHz ────────────────────────────────

    fig2, axes2 = plt.subplots(1, 4, figsize=(20, 5))
    fig2.suptitle(
        'Effect of Capture Time $\\tau_{cap}$ at 10 GHz — 500k pulses, free-running',
        fontsize=13)

    tc_ps = [tc * 1e12 for tc in tau_cap_values]
    r1_s = [tau_sweep[tc]['r1'] for tc in tau_cap_values]
    sig_phi_s = [tau_sweep[tc]['sig_phi'] for tc in tau_cap_values]
    sig_t_s = [tau_sweep[tc]['sig_t'] * 1e12 for tc in tau_cap_values]
    mean_t_s = [tau_sweep[tc]['mean_t'] * 1e12 for tc in tau_cap_values]

    # Add no-transport reference
    m_ref = data['10GHz_notrans']

    axes2[0].plot(tc_ps, r1_s, 'ko-', lw=2, ms=7)
    axes2[0].axhline(m_ref['r1'], color='C0', ls='--', lw=1.5,
                     label=f'No transport: {m_ref["r1"]:.4f}')
    axes2[0].set_ylabel('$r_1$')
    axes2[0].set_title('Phase correlation')

    axes2[1].plot(tc_ps, sig_phi_s, 'ko-', lw=2, ms=7)
    axes2[1].axhline(m_ref['sig_phi'], color='C0', ls='--', lw=1.5,
                     label=f'No transport: {m_ref["sig_phi"]:.3f}')
    axes2[1].set_ylabel('$\\sigma_{\\Delta\\phi}$ (rad)')
    axes2[1].set_title('Phase jitter')

    axes2[2].plot(tc_ps, sig_t_s, 'ko-', lw=2, ms=7)
    axes2[2].axhline(m_ref['sig_t'] * 1e12, color='C0', ls='--', lw=1.5,
                     label=f'No transport: {m_ref["sig_t"]*1e12:.2f} ps')
    axes2[2].set_ylabel('$\\sigma_t$ (ps)')
    axes2[2].set_title('Timing jitter')

    axes2[3].plot(tc_ps, mean_t_s, 'ko-', lw=2, ms=7)
    axes2[3].axhline(m_ref['mean_t'] * 1e12, color='C0', ls='--', lw=1.5,
                     label=f'No transport: {m_ref["mean_t"]*1e12:.1f} ps')
    axes2[3].set_ylabel('$\\langle t_{peak} \\rangle$ (ps)')
    axes2[3].set_title('Turn-on delay')

    for ax in axes2:
        ax.set_xlabel('Capture time $\\tau_{cap}$ (ps)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig2.savefig('images/carrier_transport/tau_cap_sweep_10GHz.png', dpi=150)
    print("  Saved: images/carrier_transport/tau_cap_sweep_10GHz.png")

    # ── Figure 3: Relative change — what transport costs ─────────────────

    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5.5))
    fig3.suptitle(
        'Relative Impact of Carrier Transport\n'
        f'Ratio: (with transport) / (no transport),  $\\tau_{{cap}}$ = '
        f'{TAU_CAP_DEFAULT*1e12:.0f} ps',
        fontsize=13)

    r1_nt = np.array([data[f'{f:.0f}GHz_notrans']['r1'] for f in f_ghz])
    r1_tr = np.array([data[f'{f:.0f}GHz_trans3ps']['r1'] for f in f_ghz])
    st_nt = np.array([data[f'{f:.0f}GHz_notrans']['sig_t'] for f in f_ghz])
    st_tr = np.array([data[f'{f:.0f}GHz_trans3ps']['sig_t'] for f in f_ghz])
    mt_nt = np.array([data[f'{f:.0f}GHz_notrans']['mean_t'] for f in f_ghz])
    mt_tr = np.array([data[f'{f:.0f}GHz_trans3ps']['mean_t'] for f in f_ghz])
    P_nt = np.array([data[f'{f:.0f}GHz_notrans']['mean_P'] for f in f_ghz])
    P_tr = np.array([data[f'{f:.0f}GHz_trans3ps']['mean_P'] for f in f_ghz])

    # r₁ change
    ax = axes3[0]
    ax.plot(f_ghz, r1_tr, 's-', color='C3', lw=2, ms=6, label='With transport')
    ax.plot(f_ghz, r1_nt, 'o-', color='C0', lw=2, ms=6, label='No transport')
    ax.set_ylabel('$r_1$')
    ax.set_title('Phase correlation: transport may break\nphase memory at high $f_{rep}$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Timing jitter ratio
    ax = axes3[1]
    ratio_st = st_tr / np.maximum(st_nt, 1e-30)
    ax.plot(f_ghz, ratio_st, 'D-', color='C4', lw=2, ms=6)
    ax.axhline(1.0, color='gray', ls='--', lw=1, alpha=0.5)
    ax.set_ylabel('$\\sigma_t$ ratio (transport / no transport)')
    ax.set_title('Timing jitter: >1 means transport\nincreases jitter')
    ax.grid(True, alpha=0.3)

    # Turn-on delay increase
    ax = axes3[2]
    delay_increase = (mt_tr - mt_nt) * 1e12
    ax.plot(f_ghz, delay_increase, '^-', color='C1', lw=2, ms=6)
    ax.axhline(0, color='gray', ls='--', lw=1, alpha=0.5)
    ax.set_ylabel('Turn-on delay increase (ps)')
    ax.set_title(f'Additional delay from $\\tau_{{cap}}$ = '
                 f'{TAU_CAP_DEFAULT*1e12:.0f} ps')
    ax.grid(True, alpha=0.3)

    for ax in axes3:
        ax.set_xlabel('Repetition rate (GHz)')
        ax.set_xticks(f_ghz)

    plt.tight_layout()
    fig3.savefig('images/carrier_transport/transport_impact.png', dpi=150)
    print("  Saved: images/carrier_transport/transport_impact.png")

    # ── Figure 4: Power and extinction ───────────────────────────────────

    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5.5))
    fig4.suptitle(
        'Output Power Impact of Carrier Transport', fontsize=13)

    ax = axes4[0]
    ax.plot(f_ghz, P_nt * 1e3, 'o-', color='C0', lw=2, ms=6,
            label='No transport')
    ax.plot(f_ghz, P_tr * 1e3, 's-', color='C3', lw=2, ms=6,
            label=f'Transport ($\\tau_{{cap}}$ = {TAU_CAP_DEFAULT*1e12:.0f} ps)')
    ax.set_ylabel('Mean peak power (mW)')
    ax.set_xlabel('Repetition rate (GHz)')
    ax.set_title('Peak power reduction from transport delay')
    ax.set_xticks(f_ghz)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes4[1]
    P_ratio = P_tr / np.maximum(P_nt, 1e-30)
    ax.plot(f_ghz, P_ratio, 'D-', color='C2', lw=2, ms=6)
    ax.axhline(1.0, color='gray', ls='--', lw=1, alpha=0.5)
    ax.set_ylabel('Power ratio (transport / no transport)')
    ax.set_xlabel('Repetition rate (GHz)')
    ax.set_title('Transport penalty increases with frequency')
    ax.set_xticks(f_ghz)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig4.savefig('images/carrier_transport/power_impact.png', dpi=150)
    print("  Saved: images/carrier_transport/power_impact.png")

    plt.close('all')

    # ── Summary ──────────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("  Summary: Carrier Transport Effects")
    print("=" * 70)

    print(f"\n  Frequency sweep (tau_cap = {TAU_CAP_DEFAULT*1e12:.0f} ps):")
    print(f"  {'f_rep':>5s}  {'r1_std':>7s}  {'r1_trn':>7s}  "
          f"{'sig_t_s':>7s}  {'sig_t_t':>7s}  {'delay':>6s}  "
          f"{'P_std':>7s}  {'P_trn':>7s}")
    print(f"  {'(GHz)':>5s}  {'':>7s}  {'':>7s}  "
          f"{'(ps)':>7s}  {'(ps)':>7s}  {'(ps)':>6s}  "
          f"{'(mW)':>7s}  {'(mW)':>7s}")
    print("  " + "-" * 62)

    for fg in f_ghz:
        ms = data[f'{fg:.0f}GHz_notrans']
        mt = data[f'{fg:.0f}GHz_trans3ps']
        dd = (mt['mean_t'] - ms['mean_t']) * 1e12
        print(f"  {fg:5.0f}  {ms['r1']:7.4f}  {mt['r1']:7.4f}  "
              f"{ms['sig_t']*1e12:7.2f}  {mt['sig_t']*1e12:7.2f}  "
              f"{dd:+6.1f}  "
              f"{ms['mean_P']*1e3:7.2f}  {mt['mean_P']*1e3:7.2f}")

    print(f"\n  tau_cap sweep at 10 GHz:")
    print(f"  {'tau_cap':>8s}  {'r1':>7s}  {'sig_phi':>8s}  "
          f"{'sig_t':>7s}  {'<t_pk>':>7s}  {'P_pk':>7s}")
    print(f"  {'(ps)':>8s}  {'':>7s}  {'(rad)':>8s}  "
          f"{'(ps)':>7s}  {'(ps)':>7s}  {'(mW)':>7s}")
    print("  " + "-" * 50)

    for tc in tau_cap_values:
        m = tau_sweep[tc]
        print(f"  {tc*1e12:8.1f}  {m['r1']:7.4f}  {m['sig_phi']:8.3f}  "
              f"{m['sig_t']*1e12:7.2f}  {m['mean_t']*1e12:7.1f}  "
              f"{m['mean_P']*1e3:7.2f}")
    # No-transport ref
    m = data['10GHz_notrans']
    print(f"  {'(none)':>8s}  {m['r1']:7.4f}  {m['sig_phi']:8.3f}  "
          f"{m['sig_t']*1e12:7.2f}  {m['mean_t']*1e12:7.1f}  "
          f"{m['mean_P']*1e3:7.2f}")

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)
