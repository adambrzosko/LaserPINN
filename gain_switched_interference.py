"""
Gain-switched DFB laser pulse train: interference and autocorrelation analysis.

Simulates a 1550 nm DFB laser gain-switched at 1 GHz, then computes:
  1. Field autocorrelation  g^(1)(tau)  — coherence function
  2. Intensity autocorrelation  g^(2)(tau) — photon bunching
  3. Unbalanced Mach-Zehnder interferometer output & fringe visibility

Requires: numpy, scipy, matplotlib
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass

from dfb_laser import DFBLaserParams, solve_transient, q, h, c


# ── Gain-switching parameters ────────────────────────────────────────────────

@dataclass
class GainSwitchParams:
    """Parameters for gain-switched pulse train generation."""

    f_rep: float = 1e9              # repetition rate (Hz)
    duty: float = 0.30              # duty cycle (fraction ON)
    I_bias_factor: float = 0.9      # bias current as fraction of I_th (OFF)
    I_peak_factor: float = 5.0      # peak current as fraction of I_th (ON)
    t_rise: float = 50e-12          # rise / fall time (s)
    n_periods: int = 30             # total periods to simulate
    n_discard: int = 5              # initial periods to discard (transient)

    @property
    def T_rep(self):
        return 1.0 / self.f_rep

    @property
    def t_on(self):
        return self.duty * self.T_rep

    @property
    def t_sim(self):
        return self.n_periods * self.T_rep

    @property
    def n_steady(self):
        return self.n_periods - self.n_discard


# ── Current waveform ─────────────────────────────────────────────────────────

def make_gain_switch_current(gs, I_th):
    """
    Return a current waveform function I(t) for gain-switched operation.

    Uses smooth (raised-cosine) transitions between OFF and ON levels.
    """
    I_off = gs.I_bias_factor * I_th
    I_on = gs.I_peak_factor * I_th
    T = gs.T_rep
    t_on = gs.t_on
    t_r = gs.t_rise

    def I_func(t):
        phase = t % T  # position within current period

        # Rising edge: 0 → t_r
        if phase < t_r:
            # Raised cosine: smooth 0→1 over t_r
            frac = 0.5 * (1 - np.cos(np.pi * phase / t_r))
            return I_off + (I_on - I_off) * frac

        # ON plateau: t_r → t_on - t_r
        elif phase < t_on - t_r:
            return I_on

        # Falling edge: t_on - t_r → t_on
        elif phase < t_on:
            frac = 0.5 * (1 - np.cos(np.pi * (t_on - phase) / t_r))
            return I_off + (I_on - I_off) * frac

        # OFF: t_on → T
        else:
            return I_off

    return I_func


# ── Pulse train simulation ───────────────────────────────────────────────────

def simulate_pulse_train(laser, gs, pts_per_period=2000):
    """
    Simulate gain-switched pulse train using the DFB rate equations.

    Returns
    -------
    dict with keys: t, N, S, phi, P, E, chirp, dt
        All arrays correspond to the steady-state portion (transient discarded).
    """
    I_th = laser.threshold_current()
    I_func = make_gain_switch_current(gs, I_th)

    t_total = gs.t_sim
    n_pts = gs.n_periods * pts_per_period
    t_eval = np.linspace(0, t_total, n_pts)

    print(f"    Solving rate equations ({t_total*1e9:.0f} ns, {n_pts} points)...")
    sol = solve_transient(laser, I_func, [0, t_total], t_eval=t_eval)

    # Discard transient
    idx_start = gs.n_discard * pts_per_period
    t = sol.t[idx_start:] - sol.t[idx_start]  # reset time origin
    N = sol.y[0, idx_start:]
    S = sol.y[1, idx_start:]
    phi = sol.y[2, idx_start:]

    # Output power and E-field envelope
    P = laser.output_power(np.maximum(S, 0))
    E = np.sqrt(np.maximum(P, 0)) * np.exp(1j * phi)

    # Chirp: instantaneous frequency deviation
    dt = t[1] - t[0]
    chirp = np.gradient(phi, dt) / (2 * np.pi)  # Hz

    return {
        't': t, 'N': N, 'S': S, 'phi': phi,
        'P': P, 'E': E, 'chirp': chirp, 'dt': dt,
    }


# ── Correlation functions ────────────────────────────────────────────────────

def field_autocorrelation(E, dt):
    """
    First-order (field) autocorrelation g^(1)(tau) via Wiener-Khinchin.

    g1(tau) = <E*(t) E(t+tau)> / <|E|^2>

    Returns (tau, g1_complex, g1_abs).
    """
    N = len(E)
    # Zero-pad to avoid circular correlation artefacts
    N_pad = 2 * N
    E_pad = np.zeros(N_pad, dtype=complex)
    E_pad[:N] = E

    # Power spectral density → autocorrelation via IFFT
    E_fft = np.fft.fft(E_pad)
    psd = np.abs(E_fft)**2
    acf = np.fft.ifft(psd)

    # Normalize
    norm = np.mean(np.abs(E)**2) * N
    g1 = acf[:N] / norm

    tau = np.arange(N) * dt

    return tau, g1, np.abs(g1)


def intensity_autocorrelation(P, dt):
    """
    Second-order (intensity) autocorrelation g^(2)(tau).

    g2(tau) = <I(t) I(t+tau)> / <I>^2

    Returns (tau, g2).
    """
    N = len(P)
    N_pad = 2 * N
    I_pad = np.zeros(N_pad)
    I_pad[:N] = P

    I_fft = np.fft.fft(I_pad)
    psd = np.abs(I_fft)**2
    acf = np.fft.ifft(psd).real

    I_mean = np.mean(P)
    g2 = acf[:N] / (I_mean**2 * N)

    tau = np.arange(N) * dt

    return tau, g2


def coherence_time(tau, g1_abs):
    """Extract coherence time as the 1/e width of |g^(1)(tau)|."""
    threshold = g1_abs[0] / np.e
    # Find first crossing below threshold
    below = np.where(g1_abs < threshold)[0]
    if len(below) > 0:
        return tau[below[0]]
    return tau[-1]


# ── Unbalanced MZI ───────────────────────────────────────────────────────────

def mzi_output(E, dt, tau_delay, n_phase_steps=64):
    """
    Simulate unbalanced Mach-Zehnder interferometer.

    E_out(t) = (E(t) + e^{i*psi} * E(t - tau_delay)) / sqrt(2)

    Parameters
    ----------
    E : complex E-field envelope array
    dt : time step (s)
    tau_delay : delay in one arm (s)
    n_phase_steps : number of phase offsets psi to sweep (for visibility)

    Returns
    -------
    dict with: I_constructive, I_destructive, visibility,
               I_vs_phase (intensity vs phase offset for one snapshot),
               t_valid (time array for valid overlap region)
    """
    delay_samples = int(round(tau_delay / dt))
    if delay_samples < 0:
        delay_samples = 0
    if delay_samples >= len(E):
        delay_samples = len(E) - 1

    # Delayed copy
    E_direct = E[delay_samples:]
    E_delayed = E[:len(E) - delay_samples]
    N_valid = len(E_direct)
    t_valid = np.arange(N_valid) * dt

    # Sweep phase offset for visibility measurement
    psi_values = np.linspace(0, 2 * np.pi, n_phase_steps, endpoint=False)
    I_mean_vs_psi = np.zeros(n_phase_steps)

    for i, psi in enumerate(psi_values):
        E_out = (E_direct + np.exp(1j * psi) * E_delayed) / np.sqrt(2)
        I_out = np.abs(E_out)**2
        I_mean_vs_psi[i] = np.mean(I_out)

    I_max = np.max(I_mean_vs_psi)
    I_min = np.min(I_mean_vs_psi)
    visibility = (I_max - I_min) / (I_max + I_min) if (I_max + I_min) > 0 else 0.0

    # Constructive (psi = 0) and destructive (psi = pi) time traces
    E_constr = (E_direct + E_delayed) / np.sqrt(2)
    E_destr = (E_direct - E_delayed) / np.sqrt(2)
    I_constr = np.abs(E_constr)**2
    I_destr = np.abs(E_destr)**2

    return {
        't_valid': t_valid,
        'I_constructive': I_constr,
        'I_destructive': I_destr,
        'visibility': visibility,
        'I_vs_phase': I_mean_vs_psi,
        'psi_values': psi_values,
    }


def mzi_visibility_vs_delay(E, dt, tau_max, n_delays=200, n_phase_steps=32):
    """
    Compute MZI fringe visibility as a function of delay.

    Returns (tau_delays, visibilities).
    """
    tau_delays = np.linspace(0, tau_max, n_delays)
    visibilities = np.zeros(n_delays)

    for i, tau in enumerate(tau_delays):
        result = mzi_output(E, dt, tau, n_phase_steps=n_phase_steps)
        visibilities[i] = result['visibility']

    return tau_delays, visibilities


# ── Visualization ────────────────────────────────────────────────────────────

def plot_pulse_train(data, gs, laser, n_show=3):
    """Plot gain-switched pulse train: carrier density, power, chirp."""
    t_ns = data['t'] * 1e9
    T_show = n_show * gs.T_rep
    mask = data['t'] <= T_show

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    I_th = laser.threshold_current()
    fig.suptitle(f'Gain-Switched Pulse Train  '
                 f'(f$_{{rep}}$ = {gs.f_rep*1e-9:.1f} GHz, '
                 f'I$_{{peak}}$ = {gs.I_peak_factor:.0f}×I$_{{th}}$, '
                 f'duty = {gs.duty*100:.0f}%)',
                 fontsize=13)

    axes[0].plot(t_ns[mask], data['N'][mask] * 1e-24, 'b-', lw=1)
    axes[0].set_ylabel('Carrier density\n(10$^{24}$ m$^{-3}$)')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_ns[mask], data['P'][mask] * 1e3, 'r-', lw=1)
    axes[1].set_ylabel('Output power\n(mW)')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_ns[mask], data['chirp'][mask] * 1e-9, 'g-', lw=1)
    axes[2].set_ylabel('Chirp\n(GHz)')
    axes[2].set_xlabel('Time (ns)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_optical_spectrum(data, laser):
    """Plot optical spectrum of the gain-switched pulse train."""
    E = data['E']
    dt = data['dt']
    N = len(E)

    # Windowed FFT for cleaner spectrum
    window = np.hanning(N)
    E_windowed = E * window

    E_fft = np.fft.fftshift(np.fft.fft(E_windowed))
    freqs = np.fft.fftshift(np.fft.fftfreq(N, dt))

    # Power spectral density (a.u., dB)
    psd = np.abs(E_fft)**2
    psd_dB = 10 * np.log10(psd / np.max(psd) + 1e-30)

    # Only show within ±100 GHz of center
    mask = np.abs(freqs) < 100e9

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Optical Spectrum of Gain-Switched Pulse Train', fontsize=13)

    axes[0].plot(freqs[mask] * 1e-9, psd[mask] / np.max(psd[mask]), 'b-', lw=1)
    axes[0].set_xlabel('Frequency offset (GHz)')
    axes[0].set_ylabel('Spectral power (normalized)')
    axes[0].set_title('Linear scale')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(freqs[mask] * 1e-9, psd_dB[mask], 'b-', lw=1)
    axes[1].set_xlabel('Frequency offset (GHz)')
    axes[1].set_ylabel('Spectral power (dB)')
    axes[1].set_title('Log scale')
    axes[1].set_ylim(-40, 3)
    axes[1].grid(True, alpha=0.3)

    # Estimate 3-dB spectral width
    psd_norm = psd[mask] / np.max(psd[mask])
    f_mask = freqs[mask]
    above_half = f_mask[psd_norm > 0.5]
    if len(above_half) >= 2:
        spectral_width = above_half[-1] - above_half[0]
        axes[0].axhline(0.5, color='r', ls='--', alpha=0.5,
                        label=f'3-dB width = {spectral_width*1e-9:.1f} GHz')
        axes[0].legend()

    plt.tight_layout()
    return fig


def plot_field_autocorrelation(tau, g1_abs, gs, tau_c):
    """Plot |g^(1)(tau)| — field autocorrelation / coherence function."""
    # Show up to 3 rep periods
    tau_max = 3 * gs.T_rep
    mask = tau <= tau_max
    tau_ns = tau[mask] * 1e9

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Field Autocorrelation  g$^{(1)}$(τ)', fontsize=13)

    # Full view
    axes[0].plot(tau_ns, g1_abs[mask], 'b-', lw=1.2)
    axes[0].set_xlabel('Delay τ (ns)')
    axes[0].set_ylabel('|g$^{(1)}$(τ)|')
    axes[0].set_title(f'Full view (τ$_c$ = {tau_c*1e12:.1f} ps)')
    axes[0].axhline(1/np.e, color='r', ls='--', alpha=0.5, label='1/e level')
    for n in range(1, 4):
        axes[0].axvline(n * gs.T_rep * 1e9, color='gray', ls=':', alpha=0.4)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.05, 1.05)

    # Zoomed view (first period)
    mask_zoom = tau <= gs.T_rep
    axes[1].plot(tau[mask_zoom] * 1e12, g1_abs[mask_zoom], 'b-', lw=1.5)
    axes[1].set_xlabel('Delay τ (ps)')
    axes[1].set_ylabel('|g$^{(1)}$(τ)|')
    axes[1].set_title('Zoomed — first period')
    axes[1].axhline(1/np.e, color='r', ls='--', alpha=0.5, label='1/e level')
    axes[1].axvline(tau_c * 1e12, color='orange', ls='--', alpha=0.7,
                    label=f'τ$_c$ = {tau_c*1e12:.1f} ps')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.05, 1.05)

    plt.tight_layout()
    return fig


def plot_intensity_autocorrelation(tau, g2, gs):
    """Plot g^(2)(tau) — intensity autocorrelation."""
    tau_max = 3 * gs.T_rep
    mask = tau <= tau_max
    tau_ns = tau[mask] * 1e9

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(tau_ns, g2[mask], 'r-', lw=1.2)
    ax.set_xlabel('Delay τ (ns)')
    ax.set_ylabel('g$^{(2)}$(τ)')
    ax.set_title(f'Intensity Autocorrelation  g$^{{(2)}}$(τ)    '
                 f'[g$^{{(2)}}$(0) = {g2[0]:.2f}]', fontsize=13)
    for n in range(1, 4):
        ax.axvline(n * gs.T_rep * 1e9, color='gray', ls=':', alpha=0.4)
    ax.axhline(1.0, color='k', ls='--', alpha=0.3, label='g$^{(2)}$ = 1 (coherent)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_mzi_traces(data, gs, delays_ns):
    """Plot MZI output time traces for several delays."""
    E = data['E']
    dt = data['dt']

    n_delays = len(delays_ns)
    fig, axes = plt.subplots(n_delays, 1, figsize=(12, 3.5 * n_delays),
                             sharex=True)
    if n_delays == 1:
        axes = [axes]

    fig.suptitle('Unbalanced MZI Output  (constructive & destructive)',
                 fontsize=13)

    for i, tau_ns_val in enumerate(delays_ns):
        tau_delay = tau_ns_val * 1e-9
        result = mzi_output(E, dt, tau_delay)
        t_ns = result['t_valid'] * 1e9

        # Show first 3 periods
        T_show = 3 * gs.T_rep * 1e9
        mask = t_ns <= T_show

        axes[i].plot(t_ns[mask], result['I_constructive'][mask] * 1e3,
                     'b-', lw=1, alpha=0.8, label='Constructive (ψ=0)')
        axes[i].plot(t_ns[mask], result['I_destructive'][mask] * 1e3,
                     'r-', lw=1, alpha=0.8, label='Destructive (ψ=π)')
        axes[i].set_ylabel('Power (mW)')
        axes[i].set_title(f'τ = {tau_ns_val:.2f} ns    '
                          f'(V = {result["visibility"]:.3f})',
                          fontsize=11)
        axes[i].legend(loc='upper right', fontsize=9)
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (ns)')
    plt.tight_layout()
    return fig


def plot_mzi_visibility(tau_delays, visibilities, gs, tau_g1=None, g1_abs_g1=None):
    """Plot MZI fringe visibility vs delay, compare with |g^(1)(tau)|."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(tau_delays * 1e9, visibilities, 'bo-', ms=3, lw=1.5,
            label='MZI visibility')

    if tau_g1 is not None and g1_abs_g1 is not None:
        mask = tau_g1 <= tau_delays[-1]
        ax.plot(tau_g1[mask] * 1e9, g1_abs_g1[mask], 'r-', lw=1, alpha=0.6,
                label='|g$^{(1)}$(τ)|')

    for n in range(1, 4):
        ax.axvline(n * gs.T_rep * 1e9, color='gray', ls=':', alpha=0.4)

    ax.set_xlabel('Delay τ (ns)')
    ax.set_ylabel('Fringe visibility')
    ax.set_title('MZI Fringe Visibility vs Delay', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    return fig


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    print("=" * 65)
    print("  Gain-Switched DFB Laser: Interference & Autocorrelation")
    print("=" * 65)

    # ── Setup ─────────────────────────────────────────────────────────────
    laser = DFBLaserParams()
    gs = GainSwitchParams()
    I_th = laser.threshold_current()

    print(f"\n  Laser:  {laser.lambda0*1e9:.0f} nm DFB, I_th = {I_th*1e3:.1f} mA")
    print(f"  Gain switching:  f_rep = {gs.f_rep*1e-9:.1f} GHz, "
          f"duty = {gs.duty*100:.0f}%")
    print(f"    I_off = {gs.I_bias_factor:.1f}×I_th = {gs.I_bias_factor*I_th*1e3:.1f} mA")
    print(f"    I_on  = {gs.I_peak_factor:.0f}×I_th = {gs.I_peak_factor*I_th*1e3:.1f} mA")
    print(f"    Rise time = {gs.t_rise*1e12:.0f} ps")
    print(f"    Simulating {gs.n_periods} periods ({gs.t_sim*1e9:.0f} ns), "
          f"discarding first {gs.n_discard}")

    # ── 1. Simulate pulse train ───────────────────────────────────────────
    print(f"\n  1/6  Simulating gain-switched pulse train...")
    data = simulate_pulse_train(laser, gs, pts_per_period=2000)

    P_peak = np.max(data['P'])
    P_avg = np.mean(data['P'])
    pulse_energy = P_avg * gs.T_rep
    print(f"       P_peak = {P_peak*1e3:.1f} mW, "
          f"P_avg = {P_avg*1e3:.2f} mW, "
          f"E_pulse = {pulse_energy*1e12:.2f} pJ")

    fig1 = plot_pulse_train(data, gs, laser)
    fig1.savefig('gs_pulse_train.png', dpi=150)
    print("       Saved: gs_pulse_train.png")

    # ── 2. Optical spectrum ───────────────────────────────────────────────
    print("\n  2/6  Optical spectrum...")
    fig2 = plot_optical_spectrum(data, laser)
    fig2.savefig('gs_spectrum.png', dpi=150)
    print("       Saved: gs_spectrum.png")

    # ── 3. Field autocorrelation ──────────────────────────────────────────
    print("\n  3/6  Field autocorrelation g^(1)(tau)...")
    tau_g1, g1_complex, g1_abs = field_autocorrelation(data['E'], data['dt'])
    tau_c = coherence_time(tau_g1, g1_abs)
    print(f"       Coherence time tau_c = {tau_c*1e12:.1f} ps")
    print(f"       |g^(1)(T_rep)| = {g1_abs[int(round(gs.T_rep/data['dt']))]:.4f}")

    fig3 = plot_field_autocorrelation(tau_g1, g1_abs, gs, tau_c)
    fig3.savefig('gs_g1_autocorrelation.png', dpi=150)
    print("       Saved: gs_g1_autocorrelation.png")

    # ── 4. Intensity autocorrelation ──────────────────────────────────────
    print("\n  4/6  Intensity autocorrelation g^(2)(tau)...")
    tau_g2, g2 = intensity_autocorrelation(data['P'], data['dt'])
    print(f"       g^(2)(0) = {g2[0]:.2f}")

    fig4 = plot_intensity_autocorrelation(tau_g2, g2, gs)
    fig4.savefig('gs_g2_autocorrelation.png', dpi=150)
    print("       Saved: gs_g2_autocorrelation.png")

    # ── 5. MZI output traces ─────────────────────────────────────────────
    print("\n  5/6  MZI output traces...")
    delays_ns = [0.10, 0.50, 1.00]
    fig5 = plot_mzi_traces(data, gs, delays_ns)
    fig5.savefig('gs_mzi_traces.png', dpi=150)
    print(f"       Delays: {delays_ns} ns")
    print("       Saved: gs_mzi_traces.png")

    # ── 6. MZI visibility vs delay ────────────────────────────────────────
    print("\n  6/6  MZI fringe visibility vs delay...")
    tau_vis, vis = mzi_visibility_vs_delay(
        data['E'], data['dt'],
        tau_max=3 * gs.T_rep, n_delays=150, n_phase_steps=32,
    )
    fig6 = plot_mzi_visibility(tau_vis, vis, gs, tau_g1, g1_abs)
    fig6.savefig('gs_mzi_visibility.png', dpi=150)
    print("       Saved: gs_mzi_visibility.png")

    # ── Summary ───────────────────────────────────────────────────────────
    plt.close('all')
    print("\n" + "-" * 65)
    print("  Summary")
    print("-" * 65)
    print(f"  Peak power:       {P_peak*1e3:.1f} mW")
    print(f"  Average power:    {P_avg*1e3:.2f} mW")
    print(f"  Pulse energy:     {pulse_energy*1e12:.2f} pJ")
    print(f"  Coherence time:   {tau_c*1e12:.1f} ps")
    print(f"  g^(2)(0):         {g2[0]:.2f}")
    print(f"  V(T_rep):         {vis[np.argmin(np.abs(tau_vis - gs.T_rep))]:.4f}")
    print("\n" + "=" * 65)
    print("  Done. All figures saved to working directory.")
    print("=" * 65)
