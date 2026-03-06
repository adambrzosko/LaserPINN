"""
Gain-switched DFB laser with SLD injection: interference & autocorrelation.

Same analysis as gain_switched_interference.py but the DFB cavity receives
CW coherent injection from a superluminescent diode.  This models the
scenario of seeding a gain-switched laser with broadband SLD light to
modify its coherence properties (chirp reduction, spectral narrowing).

Usage
-----
Default parameters:
    python gs_injected_interference.py

Custom parameters (all optional):
    python gs_injected_interference.py \\
        --f_rep 1e9 --duty 0.30 \\
        --I_bias_factor 0.9 --I_peak_factor 5.0 \\
        --I_sld 150e-3 --eta_coupling 0.10 --delta_nu 0.0 \\
        --prefix injected

Requires: numpy, scipy, matplotlib
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dfb_laser import DFBLaserParams, q, h, c
from sld_injection import (
    SLDParams, InjectionParams,
    solve_sld_steady_state, sld_to_injection_field,
    rate_equations_injection, solve_transient_injection,
)
from gain_switched_interference import (
    GainSwitchParams, make_gain_switch_current,
    field_autocorrelation, intensity_autocorrelation, coherence_time,
    mzi_output, mzi_visibility_vs_delay,
    plot_optical_spectrum, plot_field_autocorrelation,
    plot_intensity_autocorrelation, plot_mzi_traces, plot_mzi_visibility,
)


# ── Injected pulse train simulation ─────────────────────────────────────────

def simulate_injected_pulse_train(laser, gs, sld, inj, S_inj,
                                  pts_per_period=2000):
    """
    Simulate a gain-switched DFB pulse train with CW SLD injection.

    Uses the Lang-Kobayashi modified rate equations from sld_injection.py
    driven by the gain-switching current waveform.

    Parameters
    ----------
    laser : DFBLaserParams
    gs : GainSwitchParams
    sld : SLDParams
    inj : InjectionParams  — must have compute_derived() already called
    S_inj : float — injected intracavity photon density (constant / CW)
    pts_per_period : int — time-domain resolution

    Returns
    -------
    dict with keys: t, N, S, phi, P, E, chirp, dt
    """
    I_th = laser.threshold_current()
    I_func = make_gain_switch_current(gs, I_th)

    t_total = gs.t_sim
    n_pts = gs.n_periods * pts_per_period
    t_eval = np.linspace(0, t_total, n_pts)

    print(f"    Solving injected rate equations "
          f"({t_total*1e9:.0f} ns, {n_pts} points)...")

    sol = solve_transient_injection(
        laser, I_func, inj, S_inj,
        phi_inj_func=None,
        t_span=[0, t_total],
        t_eval=t_eval,
    )

    # Discard transient
    idx_start = gs.n_discard * pts_per_period
    t = sol.t[idx_start:] - sol.t[idx_start]
    N = sol.y[0, idx_start:]
    S = sol.y[1, idx_start:]
    phi = sol.y[2, idx_start:]

    P = laser.output_power(np.maximum(S, 0))
    E = np.sqrt(np.maximum(P, 0)) * np.exp(1j * phi)

    dt = t[1] - t[0]
    chirp = np.gradient(phi, dt) / (2 * np.pi)

    return {
        't': t, 'N': N, 'S': S, 'phi': phi,
        'P': P, 'E': E, 'chirp': chirp, 'dt': dt,
    }


# ── Comparison plotting ─────────────────────────────────────────────────────

def plot_pulse_train_comparison(data_free, data_inj, gs, laser, inj, S_inj,
                                n_show=3):
    """Side-by-side comparison of free-running vs injected pulse trains."""
    T_show = n_show * gs.T_rep
    I_th = laser.threshold_current()

    fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex='col')
    fig.suptitle(
        f'Gain-Switched Pulse Train — Free-running vs SLD Injected\n'
        f'(f$_{{rep}}$ = {gs.f_rep*1e-9:.1f} GHz, '
        f'I$_{{peak}}$ = {gs.I_peak_factor:.0f}×I$_{{th}}$, '
        f'κ = {inj.kappa:.2e} s$^{{-1}}$, '
        f'S$_{{inj}}$ = {S_inj:.1e} m$^{{-3}}$)',
        fontsize=12,
    )

    for col, (data, label, color) in enumerate([
        (data_free, 'Free-running', 'C0'),
        (data_inj, 'SLD injected', 'C3'),
    ]):
        t_ns = data['t'] * 1e9
        mask = data['t'] <= T_show

        axes[0, col].plot(t_ns[mask], data['N'][mask] * 1e-24,
                          color=color, lw=1)
        axes[0, col].set_ylabel('Carrier density\n(10$^{24}$ m$^{-3}$)')
        axes[0, col].set_title(label, fontsize=12, fontweight='bold')
        axes[0, col].grid(True, alpha=0.3)

        axes[1, col].plot(t_ns[mask], data['P'][mask] * 1e3,
                          color=color, lw=1)
        axes[1, col].set_ylabel('Output power\n(mW)')
        axes[1, col].grid(True, alpha=0.3)

        axes[2, col].plot(t_ns[mask], data['chirp'][mask] * 1e-9,
                          color=color, lw=1)
        axes[2, col].set_ylabel('Chirp\n(GHz)')
        axes[2, col].set_xlabel('Time (ns)')
        axes[2, col].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_spectrum_comparison(data_free, data_inj, laser):
    """Overlay optical spectra of free-running and injected pulse trains."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Optical Spectrum — Free-running vs SLD Injected', fontsize=13)

    for data, label, color, alpha in [
        (data_free, 'Free-running', 'C0', 0.7),
        (data_inj, 'SLD injected', 'C3', 0.9),
    ]:
        E = data['E']
        dt = data['dt']
        Npts = len(E)
        window = np.hanning(Npts)
        E_fft = np.fft.fftshift(np.fft.fft(E * window))
        freqs = np.fft.fftshift(np.fft.fftfreq(Npts, dt))
        psd = np.abs(E_fft)**2

        mask = np.abs(freqs) < 100e9
        psd_norm = psd[mask] / np.max(psd[mask])
        psd_dB = 10 * np.log10(psd_norm + 1e-30)

        axes[0].plot(freqs[mask] * 1e-9, psd_norm, color=color, lw=1,
                     alpha=alpha, label=label)
        axes[1].plot(freqs[mask] * 1e-9, psd_dB, color=color, lw=1,
                     alpha=alpha, label=label)

    axes[0].set_xlabel('Frequency offset (GHz)')
    axes[0].set_ylabel('Spectral power (normalized)')
    axes[0].set_title('Linear scale')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Frequency offset (GHz)')
    axes[1].set_ylabel('Spectral power (dB)')
    axes[1].set_title('Log scale')
    axes[1].set_ylim(-40, 3)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_g1_comparison(tau_free, g1_free, tau_inj, g1_inj, gs,
                       tau_c_free, tau_c_inj):
    """Overlay |g^(1)(tau)| for free-running and injected."""
    tau_max = 3 * gs.T_rep

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Field Autocorrelation g$^{(1)}$(τ) — Comparison', fontsize=13)

    # Full view
    for tau, g1, label, color in [
        (tau_free, g1_free, f'Free (τ$_c$={tau_c_free*1e12:.1f} ps)', 'C0'),
        (tau_inj, g1_inj, f'Injected (τ$_c$={tau_c_inj*1e12:.1f} ps)', 'C3'),
    ]:
        mask = tau <= tau_max
        axes[0].plot(tau[mask] * 1e9, g1[mask], color=color, lw=1.2,
                     label=label)
    axes[0].axhline(1/np.e, color='gray', ls='--', alpha=0.5)
    for n in range(1, 4):
        axes[0].axvline(n * gs.T_rep * 1e9, color='gray', ls=':', alpha=0.3)
    axes[0].set_xlabel('Delay τ (ns)')
    axes[0].set_ylabel('|g$^{(1)}$(τ)|')
    axes[0].set_title('Full view')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.05, 1.05)

    # Zoomed
    for tau, g1, label, color in [
        (tau_free, g1_free, 'Free-running', 'C0'),
        (tau_inj, g1_inj, 'SLD injected', 'C3'),
    ]:
        mask_z = tau <= gs.T_rep
        axes[1].plot(tau[mask_z] * 1e12, g1[mask_z], color=color, lw=1.5,
                     label=label)
    axes[1].axhline(1/np.e, color='gray', ls='--', alpha=0.5)
    axes[1].set_xlabel('Delay τ (ps)')
    axes[1].set_ylabel('|g$^{(1)}$(τ)|')
    axes[1].set_title('Zoomed — first period')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.05, 1.05)

    plt.tight_layout()
    return fig


def plot_g2_comparison(tau_free, g2_free, tau_inj, g2_inj, gs):
    """Overlay g^(2)(tau) for free-running and injected."""
    tau_max = 3 * gs.T_rep
    fig, ax = plt.subplots(figsize=(10, 5))

    for tau, g2, label, color in [
        (tau_free, g2_free, f'Free [g²(0)={g2_free[0]:.2f}]', 'C0'),
        (tau_inj, g2_inj, f'Injected [g²(0)={g2_inj[0]:.2f}]', 'C3'),
    ]:
        mask = tau <= tau_max
        ax.plot(tau[mask] * 1e9, g2[mask], color=color, lw=1.2, label=label)

    for n in range(1, 4):
        ax.axvline(n * gs.T_rep * 1e9, color='gray', ls=':', alpha=0.3)
    ax.axhline(1.0, color='k', ls='--', alpha=0.3)
    ax.set_xlabel('Delay τ (ns)')
    ax.set_ylabel('g$^{(2)}$(τ)')
    ax.set_title('Intensity Autocorrelation g$^{(2)}$(τ) — Comparison',
                 fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_visibility_comparison(tau_free, vis_free, tau_inj, vis_inj, gs):
    """Overlay MZI visibility curves."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(tau_free * 1e9, vis_free, 'C0o-', ms=2, lw=1.2,
            label='Free-running')
    ax.plot(tau_inj * 1e9, vis_inj, 'C3s-', ms=2, lw=1.2,
            label='SLD injected')

    for n in range(1, 4):
        ax.axvline(n * gs.T_rep * 1e9, color='gray', ls=':', alpha=0.3)

    ax.set_xlabel('Delay τ (ns)')
    ax.set_ylabel('Fringe visibility')
    ax.set_title('MZI Fringe Visibility — Comparison', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    return fig


# ── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Gain-switched DFB + SLD injection: interference analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Gain-switching
    g = p.add_argument_group('Gain switching')
    g.add_argument('--f_rep', type=float, default=1e9,
                   help='Repetition rate (Hz)')
    g.add_argument('--duty', type=float, default=0.30,
                   help='Duty cycle (0–1)')
    g.add_argument('--I_bias_factor', type=float, default=0.9,
                   help='Bias current as fraction of I_th (OFF level)')
    g.add_argument('--I_peak_factor', type=float, default=5.0,
                   help='Peak current as fraction of I_th (ON level)')
    g.add_argument('--t_rise', type=float, default=50e-12,
                   help='Rise/fall time (s)')
    g.add_argument('--n_periods', type=int, default=30,
                   help='Total periods to simulate')
    g.add_argument('--n_discard', type=int, default=5,
                   help='Initial periods to discard')

    # SLD injection
    s = p.add_argument_group('SLD injection')
    s.add_argument('--I_sld', type=float, default=150e-3,
                   help='SLD drive current (A)')
    s.add_argument('--eta_coupling', type=float, default=0.10,
                   help='Injection coupling efficiency (0–1)')
    s.add_argument('--delta_nu', type=float, default=0.0,
                   help='Frequency detuning nu_laser - nu_SLD (Hz)')
    s.add_argument('--sld_lambda', type=float, default=1550e-9,
                   help='SLD center wavelength (m)')

    # Output
    o = p.add_argument_group('Output')
    o.add_argument('--prefix', type=str, default='gs_inj',
                   help='Filename prefix for saved plots')
    o.add_argument('--dpi', type=int, default=150,
                   help='Plot resolution (DPI)')

    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 65)
    print("  Gain-Switched DFB + SLD Injection: Interference Analysis")
    print("=" * 65)

    # ── Build objects from CLI args ───────────────────────────────────────
    laser = DFBLaserParams()
    I_th = laser.threshold_current()

    gs = GainSwitchParams(
        f_rep=args.f_rep,
        duty=args.duty,
        I_bias_factor=args.I_bias_factor,
        I_peak_factor=args.I_peak_factor,
        t_rise=args.t_rise,
        n_periods=args.n_periods,
        n_discard=args.n_discard,
    )

    sld = SLDParams(lambda0=args.sld_lambda)

    inj = InjectionParams(
        eta_coupling=args.eta_coupling,
        delta_nu=args.delta_nu,
    )
    inj.compute_derived(laser)

    prefix = args.prefix

    # ── SLD steady state ──────────────────────────────────────────────────
    print(f"\n  Laser:  {laser.lambda0*1e9:.0f} nm DFB, "
          f"I_th = {I_th*1e3:.1f} mA")
    print(f"  Gain switching:  f_rep = {gs.f_rep*1e-9:.1f} GHz, "
          f"duty = {gs.duty*100:.0f}%, "
          f"I_peak = {gs.I_peak_factor:.0f}×I_th")
    print(f"  SLD:  λ = {sld.lambda0*1e9:.0f} nm, "
          f"I_sld = {args.I_sld*1e3:.0f} mA")

    print(f"\n  0/7  Solving SLD steady state...")
    sld_result = solve_sld_steady_state(sld, args.I_sld)
    P_sld = sld_result['P_out']
    print(f"       P_sld_out = {P_sld*1e3:.2f} mW  "
          f"({'converged' if sld_result['converged'] else 'NOT converged'})")

    S_inj, _ = sld_to_injection_field(sld, P_sld, laser, inj)
    print(f"  Injection:  η = {args.eta_coupling:.0%}, "
          f"Δν = {args.delta_nu*1e-9:.1f} GHz")
    print(f"       κ = {inj.kappa:.2e} s⁻¹, "
          f"S_inj = {S_inj:.2e} m⁻³")

    # ── 1. Free-running pulse train ───────────────────────────────────────
    print(f"\n  1/7  Free-running gain-switched pulse train...")
    from gain_switched_interference import simulate_pulse_train
    data_free = simulate_pulse_train(laser, gs, pts_per_period=2000)
    P_peak_free = np.max(data_free['P'])
    print(f"       P_peak = {P_peak_free*1e3:.1f} mW")

    # ── 2. Injected pulse train ───────────────────────────────────────────
    print(f"\n  2/7  SLD-injected gain-switched pulse train...")
    data_inj = simulate_injected_pulse_train(
        laser, gs, sld, inj, S_inj, pts_per_period=2000,
    )
    P_peak_inj = np.max(data_inj['P'])
    print(f"       P_peak = {P_peak_inj*1e3:.1f} mW")

    # ── 3. Pulse train comparison ─────────────────────────────────────────
    print(f"\n  3/7  Pulse train comparison...")
    fig1 = plot_pulse_train_comparison(data_free, data_inj, gs, laser,
                                       inj, S_inj)
    fname = f'{prefix}_pulse_comparison.png'
    fig1.savefig(fname, dpi=args.dpi)
    print(f"       Saved: {fname}")

    # ── 4. Spectrum comparison ────────────────────────────────────────────
    print(f"\n  4/7  Spectrum comparison...")
    fig2 = plot_spectrum_comparison(data_free, data_inj, laser)
    fname = f'{prefix}_spectrum_comparison.png'
    fig2.savefig(fname, dpi=args.dpi)
    print(f"       Saved: {fname}")

    # ── 5. Field autocorrelation comparison ───────────────────────────────
    print(f"\n  5/7  Field autocorrelation g^(1)(tau)...")
    tau_f, _, g1_f = field_autocorrelation(data_free['E'], data_free['dt'])
    tau_i, _, g1_i = field_autocorrelation(data_inj['E'], data_inj['dt'])
    tc_free = coherence_time(tau_f, g1_f)
    tc_inj = coherence_time(tau_i, g1_i)
    print(f"       Free:     τ_c = {tc_free*1e12:.1f} ps")
    print(f"       Injected: τ_c = {tc_inj*1e12:.1f} ps  "
          f"(×{tc_inj/tc_free:.2f})")

    fig3 = plot_g1_comparison(tau_f, g1_f, tau_i, g1_i, gs, tc_free, tc_inj)
    fname = f'{prefix}_g1_comparison.png'
    fig3.savefig(fname, dpi=args.dpi)
    print(f"       Saved: {fname}")

    # ── 6. Intensity autocorrelation comparison ───────────────────────────
    print(f"\n  6/7  Intensity autocorrelation g^(2)(tau)...")
    tau_g2f, g2_f = intensity_autocorrelation(data_free['P'], data_free['dt'])
    tau_g2i, g2_i = intensity_autocorrelation(data_inj['P'], data_inj['dt'])
    print(f"       Free:     g²(0) = {g2_f[0]:.2f}")
    print(f"       Injected: g²(0) = {g2_i[0]:.2f}")

    fig4 = plot_g2_comparison(tau_g2f, g2_f, tau_g2i, g2_i, gs)
    fname = f'{prefix}_g2_comparison.png'
    fig4.savefig(fname, dpi=args.dpi)
    print(f"       Saved: {fname}")

    # ── 7. MZI visibility comparison ──────────────────────────────────────
    print(f"\n  7/7  MZI visibility comparison...")
    tau_vf, vis_f = mzi_visibility_vs_delay(
        data_free['E'], data_free['dt'],
        tau_max=3 * gs.T_rep, n_delays=150, n_phase_steps=32,
    )
    tau_vi, vis_i = mzi_visibility_vs_delay(
        data_inj['E'], data_inj['dt'],
        tau_max=3 * gs.T_rep, n_delays=150, n_phase_steps=32,
    )
    fig5 = plot_visibility_comparison(tau_vf, vis_f, tau_vi, vis_i, gs)
    fname = f'{prefix}_visibility_comparison.png'
    fig5.savefig(fname, dpi=args.dpi)
    print(f"       Saved: {fname}")

    # ── Summary ───────────────────────────────────────────────────────────
    plt.close('all')

    # Visibility at one rep period
    idx_T_free = np.argmin(np.abs(tau_vf - gs.T_rep))
    idx_T_inj = np.argmin(np.abs(tau_vi - gs.T_rep))

    print("\n" + "-" * 65)
    print("  Summary")
    print("-" * 65)
    print(f"  {'':30s} {'Free':>12s}  {'Injected':>12s}")
    print(f"  {'':30s} {'----':>12s}  {'--------':>12s}")
    print(f"  {'Peak power (mW)':30s} {P_peak_free*1e3:12.1f}  "
          f"{P_peak_inj*1e3:12.1f}")
    print(f"  {'Avg power (mW)':30s} "
          f"{np.mean(data_free['P'])*1e3:12.2f}  "
          f"{np.mean(data_inj['P'])*1e3:12.2f}")
    print(f"  {'Coherence time (ps)':30s} {tc_free*1e12:12.1f}  "
          f"{tc_inj*1e12:12.1f}")
    print(f"  {'g²(0)':30s} {g2_f[0]:12.2f}  {g2_i[0]:12.2f}")
    print(f"  {'V(T_rep)':30s} {vis_f[idx_T_free]:12.4f}  "
          f"{vis_i[idx_T_inj]:12.4f}")

    print(f"\n  Injection impact:")
    print(f"    Coherence time: ×{tc_inj/tc_free:.2f}")
    print(f"    Visibility at T_rep: "
          f"{vis_f[idx_T_free]:.4f} → {vis_i[idx_T_inj]:.4f}")

    print("\n" + "=" * 65)
    print(f"  Done. All figures saved with prefix '{prefix}_'")
    print("=" * 65)


if __name__ == '__main__':
    main()
