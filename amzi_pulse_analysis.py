"""
Asymmetric Mach-Zehnder Interferometer (AMZI) analysis of gain-switched
pulse trains: free-running vs SLD-injected, 1-10 GHz.

The AMZI delay is set to exactly one repetition period T_rep, so
consecutive pulses interfere at the output coupler:

  Port A:  E_A(n) = [ E(n) + e^{i*psi} E(n-1) ] / sqrt(2)
  Port B:  E_B(n) = [ E(n) - e^{i*psi} E(n-1) ] / sqrt(2)

If the pulse-to-pulse phase is correlated (free-running at high f_rep),
almost all power exits one port → narrow intensity distribution.
If the phase is randomised (SLD-injected), power splits randomly →
broad arcsine-like distribution.

Uses the numba JIT solver from million_pulse_comparison.py.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time as _time

from dfb_laser import DFBLaserParams, q, h, c
from sld_injection import (
    SLDParams, InjectionParams,
    solve_sld_steady_state, sld_to_injection_field,
)
from million_pulse_comparison import simulate_pulses


# ── AMZI functions ───────────────────────────────────────────────────────────

def amzi_outputs(peak_P, peak_phi, psi=0.0):
    """Compute AMZI port intensities for consecutive pulse pairs.

    Parameters
    ----------
    peak_P : (N,) array — peak power of each pulse
    peak_phi : (N,) array — phase at peak of each pulse
    psi : float — additional phase offset in delayed arm

    Returns
    -------
    I_A, I_B : (N-1,) arrays — output port intensities (W)
    eta : (N-1,) array — splitting ratio I_A / (I_A + I_B), in [0, 1]
    """
    E = np.sqrt(np.maximum(peak_P, 0)) * np.exp(1j * peak_phi)
    E_n = E[1:]
    E_prev = E[:-1]

    E_A = (E_n + np.exp(1j * psi) * E_prev) / np.sqrt(2)
    E_B = (E_n - np.exp(1j * psi) * E_prev) / np.sqrt(2)

    I_A = np.abs(E_A)**2
    I_B = np.abs(E_B)**2

    total = I_A + I_B
    eta = np.where(total > 0, I_A / total, 0.5)

    return I_A, I_B, eta


def amzi_visibility(peak_P, peak_phi, n_psi=64):
    """Fringe visibility from sweeping AMZI phase offset.

    V = (max<I_A> - min<I_A>) / (max<I_A> + min<I_A>)
    """
    E = np.sqrt(np.maximum(peak_P, 0)) * np.exp(1j * peak_phi)
    E_n = E[1:]
    E_prev = E[:-1]

    psi_vals = np.linspace(0, 2 * np.pi, n_psi, endpoint=False)
    I_A_mean = np.zeros(n_psi)

    for i, psi in enumerate(psi_vals):
        E_A = (E_n + np.exp(1j * psi) * E_prev) / np.sqrt(2)
        I_A_mean[i] = np.mean(np.abs(E_A)**2)

    I_max = np.max(I_A_mean)
    I_min = np.min(I_A_mean)
    V = (I_max - I_min) / (I_max + I_min) if (I_max + I_min) > 0 else 0.0

    psi_opt = psi_vals[np.argmax(I_A_mean)]
    return V, psi_opt, psi_vals, I_A_mean


def optimal_psi(peak_phi):
    """Phase offset that centres the coherent distribution at eta ~ 1."""
    dphi = np.diff(peak_phi)
    return -np.angle(np.mean(np.exp(1j * dphi)))


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 70)
    print("  AMZI Pulse-to-Pulse Coherence Analysis, 1-10 GHz")
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

    N_PULSES = 1_000_000
    N_DISCARD = 1000
    DT = 1.0e-12
    DUTY = 0.30
    I_BIAS_FACTOR = 0.9
    I_PEAK_FACTOR = 5.0

    I_off = I_BIAS_FACTOR * I_th
    I_on = I_PEAK_FACTOR * I_th

    freqs = np.arange(1, 11) * 1e9
    f_ghz = freqs * 1e-9

    print(f"  S_inj = {S_INJ:.2e} m^-3,  {N_PULSES/1e6:.0f}M pulses per run")

    # ── Run simulations & compute AMZI ───────────────────────────────────

    data = {}
    total_t0 = _time.time()

    for f_rep in freqs:
        T_rep = 1.0 / f_rep
        t_on = DUTY * T_rep
        pts = max(int(round(T_rep / DT)), 50)
        dt = T_rep / pts
        t_rise = min(20e-12, t_on / 4.0)

        for case, s_inj in [('free', 0.0), ('sld', S_INJ)]:
            key = f"{f_rep*1e-9:.0f}GHz_{case}"
            seed = 42 + int(f_rep * 1e-9) * 100 + (0 if case == 'free' else 1)

            print(f"  {key:14s} ...", end="", flush=True)
            t0 = _time.time()

            phi, pk_S, sm_S, _ = simulate_pulses(
                N_PULSES + N_DISCARD, N_DISCARD, pts, dt,
                I_off, I_on, t_on, t_rise,
                laser.V, laser.Gamma, laser.v_g, laser.a,
                laser.N_tr, laser.epsilon,
                laser.A, laser.B, laser.C,
                laser.tau_p, laser.beta_sp, laser.alpha_H, q,
                s_inj, seed)

            pk_P = laser.output_power(np.maximum(pk_S, 0))
            elapsed = _time.time() - t0

            # Optimal psi from free-running (or own data)
            psi_opt_val = optimal_psi(phi)

            # AMZI outputs
            I_A, I_B, eta = amzi_outputs(pk_P, phi, psi=psi_opt_val)
            V, _, psi_arr, I_A_vs_psi = amzi_visibility(pk_P, phi)

            data[key] = dict(
                f_rep=f_rep, case=case,
                phi=phi, pk_P=pk_P,
                I_A=I_A, I_B=I_B, eta=eta,
                V=V, psi_opt=psi_opt_val,
                psi_arr=psi_arr, I_A_vs_psi=I_A_vs_psi,
            )

            print(f" {elapsed:5.1f}s  V={V:.4f}  "
                  f"<eta>={np.mean(eta):.3f} +/- {np.std(eta):.3f}")

    # Use free-running psi for SLD case too (fair comparison)
    for f_rep in freqs:
        fg = f"{f_rep*1e-9:.0f}GHz"
        psi_free = data[f"{fg}_free"]['psi_opt']
        d_sld = data[f"{fg}_sld"]
        I_A, I_B, eta = amzi_outputs(d_sld['pk_P'], d_sld['phi'], psi=psi_free)
        d_sld['I_A'] = I_A
        d_sld['I_B'] = I_B
        d_sld['eta'] = eta

    total_elapsed = _time.time() - total_t0
    print(f"\n  Total: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")

    # ── Figure 1: Splitting-ratio density map ────────────────────────────

    fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))
    fig1.suptitle(
        'AMZI Splitting Ratio Distribution vs Repetition Rate\n'
        r'$\eta = I_A / (I_A + I_B)$, delay $= T_{rep}$, '
        r'$\psi$ tuned for constructive (free-running)',
        fontsize=13)

    eta_bins = np.linspace(0, 1, 201)
    eta_centers = 0.5 * (eta_bins[:-1] + eta_bins[1:])

    for ax, case, title in [
        (axes1[0], 'free', 'Free-running'),
        (axes1[1], 'sld', 'SLD-injected'),
    ]:
        density_map = np.zeros((10, len(eta_centers)))
        for i, fg in enumerate(f_ghz):
            key = f"{fg:.0f}GHz_{case}"
            hist, _ = np.histogram(data[key]['eta'], bins=eta_bins, density=True)
            density_map[i, :] = hist

        im = ax.pcolormesh(
            eta_centers, f_ghz, density_map,
            shading='auto', cmap='inferno',
            norm=LogNorm(vmin=0.01, vmax=np.max(density_map) * 1.2))
        ax.set_xlabel(r'Splitting ratio $\eta$')
        ax.set_ylabel('Repetition rate (GHz)')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_yticks(f_ghz)

    plt.colorbar(im, ax=axes1[1], label='Probability density', shrink=0.8)
    plt.tight_layout()
    fig1.savefig('images/amzi/amzi_splitting_ratio_map.png', dpi=150)
    print("  Saved: images/amzi/amzi_splitting_ratio_map.png")

    # ── Figure 2: Splitting-ratio histograms at selected frequencies ─────

    sel = [1, 3, 5, 7, 10]
    n_sel = len(sel)

    fig2, axes2 = plt.subplots(2, n_sel, figsize=(22, 8))
    fig2.suptitle(
        'AMZI Splitting Ratio Histograms — Free-running vs SLD-injected\n'
        r'Narrow peak $\Rightarrow$ coherent;  '
        r'Broad / U-shaped $\Rightarrow$ phase-randomised',
        fontsize=13)

    # Theoretical arcsine for fully random phase
    eta_th = np.linspace(0.001, 0.999, 500)
    arcsine = 1.0 / (np.pi * np.sqrt(eta_th * (1.0 - eta_th)))

    for col, fg in enumerate(sel):
        for row, (case, color, label) in enumerate([
            ('free', 'C0', 'Free-running'),
            ('sld', 'C3', 'SLD-injected'),
        ]):
            key = f"{fg}GHz_{case}"
            ax = axes2[row, col]

            eta_vals = data[key]['eta']
            ax.hist(eta_vals, bins=200, density=True, color=color,
                    alpha=0.7, edgecolor='none')

            # Overlay arcsine reference
            ax.plot(eta_th, arcsine, 'k--', lw=1, alpha=0.4,
                    label='Arcsine (fully random)')

            V = data[key]['V']
            ax.set_title(f'{fg} GHz — {label}\n'
                         f'V = {V:.4f}, '
                         f'$\\sigma_\\eta$ = {np.std(eta_vals):.4f}',
                         fontsize=10)
            ax.set_xlim(-0.02, 1.02)
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel('Probability density')

        axes2[1, col].set_xlabel(r'Splitting ratio $\eta$')
        axes2[0, col].legend(fontsize=7, loc='upper left')

    plt.tight_layout()
    fig2.savefig('images/amzi/amzi_splitting_histograms.png', dpi=150)
    print("  Saved: images/amzi/amzi_splitting_histograms.png")

    # ── Figure 3: Port intensity distributions ───────────────────────────

    fig3, axes3 = plt.subplots(2, n_sel, figsize=(22, 8))
    fig3.suptitle(
        'AMZI Output Port Intensities — 1M Pulses\n'
        'Blue: Port A (constructive)    Orange: Port B (destructive)',
        fontsize=13)

    for col, fg in enumerate(sel):
        for row, (case, label) in enumerate([
            ('free', 'Free-running'), ('sld', 'SLD-injected'),
        ]):
            key = f"{fg}GHz_{case}"
            ax = axes3[row, col]
            I_A = data[key]['I_A'] * 1e3   # mW
            I_B = data[key]['I_B'] * 1e3

            # Common bin range
            all_I = np.concatenate([I_A, I_B])
            lo, hi = np.percentile(all_I, [0.1, 99.9])
            bins = np.linspace(lo, hi, 200)

            ax.hist(I_A, bins=bins, density=True, color='C0', alpha=0.6,
                    edgecolor='none', label='Port A')
            ax.hist(I_B, bins=bins, density=True, color='C1', alpha=0.6,
                    edgecolor='none', label='Port B')

            V = data[key]['V']
            ax.set_title(f'{fg} GHz — {label}  (V={V:.3f})', fontsize=10)
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel('Probability density')
            ax.legend(fontsize=8)

        axes3[1, col].set_xlabel('Output power (mW)')

    plt.tight_layout()
    fig3.savefig('images/amzi/amzi_port_distributions.png', dpi=150)
    print("  Saved: images/amzi/amzi_port_distributions.png")

    # ── Figure 4: Visibility & fringe curve ──────────────────────────────

    fig4, axes4 = plt.subplots(1, 3, figsize=(20, 5))
    fig4.suptitle('AMZI Fringe Visibility — Delay = $T_{rep}$', fontsize=13)

    # Panel 1: Visibility vs frequency
    ax = axes4[0]
    for case, color, marker, label in [
        ('free', 'C0', 'o', 'Free-running'),
        ('sld', 'C3', 's', 'SLD-injected'),
    ]:
        V_vals = [data[f"{fg:.0f}GHz_{case}"]['V'] for fg in f_ghz]
        ax.plot(f_ghz, V_vals, f'{marker}-', color=color, lw=2, ms=7,
                label=label)
    ax.set_xlabel('Repetition rate (GHz)')
    ax.set_ylabel('Fringe visibility V')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(f_ghz)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Visibility vs frequency')

    # Panel 2: Std of splitting ratio vs frequency
    ax = axes4[1]
    for case, color, marker, label in [
        ('free', 'C0', 'o', 'Free-running'),
        ('sld', 'C3', 's', 'SLD-injected'),
    ]:
        sig_vals = [np.std(data[f"{fg:.0f}GHz_{case}"]['eta']) for fg in f_ghz]
        ax.plot(f_ghz, sig_vals, f'{marker}-', color=color, lw=2, ms=7,
                label=label)
    ax.axhline(1.0 / np.sqrt(8), color='gray', ls=':', alpha=0.5,
               label=f'Uniform random: {1/np.sqrt(8):.3f}')
    ax.set_xlabel('Repetition rate (GHz)')
    ax.set_ylabel(r'$\sigma_\eta$ (splitting ratio std)')
    ax.set_xticks(f_ghz)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Splitting ratio spread')

    # Panel 3: Fringe curve I_A(psi) at 10 GHz
    ax = axes4[2]
    for case, color, label in [
        ('free', 'C0', 'Free-running'),
        ('sld', 'C3', 'SLD-injected'),
    ]:
        key = f"10GHz_{case}"
        psi_arr = data[key]['psi_arr']
        I_vs_psi = data[key]['I_A_vs_psi']
        I_norm = I_vs_psi / np.max(I_vs_psi)
        ax.plot(psi_arr / np.pi, I_norm, color=color, lw=2, label=label)
    ax.set_xlabel(r'Phase offset $\psi / \pi$')
    ax.set_ylabel(r'$\langle I_A \rangle$ (normalised)')
    ax.set_title('Fringe curve at 10 GHz')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig4.savefig('images/amzi/amzi_visibility_summary.png', dpi=150)
    print("  Saved: images/amzi/amzi_visibility_summary.png")

    # ── Figure 5: Extinction ratio histogram ─────────────────────────────

    fig5, axes5 = plt.subplots(2, n_sel, figsize=(22, 8))
    fig5.suptitle(
        'AMZI Extinction Ratio  $I_A / I_B$ — 1M Pulses\n'
        'Coherent: peaked at high ER.  Random: broad, centred near 0 dB.',
        fontsize=13)

    for col, fg in enumerate(sel):
        for row, (case, color, label) in enumerate([
            ('free', 'C0', 'Free-running'),
            ('sld', 'C3', 'SLD-injected'),
        ]):
            key = f"{fg}GHz_{case}"
            ax = axes5[row, col]
            I_A = data[key]['I_A']
            I_B = data[key]['I_B']

            # Extinction ratio in dB (clamp to avoid log(0))
            with np.errstate(divide='ignore', invalid='ignore'):
                ER_dB = 10 * np.log10(np.maximum(I_A, 1e-30) /
                                       np.maximum(I_B, 1e-30))
            ER_dB = np.clip(ER_dB, -40, 40)

            ax.hist(ER_dB, bins=300, density=True, color=color, alpha=0.7,
                    edgecolor='none')
            ax.axvline(0, color='k', ls=':', alpha=0.3)
            median_ER = np.median(ER_dB)
            ax.set_title(f'{fg} GHz — {label}\n'
                         f'median ER = {median_ER:.1f} dB',
                         fontsize=10)
            ax.set_xlim(-35, 35)
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel('Probability density')

        axes5[1, col].set_xlabel('Extinction ratio (dB)')

    plt.tight_layout()
    fig5.savefig('images/amzi/amzi_extinction_ratio.png', dpi=150)
    print("  Saved: images/amzi/amzi_extinction_ratio.png")

    # ── Summary ──────────────────────────────────────────────────────────

    plt.close('all')

    print("\n" + "=" * 70)
    print("  AMZI Summary")
    print("=" * 70)
    print(f"  {'f_rep':>5s}  {'case':8s}  {'V':>6s}  {'sig_eta':>7s}  "
          f"{'<eta>':>6s}  {'med_ER':>7s}")
    print("  " + "-" * 52)

    for fg in f_ghz:
        for case in ['free', 'sld']:
            key = f"{fg:.0f}GHz_{case}"
            d = data[key]
            I_A, I_B = d['I_A'], d['I_B']
            with np.errstate(divide='ignore', invalid='ignore'):
                ER = 10 * np.log10(np.maximum(I_A, 1e-30) /
                                    np.maximum(I_B, 1e-30))
            print(f"  {fg:5.0f}  {case:8s}  {d['V']:6.4f}  "
                  f"{np.std(d['eta']):7.4f}  "
                  f"{np.mean(d['eta']):6.3f}  "
                  f"{np.median(ER):7.1f} dB")

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)
