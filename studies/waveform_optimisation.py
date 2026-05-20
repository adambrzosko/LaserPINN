"""
Optimal modulation waveform design for gain-switched DFB lasers.

Parameterises the current waveform (Fourier coefficients, or
piecewise-linear) and optimises for a user-selected objective:

  --objective  min_jitter     Minimise timing jitter (sigma_t)
  --objective  max_phase_rand Maximise phase randomisation (minimise r1)
  --objective  min_sld        Minimise S_inj needed to reach r1 < 0.01
  --objective  max_power      Maximise peak output power
  --objective  balanced       Multi-objective: weighted combination

  --waveform   fourier        Fourier-parameterised envelope (default)
  --waveform   trapezoid      Asymmetric trapezoid (rise != fall)

  --freq       5              Repetition rate in GHz (default: 5)
  --n_pulses   50000          Pulses per evaluation (default: 50000)
  --sld        3e19           SLD injection level (default: 0 = free-running)
  --n_harmonics 4             Number of Fourier harmonics (default: 4)

Uses scipy.optimize.differential_evolution (global optimiser) with
the numba JIT solver in the cost function.

Produces:
  1. Optimised waveform shape vs the default raised-cosine
  2. Comparison of pulse statistics (jitter, phase, power)
  3. Convergence history
"""
import argparse
import numpy as np
import time as _time
from scipy.optimize import differential_evolution

from gsdfb import compute_r1, setup_plotting
from gsdfb.plotting import img_dir
from core.dfb_laser import DFBLaserParams, q, h, c
from core.million_pulse_comparison import (
    simulate_pulses, simulate_pulses_waveform,
    build_raised_cosine, build_fourier, build_trapezoid,
)


# ── Cost functions ────────────────────────────────────────────────────

def evaluate_waveform(waveform, pts_period, dt, T_rep,
                      laser, S_inj, n_pulses, n_discard, seed):
    """Run the solver with a given waveform and return metrics."""
    phi, pk_S, _, pk_k = simulate_pulses_waveform(
        n_pulses + n_discard, n_discard, pts_period, dt,
        waveform,
        laser.V, laser.Gamma, laser.v_g, laser.a,
        laser.N_tr, laser.epsilon,
        laser.A, laser.B, laser.C,
        laser.tau_p, laser.beta_sp, laser.alpha_H, q,
        S_inj, seed)

    r1, dphi = compute_r1(phi)
    sig_phi = float(np.std(dphi))
    sig_t = float(np.std(pk_k.astype(np.float64) * dt))
    mean_t = float(np.mean(pk_k.astype(np.float64) * dt))
    pk_P = laser.output_power(np.maximum(pk_S, 0))
    mean_P = float(np.mean(pk_P))
    cv = float(np.std(pk_P) / mean_P) if mean_P > 0 else 1.0

    return dict(r1=r1, sig_phi=sig_phi, sig_t=sig_t, mean_t=mean_t,
                cv=cv, mean_P=mean_P)


def cost_min_jitter(metrics):
    """Minimise timing jitter."""
    return metrics['sig_t']


def cost_max_phase_rand(metrics):
    """Maximise phase randomisation (minimise r1)."""
    return metrics['r1']


def cost_min_sld(metrics):
    """For a given S_inj, maximise the decorrelation.
    This is used with a sweep over S_inj to find the minimum needed."""
    return metrics['r1']


def cost_max_power(metrics):
    """Maximise peak output power."""
    return -metrics['mean_P']


def cost_balanced(metrics):
    """Balanced: low jitter + low r1 + high power.
    Normalise each term so they're comparable."""
    # Target scales (rough order of magnitude)
    jitter_norm = 5e-12     # 5 ps reference
    r1_norm = 0.5           # reference
    power_norm = 10e-3      # 10 mW reference

    return (metrics['sig_t'] / jitter_norm
            + metrics['r1'] / r1_norm
            - metrics['mean_P'] / power_norm)


COST_FUNCTIONS = {
    'min_jitter': cost_min_jitter,
    'max_phase_rand': cost_max_phase_rand,
    'min_sld': cost_min_sld,
    'max_power': cost_max_power,
    'balanced': cost_balanced,
}


# ── Waveform constructors from parameter vectors ─────────────────────

def params_to_fourier_waveform(x, pts_period, dt, I_off, I_on, t_on, n_harmonics):
    """Convert optimiser parameter vector to Fourier waveform.

    x : array of length 2 * n_harmonics
        [a1, b1, a2, b2, ...] Fourier coefficients, each in [-0.5, 0.5]
    """
    coeffs = x.reshape(n_harmonics, 2)
    return build_fourier(pts_period, dt, I_off, I_on, t_on, coeffs)


def params_to_trapezoid_waveform(x, pts_period, dt, I_off, I_on, t_on):
    """Convert optimiser parameter vector to trapezoid waveform.

    x : [t_rise_frac, t_fall_frac, I_on_factor]
        t_rise_frac, t_fall_frac in [0.01, 0.49] (fraction of t_on)
        I_on_factor in [0.5, 2.0] (multiplier on I_on)
    """
    t_rise = x[0] * t_on
    t_fall = x[1] * t_on
    I_on_adj = I_on * x[2]
    return build_trapezoid(pts_period, dt, I_off, I_on_adj, t_on, t_rise, t_fall)


# ── Main optimisation loop ────────────────────────────────────────────

def run_optimisation(objective, waveform_type, f_ghz, n_pulses, S_inj,
                     n_harmonics, seed_base=42):

    laser = DFBLaserParams()
    I_th = laser.threshold_current()

    DT = 1.0e-12
    DUTY = 0.30
    I_BIAS_FACTOR = 0.9
    I_PEAK_FACTOR = 5.0
    I_off = I_BIAS_FACTOR * I_th
    I_on = I_PEAK_FACTOR * I_th

    f_rep = f_ghz * 1e9
    T_rep = 1.0 / f_rep
    t_on = DUTY * T_rep
    pts_period = max(int(round(T_rep / DT)), 50)
    dt = T_rep / pts_period
    t_rise_default = min(20e-12, t_on / 4.0)

    n_discard = min(100, n_pulses // 10)

    cost_fn = COST_FUNCTIONS[objective]

    print(f"\n  Objective: {objective}")
    print(f"  Waveform: {waveform_type}")
    print(f"  f_rep = {f_ghz} GHz, S_inj = {S_inj:.1e}")
    print(f"  {n_pulses//1000}k pulses per evaluation")

    # Baseline: default raised-cosine
    wf_baseline = build_raised_cosine(pts_period, dt, I_off, I_on, t_on, t_rise_default)
    m_baseline = evaluate_waveform(
        wf_baseline, pts_period, dt, T_rep, laser, S_inj,
        n_pulses, n_discard, seed_base)
    cost_baseline = cost_fn(m_baseline)

    print(f"\n  Baseline (raised-cosine):")
    print(f"    r1={m_baseline['r1']:.4f}  sig_t={m_baseline['sig_t']*1e12:.2f} ps  "
          f"P={m_baseline['mean_P']*1e3:.2f} mW  CV={m_baseline['cv']:.4f}")
    print(f"    cost = {cost_baseline:.6e}")

    # Set up bounds
    if waveform_type == 'fourier':
        n_params = 2 * n_harmonics
        bounds = [(-0.5, 0.5)] * n_params

        def make_waveform(x):
            return params_to_fourier_waveform(
                x, pts_period, dt, I_off, I_on, t_on, n_harmonics)
    else:  # trapezoid
        n_params = 3
        bounds = [(0.01, 0.49), (0.01, 0.49), (0.5, 2.0)]

        def make_waveform(x):
            return params_to_trapezoid_waveform(
                x, pts_period, dt, I_off, I_on, t_on)

    # Track convergence
    history = []
    eval_count = [0]

    def objective_fn(x):
        wf = make_waveform(x)
        # Use different seeds for robustness but keep it deterministic
        m = evaluate_waveform(wf, pts_period, dt, T_rep, laser, S_inj,
                              n_pulses, n_discard, seed_base + eval_count[0])
        cost = cost_fn(m)
        history.append(dict(cost=cost, params=x.copy(), metrics=m))
        eval_count[0] += 1
        if eval_count[0] % 20 == 0:
            print(f"    eval {eval_count[0]:4d}: cost={cost:.6e}  "
                  f"r1={m['r1']:.4f}  sig_t={m['sig_t']*1e12:.2f}ps  "
                  f"P={m['mean_P']*1e3:.2f}mW")
        return cost

    print(f"\n  Optimising ({n_params} parameters, "
          f"differential evolution)...")
    t0 = _time.time()

    result = differential_evolution(
        objective_fn, bounds,
        maxiter=30,
        popsize=8,
        tol=1e-3,
        seed=seed_base,
        disp=False,
        polish=True,
    )

    elapsed = _time.time() - t0
    print(f"\n  Optimisation done in {elapsed:.0f}s "
          f"({eval_count[0]} evaluations)")
    print(f"  Best cost: {result.fun:.6e}")

    # Final evaluation with more pulses for reliable statistics
    n_final = max(n_pulses * 4, 200_000)
    n_discard_final = min(500, n_final // 10)
    wf_best = make_waveform(result.x)

    print(f"\n  Final evaluation ({n_final//1000}k pulses)...")
    m_best = evaluate_waveform(
        wf_best, pts_period, dt, T_rep, laser, S_inj,
        n_final, n_discard_final, seed_base)

    # Re-evaluate baseline with same number of pulses
    m_baseline_final = evaluate_waveform(
        wf_baseline, pts_period, dt, T_rep, laser, S_inj,
        n_final, n_discard_final, seed_base)

    print(f"\n  Final results:")
    print(f"  {'':>20s}  {'r1':>8s}  {'sig_t':>8s}  {'P_pk':>8s}  "
          f"{'CV':>8s}  {'<t>':>8s}")
    print(f"  {'':>20s}  {'':>8s}  {'(ps)':>8s}  {'(mW)':>8s}  "
          f"{'':>8s}  {'(ps)':>8s}")
    print("  " + "-" * 60)
    for tag, m in [('Raised-cosine', m_baseline_final),
                   ('Optimised', m_best)]:
        print(f"  {tag:>20s}  {m['r1']:8.4f}  {m['sig_t']*1e12:8.2f}  "
              f"{m['mean_P']*1e3:8.2f}  {m['cv']:8.4f}  "
              f"{m['mean_t']*1e12:8.1f}")

    return dict(
        laser=laser, f_ghz=f_ghz, S_inj=S_inj,
        objective=objective, waveform_type=waveform_type,
        result=result,
        wf_baseline=wf_baseline, wf_best=wf_best,
        m_baseline=m_baseline_final, m_best=m_best,
        history=history, dt=dt, pts_period=pts_period,
        t_on=t_on, I_off=I_off, I_on=I_on,
    )


# ── Plotting ──────────────────────────────────────────────────────────

def plot_results(data):
    """Generate all figures from optimisation results."""
    import matplotlib.pyplot as plt

    dt = data['dt']
    pts = data['pts_period']
    t_ps = np.arange(pts) * dt * 1e12

    # ── Figure 1: Waveform comparison ─────────────────────────────
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
    fig1.suptitle(
        f'Waveform Optimisation: {data["objective"]}\n'
        f'{data["f_ghz"]:.0f} GHz, '
        f'S_{{inj}} = {data["S_inj"]:.1e} m$^{{-3}}$',
        fontsize=13)

    ax = axes1[0]
    ax.plot(t_ps, data['wf_baseline'] * 1e3, 'C0-', lw=2,
            label='Raised-cosine (baseline)')
    ax.plot(t_ps, data['wf_best'] * 1e3, 'C3-', lw=2,
            label='Optimised')
    ax.axvline(data['t_on'] * 1e12, color='gray', ls=':', alpha=0.5,
               label='$t_{on}$')
    ax.set_xlabel('Time within period (ps)')
    ax.set_ylabel('Current (mA)')
    ax.set_title('Modulation waveform')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Metric comparison bar chart
    ax = axes1[1]
    tags = ['$r_1$', '$\\sigma_t$ (ps)', 'Power (mW)', 'CV (%)']
    mb = data['m_baseline']
    mo = data['m_best']
    vals_b = [mb['r1'], mb['sig_t'] * 1e12, mb['mean_P'] * 1e3, mb['cv'] * 100]
    vals_o = [mo['r1'], mo['sig_t'] * 1e12, mo['mean_P'] * 1e3, mo['cv'] * 100]

    x = np.arange(len(tags))
    w = 0.35
    bars1 = ax.bar(x - w / 2, vals_b, w, label='Baseline', color='C0', alpha=0.7)
    bars2 = ax.bar(x + w / 2, vals_o, w, label='Optimised', color='C3', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(tags)
    ax.set_title('Metric comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate improvement
    for i, (vb, vo) in enumerate(zip(vals_b, vals_o)):
        if vb > 0:
            pct = (vo - vb) / vb * 100
            color = 'green' if (pct < 0 and i != 2) or (pct > 0 and i == 2) else 'red'
            ax.annotate(f'{pct:+.1f}%', xy=(i + w / 2, vo),
                        ha='center', va='bottom', fontsize=8, color=color,
                        fontweight='bold')

    plt.tight_layout()
    fname1 = f'images/waveform_opt/waveform_{data["objective"]}_{data["f_ghz"]:.0f}GHz.png'
    fig1.savefig(fname1, dpi=150)
    print(f"  Saved: {fname1}")

    # ── Figure 2: Convergence ─────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    costs = [h['cost'] for h in data['history']]
    best_so_far = np.minimum.accumulate(costs)
    ax2.plot(costs, 'k.', ms=2, alpha=0.3, label='Each evaluation')
    ax2.plot(best_so_far, 'C3-', lw=2, label='Best so far')
    ax2.axhline(vals_b[0] if data['objective'] == 'max_phase_rand'
                else data['history'][0]['cost'],
                color='C0', ls='--', lw=1.5, label='Baseline', alpha=0.7)
    ax2.set_xlabel('Evaluation number')
    ax2.set_ylabel('Cost')
    ax2.set_title(f'Convergence — {data["objective"]}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fname2 = f'images/waveform_opt/convergence_{data["objective"]}_{data["f_ghz"]:.0f}GHz.png'
    fig2.savefig(fname2, dpi=150)
    print(f"  Saved: {fname2}")

    plt.close('all')


# ── Multi-objective comparison ────────────────────────────────────────

def run_all_objectives(f_ghz=5.0, S_inj=0.0, n_pulses=50_000,
                       n_harmonics=4, waveform_type='fourier'):
    """Run all objectives and produce a summary comparison figure."""
    import matplotlib.pyplot as plt

    objectives = ['min_jitter', 'max_phase_rand', 'max_power', 'balanced']
    all_data = {}

    for obj in objectives:
        print(f"\n{'='*70}")
        print(f"  Objective: {obj}")
        print(f"{'='*70}")
        data = run_optimisation(
            obj, waveform_type, f_ghz, n_pulses, S_inj, n_harmonics)
        plot_results(data)
        all_data[obj] = data

    # ── Summary figure ────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f'Waveform Optimisation Summary — {f_ghz:.0f} GHz, '
        f'{"free-running" if S_inj == 0 else f"S_inj = {S_inj:.0e}"}\n'
        f'{waveform_type} parameterisation, {n_harmonics} harmonics',
        fontsize=13)

    # Top-left: all waveforms
    ax = axes[0, 0]
    dt = all_data[objectives[0]]['dt']
    pts = all_data[objectives[0]]['pts_period']
    t_ps = np.arange(pts) * dt * 1e12
    ax.plot(t_ps, all_data[objectives[0]]['wf_baseline'] * 1e3,
            'k-', lw=2.5, label='Baseline (raised-cosine)', alpha=0.8)
    colors_obj = {'min_jitter': 'C0', 'max_phase_rand': 'C1',
                  'max_power': 'C2', 'balanced': 'C3'}
    for obj in objectives:
        ax.plot(t_ps, all_data[obj]['wf_best'] * 1e3, '-', lw=1.8,
                color=colors_obj[obj], label=obj)
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('Current (mA)')
    ax.set_title('Optimised waveforms')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Top-right: radar/bar chart of metrics
    ax = axes[0, 1]
    metric_names = ['$r_1$', '$\\sigma_t$ (ps)', 'Power (mW)', 'CV (%)']
    n_metrics = len(metric_names)
    x = np.arange(n_metrics)
    w = 0.15

    mb = all_data[objectives[0]]['m_baseline']
    vals_base = [mb['r1'], mb['sig_t'] * 1e12, mb['mean_P'] * 1e3, mb['cv'] * 100]
    ax.bar(x - 2 * w, vals_base, w, label='Baseline', color='gray', alpha=0.5)

    for i, obj in enumerate(objectives):
        mo = all_data[obj]['m_best']
        vals = [mo['r1'], mo['sig_t'] * 1e12, mo['mean_P'] * 1e3, mo['cv'] * 100]
        ax.bar(x + (i - 1) * w, vals, w, label=obj, color=colors_obj[obj], alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_title('Metrics comparison')
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')

    # Bottom-left: jitter vs r1 trade-off
    ax = axes[1, 0]
    ax.plot(mb['sig_t'] * 1e12, mb['r1'], 'k*', ms=15, label='Baseline',
            zorder=5)
    for obj in objectives:
        mo = all_data[obj]['m_best']
        ax.plot(mo['sig_t'] * 1e12, mo['r1'], 'o', ms=10,
                color=colors_obj[obj], label=obj, zorder=5)
    ax.set_xlabel('Timing jitter $\\sigma_t$ (ps)')
    ax.set_ylabel('Phase correlation $r_1$')
    ax.set_title('Jitter vs phase correlation trade-off')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-right: power vs jitter
    ax = axes[1, 1]
    ax.plot(mb['mean_P'] * 1e3, mb['sig_t'] * 1e12, 'k*', ms=15,
            label='Baseline', zorder=5)
    for obj in objectives:
        mo = all_data[obj]['m_best']
        ax.plot(mo['mean_P'] * 1e3, mo['sig_t'] * 1e12, 'o', ms=10,
                color=colors_obj[obj], label=obj, zorder=5)
    ax.set_xlabel('Mean peak power (mW)')
    ax.set_ylabel('Timing jitter $\\sigma_t$ (ps)')
    ax.set_title('Power vs jitter trade-off')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f'images/waveform_opt/summary_{f_ghz:.0f}GHz.png'
    fig.savefig(fname, dpi=150)
    print(f"\n  Saved: {fname}")
    plt.close('all')

    return all_data


# ── CLI entry point ───────────────────────────────────────────────────

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    setup_plotting()

    parser = argparse.ArgumentParser(
        description='Optimal modulation waveform for gain-switched DFB laser')
    parser.add_argument('--objective', default='all',
                        choices=['min_jitter', 'max_phase_rand', 'max_power',
                                 'balanced', 'all'],
                        help='Optimisation objective (default: all)')
    parser.add_argument('--waveform', default='fourier',
                        choices=['fourier', 'trapezoid'],
                        help='Waveform parameterisation (default: fourier)')
    parser.add_argument('--freq', type=float, default=5.0,
                        help='Repetition rate in GHz (default: 5)')
    parser.add_argument('--n_pulses', type=int, default=50_000,
                        help='Pulses per cost evaluation (default: 50000)')
    parser.add_argument('--sld', type=float, default=0.0,
                        help='SLD injection S_inj (default: 0 = free-running)')
    parser.add_argument('--n_harmonics', type=int, default=4,
                        help='Fourier harmonics (default: 4)')
    args = parser.parse_args()

    img_dir('waveform_opt')

    print("=" * 70)
    print("  Modulation Waveform Optimisation")
    print("=" * 70)

    # Compile solvers
    from million_pulse_comparison import simulate_pulses_waveform
    laser = DFBLaserParams()
    I_th = laser.threshold_current()
    wf_test = build_raised_cosine(100, 1e-12, 0.9 * I_th, 5.0 * I_th, 30e-12, 7e-12)
    print("  Compiling JIT solvers...")
    t0 = _time.time()
    _w = simulate_pulses_waveform(
        100, 10, 100, 1e-12, wf_test,
        laser.V, laser.Gamma, laser.v_g, laser.a, laser.N_tr, laser.epsilon,
        laser.A, laser.B, laser.C, laser.tau_p, laser.beta_sp, laser.alpha_H, q,
        0.0, 0)
    _w2 = simulate_pulses(
        100, 10, 100, 1e-12, 0.9 * I_th, 5.0 * I_th, 30e-12, 7e-12,
        laser.V, laser.Gamma, laser.v_g, laser.a, laser.N_tr, laser.epsilon,
        laser.A, laser.B, laser.C, laser.tau_p, laser.beta_sp, laser.alpha_H, q,
        0.0, 0)
    del _w, _w2
    print(f"  Compiled in {_time.time()-t0:.1f}s")

    if args.objective == 'all':
        run_all_objectives(
            f_ghz=args.freq, S_inj=args.sld,
            n_pulses=args.n_pulses, n_harmonics=args.n_harmonics,
            waveform_type=args.waveform)
    else:
        data = run_optimisation(
            args.objective, args.waveform, args.freq,
            args.n_pulses, args.sld, args.n_harmonics)
        plot_results(data)

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)
