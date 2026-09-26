"""Coherence to fourth order of the 10 GHz SLD-injected gain-switched source.

Applies the single-photodiode estimator of Chapter 6, g^(m)(0) = <I^m>/<I>^m, to the
10 GHz jitter acquisitions in ~/Downloads/SLD.

Those records are DC-blocked, but that does NOT obstruct the estimator: g^(m) is
scale-invariant, and Chapter 6 takes the zero from an inter-pulse region of the same
acquisition, which cancels both the coupling term and the scope offset. Writing the trace
as v = G(u - u_bar) + q, the per-pulse difference sum(pulse window) - sum(inter-pulse
window) removes u_bar and q exactly, and is proportional to the pulse energy whenever the
modulation depth is constant. What the DC block costs is the absolute mean optical power,
which g^(m) does not need.

The obstruction at 10 GHz is timing jitter, not coupling. Chapter 6's window is fixed in
phase, so a pulse arriving late puts energy into the inter-pulse window and has it
subtracted rather than counted. At 1 GHz with 80 samples/pulse that is negligible; here
there are 4 samples in the window and the jitter reaches 16.3 ps of a 100 ps period, which
inflates the phase-fixed g^(4) from 3.2 to 41.7 at the top of the sweep.

For a WIDE detection gate -- one that collects the pulse wherever it arrives -- jitter must
not count, so the estimator here is the per-period 10 GHz DFT magnitude |z|, which is
immune to a time shift (a shift rotates the phase of the fundamental and leaves its
magnitude alone; the records are 98% fundamental, 2nd harmonic 1.7%). cross_check() runs
Chapter 6's phase-fixed windows alongside it: the two agree to 0.001 in g^(2) where the
jitter is under ~1 ps, and their ratio is accounted for by <cos^2 phi>/<cos phi>^2 up to
0.6 A. For a NARROW phase-fixed gate the window column is the operationally relevant one.

Since f_rep is locked to the sampling clock (f_s/8 = 10.000000 GHz exactly), an 8-point DFT
of each period separates the observables:

    bin 0  per-period sum -- unusable here: a full-period sum nulls the fundamental, so for
           a near-sinusoidal waveform it carries drift (78% below 10 MHz), not pulse energy
    bin 1  the carrier, |z| proportional to pulse energy and immune to timing jitter
    bin 4  40 GHz, signal-free: the noise record I_e

Validated against 1 GHz records where the pulses are fully resolved, so the |z| route and
ordinary windowed integration can both be formed and compared: |z| reads high by 3.5% of the
excess at m = 2 rising to 12% at m = 9, and the windowed values there reproduce independently
published coherence orders for the same source across m = 2..9. An earlier calibration against
the 8 GHz pickles gave -10%, but those records are post-AMZI, so that comparison measured
interferometer output rather than photon statistics and has been dropped. Note those 8 GHz records, and the
10 GHz pickles, are post-AMZI (excess kurtosis -1.42..-1.45 against the arcsine -1.500),
so their g^(m) measures phase randomisation, not photon statistics. This .txt sweep is not
(|z| excess kurtosis -0.81..+0.22), so it does carry the source's own intensity statistics.

Caveats on the numbers below: |z| tracks pulse energy only if the modulation depth is
constant pulse-to-pulse. V_e is taken from the 40 GHz bin, which is an UPPER bound since
that bin also carries optical noise (it scales with SLD current, 7.0 -> 489 code^2), but
at low order the result barely depends on it: sweeping V_e from zero to that bound moves
g^(2) by 0.00016 free-running, less than half the bootstrap width, and by 0.006 at mid-sweep.
That holds only to about m = 4. By m = 9 the same sweep moves g^(9) by 12.5% at 0.25 A,
eight times the bootstrap width, so if the hierarchy is carried to high order the electronic
floor has to be pinned -- a dark record would do it, not as the background reference (there
is no valid off-pulse region here) but to split the 40 GHz bin into electronic and optical
parts. Free-running, V_e stays irrelevant at every order (0.8x the statistical error). Note there is no off-pulse region here in Chapter 6's
sense -- the laser does not extinguish within the 100 ps period, so the inter-pulse window
holds adjacent-pulse light, which is signal and must not be subtracted.
"""
import os, re, json
import numpy as np
from gsdfb.plotting import setup_plotting, save_fig
import matplotlib.pyplot as plt

DATA = os.path.expanduser('~/Downloads/SLD')
OUT = 'images/sld_coherence_10ghz'
F_REP = 10e9            # Hz, locked to f_s/8
N_S = 8                 # samples per period at 80 GSa/s
SEED = 20260923
rng = np.random.default_rng(SEED)

SWEEP = ['noSLD','0,05A','0,1A','0,15A','0,2A','0,25A','0,3A','0,35A','0,4A',
         '0,45','0,5','0,55','0,6','0,65','0,7']


def load_codes(stem):
    """ADC codes from one jitter record (cached as .npy beside the source)."""
    path = os.path.join(DATA, f'10GHz-SLD_jitter-{stem}.txt' if stem else '10GHz-SLD_jitter.txt')
    cache = os.path.join(OUT, 'cache', os.path.basename(path).replace(',', 'p') + '.npy')
    if os.path.exists(cache):
        return np.load(cache)
    v = np.loadtxt(path, usecols=1).astype(np.float32)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.save(cache, v)
    return v


def period_bins(v, n_s=N_S):
    n = (len(v) // n_s) * n_s
    return np.fft.fft(v[:n].reshape(-1, n_s).astype(np.float64), axis=1)


def g_orders(a, orders=(2, 3, 4)):
    return np.array([(a ** m).mean() / a.mean() ** m for m in orders])


def debias(a, v_quad, iters=2, trials=6):
    """Divide out the bias additive complex noise puts on g^(m) computed from |z|."""
    g = g_orders(a)
    for _ in range(iters):
        r = [g_orders(np.abs(a + rng.normal(0, np.sqrt(v_quad), a.size)
                            + 1j * rng.normal(0, np.sqrt(v_quad), a.size))) / g_orders(a)
             for _ in range(trials)]
        g = g_orders(a) / np.mean(r, axis=0)
    return g, np.mean(r, axis=0)


def block_bootstrap(a, v_quad, nboot=1000, block=2000):
    """Moving-block bootstrap, as Section 6.5 of the thesis specifies: resample in
    neighbouring blocks of pulses so slow baseline drift enters the interval, 10^3 resamples,
    2.5/97.5 percentiles.  A parameter-perturbation Monte Carlo is not equivalent: perturbing
    the two means and two variances while holding the empirical distribution fixed misses the
    sampling error of the higher moments and all serial correlation, and comes out ~10-16x too
    narrow on these records; the same MC without a 1/sqrt(N) comes out ~20-27x too wide."""
    n, idx = len(a), np.arange(block)
    draws = []
    for _ in range(nboot):
        starts = rng.integers(0, n - block, n // block)
        draws.append(debias(a[(starts[:, None] + idx[None, :]).ravel()], v_quad, 1, 2)[0])
    return np.percentile(np.array(draws), [2.5, 97.5], axis=0)


def analyse(stem):
    B = period_bins(load_codes(stem))
    z, noise = B[:, 1], B[:, 4].real
    a = np.abs(z)
    v_quad = noise.var() / 2.0                      # per-quadrature noise variance in bin 1
    g, bias = debias(a, v_quad)
    lo, hi = block_bootstrap(a, v_quad)
    phi = np.angle(z * np.exp(-1j * np.angle(z.mean())))
    var_phi = max(phi.var() - v_quad / a.mean() ** 2, 0.0)   # strip the amplitude-noise term
    return dict(label=stem or 'long', n_pulses=len(a), mean_amp=a.mean(),
                rin=a.std() / a.mean(), g=g, ci_lo=lo, ci_hi=hi,
                jitter_ps=np.sqrt(var_phi) / (2 * np.pi * F_REP) * 1e12,
                noise_bias=bias, v_over_vi=noise.var() / B[:, 0].real.var())


def cross_check(stem):
    """Chapter 6's phase-fixed windows on the same record, for comparison with |z|."""
    v = load_codes(stem)
    n = (len(v) // N_S) * N_S
    prof = v[:n].reshape(-1, N_S).mean(axis=0)
    w = np.roll(v[:n], (1 - int(np.argmax(prof))) % N_S).reshape(-1, N_S).astype(np.float64)
    I = w[:, :N_S // 2].sum(axis=1) - w[:, N_S // 2:].sum(axis=1)
    return g_orders(I)


def validate():
    """Chapter 6 verbatim on the DC-coupled 8 GHz records, against the |z| route."""
    import pickle
    out = []
    for f in ['8GHz-noSLD-data.pkl', '8GHz-SLD-config2-1 (1).pkl']:
        v = pickle.load(open(os.path.join(DATA, f), 'rb'))[:, 1]
        n_s = 10
        n = (len(v) // n_s) * n_s
        prof = v[:n].reshape(-1, n_s).mean(axis=0)
        w = np.roll(v[:n], (2 - int(np.argmax(prof))) % n_s).reshape(-1, n_s)
        i_e = w[:, 5:].sum(axis=1)
        I = w[:, :5].sum(axis=1) - i_e.mean()       # baseline from the pre-pulse region
        def cum(x):
            d = x - x.mean(); m2 = (d**2).mean(); m4 = (d**4).mean()
            return x.mean(), m2, (d**3).mean(), m4 - 3*m2**2
        k1, k2, k3, k4 = cum(I); _, e2, e3, e4 = cum(i_e)
        K2, K3, K4 = k2-e2, k3-e3, k4-e4
        m2, m3, m4 = K2, K3, K4 + 3*K2**2
        truth = np.array([1 + m2/k1**2, 1 + 3*m2/k1**2 + m3/k1**3,
                          1 + 6*m2/k1**2 + 4*m3/k1**3 + m4/k1**4])
        a = np.abs(np.fft.fft(w, axis=1)[:, 1])
        out.append((f, truth, g_orders(a), np.corrcoef(w.sum(axis=1), a)[0, 1]))
    return out


def main():
    setup_plotting()
    rows = [analyse(s) for s in SWEEP] + [analyse('')]
    cur = np.array([0.0] + [float(s.replace('A', '').replace(',', '.')) for s in SWEEP[1:]])
    sweep = rows[:len(SWEEP)]
    g = np.array([r['g'] for r in sweep])
    lo = np.array([r['ci_lo'] for r in sweep])
    hi = np.array([r['ci_hi'] for r in sweep])

    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.8))
    cols = ['#c0622a', '#16917f', '#2a78d6']
    gx = np.array([cross_check(s) for s in SWEEP])
    for k, m in enumerate((2, 3, 4)):
        ax[0].plot(cur, g[:, k], 'o-', ms=3.5, color=cols[k], label=f'$g^{{({m})}}(0)$')
        ax[0].fill_between(cur, lo[:, k], hi[:, k], color=cols[k], alpha=0.25, lw=0)
        ax[0].plot(cur, gx[:, k], ':', lw=1.1, color=cols[k], alpha=.8)
    ax[0].set_xlabel('SLD current (A)'); ax[0].set_ylabel('$g^{(m)}(0)$')
    ax[0].set_yscale('log'); ax[0].legend(frameon=False)
    ax[0].set_title('Wide gate (solid) vs phase-fixed window (dotted)')

    ax[1].plot(cur, [r['jitter_ps'] for r in sweep], 'o-', ms=3.5, color='#b8458c')
    ax[1].set_xlabel('SLD current (A)'); ax[1].set_ylabel('timing jitter (ps)')
    ax[1].set_title('Pulse-position jitter')
    tw = ax[1].twinx()
    tw.plot(cur, [r['mean_amp'] for r in sweep], 's--', ms=3, color='#666', alpha=.7)
    tw.set_ylabel('mean carrier amplitude (codes)', color='#666')
    tw.spines['right'].set_visible(True)

    for stem, c in zip(['noSLD', '0,25A', '0,7'], cols):
        a = np.abs(period_bins(load_codes(stem))[:, 1])
        ax[2].hist(a / a.mean(), bins=160, density=True, histtype='step', color=c,
                   label=f'{stem}')
    ax[2].set_xlabel('pulse amplitude / mean'); ax[2].set_ylabel('density')
    ax[2].set_xlim(0, 3); ax[2].legend(frameon=False, title='SLD current')
    ax[2].set_title('Per-pulse amplitude distribution')
    fig.tight_layout()

    save_fig(fig, os.path.join(OUT, 'coherence_vs_injection.png'), params=dict(
        source=DATA, f_rep_Hz=F_REP, samples_per_period=N_S, seed=SEED,
        estimator='wide gate: g^(m)=<|z|^m>/<|z|>^m from the 10 GHz DFT bin of each period',
        cross_check='Chapter 6 phase-fixed half-period windows, dotted in panel 1',
        gate='wide -- collects the pulse regardless of arrival time, so jitter is excluded',
        noise_reference='40 GHz DFT bin (bin 4), signal-free, same window length',
        dc_reference='DC-blocked, which does not obstruct g^(m): the inter-pulse '
                     'subtraction cancels the offset. See module docstring.',
        validation='8 GHz DC-coupled pickles, Chapter 6 verbatim vs |z| route',
        currents_A=cur.tolist()))

    # compact archive for replotting (studies/plots.ipynb) -- keeps the notebook off the
    # 800 MB of raw traces.  AMP_KEEP per-pulse amplitudes are stored for the histogram panel.
    AMP_KEEP = 125000
    amps = {s: np.abs(period_bins(load_codes(s))[:, 1])[:AMP_KEEP].astype(np.float32)
            for s in ('noSLD', '0,25A', '0,7')}
    np.savez_compressed(
        os.path.join(OUT, 'sld_coherence_10ghz.npz'),
        currents=cur, labels=np.array(SWEEP),
        g=g, ci_lo=lo, ci_hi=hi, g_phase_fixed=gx,
        jitter_ps=np.array([r['jitter_ps'] for r in sweep]),
        mean_amp=np.array([r['mean_amp'] for r in sweep]),
        rin=np.array([r['rin'] for r in sweep]),
        n_pulses=np.array([r['n_pulses'] for r in sweep]),
        long_g=rows[-1]['g'], long_ci_lo=rows[-1]['ci_lo'], long_ci_hi=rows[-1]['ci_hi'],
        long_n=rows[-1]['n_pulses'], long_jitter_ps=rows[-1]['jitter_ps'],
        amp_noSLD=amps['noSLD'], amp_025=amps['0,25A'], amp_07=amps['0,7'],
        orders=np.array([2, 3, 4]))

    res = [dict(label=r['label'], current_A=(cur[i] if i < len(cur) else None),
                n_pulses=r['n_pulses'], mean_amp=r['mean_amp'], rin=r['rin'],
                g2=r['g'][0], g3=r['g'][1], g4=r['g'][2],
                g2_ci=[r['ci_lo'][0], r['ci_hi'][0]],
                g3_ci=[r['ci_lo'][1], r['ci_hi'][1]],
                g4_ci=[r['ci_lo'][2], r['ci_hi'][2]],
                jitter_ps=r['jitter_ps']) for i, r in enumerate(rows)]
    with open(os.path.join(OUT, 'coherence.json'), 'w') as fh:
        json.dump(res, fh, indent=1)

    print(f"{'I_SLD':<7s} {'pulses':>8s} {'RIN%':>6s} {'g2':>9s} {'g3':>9s} {'g4':>9s} {'jitter/ps':>9s}")
    for r, c in zip(rows, list(cur) + [None]):
        lbl = r['label']
        print(f"{lbl:<7s} {r['n_pulses']:8d} {100*r['rin']:6.2f} {r['g'][0]:9.5f} "
              f"{r['g'][1]:9.5f} {r['g'][2]:9.5f} {r['jitter_ps']:9.2f}")
    print('\n8 GHz DC-coupled validation (Chapter 6 verbatim vs the |z| route used above):')
    for f, truth, gz, corr in validate():
        print(f'  {f:<26s} corr(sum,|z|)={corr:+.4f}')
        print(f'     Ch6 verbatim : g2={truth[0]:.5f} g3={truth[1]:.5f} g4={truth[2]:.5f}')
        print(f'     |z| route    : g2={gz[0]:.5f} g3={gz[1]:.5f} g4={gz[2]:.5f}'
              f'   ({100*(gz[0]-truth[0])/(truth[0]-1):+.1f}% of the excess)')


if __name__ == '__main__':
    main()
