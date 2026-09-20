"""Validation of the source explorer back end (app/pulse_lab.py) and the analysis it adds to
gsdfb/analysis.py.

    python tests/test_pulse_lab.py                  # all checks
    python tests/test_pulse_lab.py arcsine floors   # selected checks

Each check compares against something independent: the Chapter 5 study's own code path, an
explicit sum instead of the FFT, a distribution with a known answer, or Monte Carlo.
"""
import json
import math
import os
import sys
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                                                        # noqa: E402

from app import pulse_lab as lab                                          # noqa: E402
from core.dfb_laser import q                                              # noqa: E402
from core.million_pulse_comparison import (                               # noqa: E402
    build_raised_cosine, phase_autocorrelation, simulate_pulses_waveform)
from gsdfb.analysis import (                                              # noqa: E402
    allan_deviation, amplitude_contrast, amzi_outputs, arcsine_bin_density, arcsine_cdf, bartlett_bound,
    compute_r1, eta_null_cdf, intensity_autocorrelation, ks_distance)
import studies.paper_10ghz_simulation as paper                            # noqa: E402

CH5 = {'laser': 'lo2025', 'shape': 'sine', 'injection': {'mode': 'sld', 'P_sld_mW': 19.0},
       'n_pulses': 3000, 'n_discard': 200, 'seed': 7}


def check_waveforms():
    """The sine is the study's builder byte for byte; every pulse shape spans exactly
    [max(I_DC - I_RF, 0), I_DC + I_RF]; impossible shapes are refused with a reason."""
    spec = lab.normalise_spec(CH5)
    w, pts, dt, T = lab.drive_waveform(spec)
    ref = paper.build_sine_waveform(pts, dt, 1 / T, paper.I_DC, paper.I_RF)
    assert np.array_equal(w, ref)
    assert pts == max(int(round(T / 0.5e-12)), 100)
    I_dc, I_rf = 30.0, 12.0
    for shape, params in [('raised_cosine', {'duty': 0.3, 't_rise_ps': 7.5}), ('square', {'duty': 0.3}),
                          ('gaussian', {'center_frac': 0.3, 'sigma_ps': 7.5}),
                          ('trapezoid', {'duty': 0.3, 't_rise_ps': 5, 't_fall_ps': 9})]:
        s = lab.normalise_spec({'shape': shape, 'I_DC_mA': I_dc, 'I_RF_mA': I_rf, 'shape_params': params})
        w = lab.drive_waveform(s)[0] * 1e3
        assert w.min() >= I_dc - I_rf - 1e-9 and w.max() <= I_dc + I_rf + 1e-9, shape
        assert abs(w.max() - (I_dc + I_rf)) < 1e-9, f'{shape} does not reach the full swing'
    # the Fourier envelope is clipped to [0, 1] from a base of 0.5, so it reaches I_on only if
    # the harmonics sum to 0.5; the docstring and the page hint must say so rather than claim
    # equal swing for every shape
    s = lab.normalise_spec({'shape': 'fourier', 'I_DC_mA': I_dc, 'I_RF_mA': I_rf,
                            'shape_params': {'duty': 0.3, 'harmonics': [[0.25, 0.0]]}})
    w = lab.drive_waveform(s)[0] * 1e3
    assert abs(w.max() - (I_dc - I_rf + 0.75 * 2 * I_rf)) < 1e-6, w.max()
    s = lab.normalise_spec({'shape': 'fourier', 'I_DC_mA': I_dc, 'I_RF_mA': I_rf,
                            'shape_params': {'duty': 0.3, 'harmonics': [[0.5, 0.0]]}})
    assert abs(lab.drive_waveform(s)[0].max() * 1e3 - (I_dc + I_rf)) < 1e-9
    # dt = 1 ps belongs to the 300 um laser, which is what studies/qkd_sinj_sweep.py uses; on the
    # Chapter 5 laser that step diverges and is refused (see check_stability_guard)
    s = lab.normalise_spec({'laser': 'dfb', 'shape': 'raised_cosine',
                            'shape_params': {'duty': 0.3, 't_rise_ps': 7.5},
                            'dt_ps': 1.0, 'I_DC_mA': 20, 'I_RF_mA': 5})
    w, pts, dt, T = lab.drive_waveform(s)
    assert np.array_equal(w, build_raised_cosine(pts, dt, 15e-3, 25e-3, 0.3 * T, 7.5e-12))
    for bad in [{'shape': 'raised_cosine', 'shape_params': {'duty': 0.1, 't_rise_ps': 40}},
                {'shape': 'trapezoid', 'shape_params': {'duty': 0.1, 't_rise_ps': 6, 't_fall_ps': 6}}]:
        try:
            lab.drive_waveform(lab.normalise_spec(bad))
        except ValueError as exc:
            assert 'on-time' in str(exc)
        else:
            raise AssertionError(f'accepted an impossible waveform {bad}')
    a = lab.spec_id(lab.normalise_spec({'shape': 'sine', 'shape_params': {'duty': 0.5}}))
    b = lab.spec_id(lab.normalise_spec({'shape': 'sine', 'shape_params': {'duty': 0.9}}))
    assert a == b, 'a parameter the sine does not use changed the run id'
    print('waveforms OK: sine identical to the study builder, every shape except Fourier spans '
          'exactly I_DC ± I_RF (Fourier reaches it only when its harmonics sum to 0.5), '
          'impossible shapes refused, unused parameters do not change the id')


def check_kernel_fidelity():
    """The explorer's run is the study's run: same kernel arguments as run_config, same arrays."""
    spec = lab.normalise_spec(CH5)
    (phi, S, samp, k), _ = lab.simulate_arrays(spec)
    laser = paper.make_paper_laser()
    f_rep = 10e9
    T = 1 / f_rep
    pts = max(int(round(T / paper.DT_TARGET)), 100)
    dt = T / pts
    wf = paper.build_sine_waveform(pts, dt, f_rep, paper.I_DC, paper.I_RF)
    ref = simulate_pulses_waveform(
        3200, 200, pts, dt, wf, laser.V, laser.Gamma, laser.v_g, laser.a, laser.N_tr, laser.epsilon,
        laser.A, laser.B, laser.C, laser.tau_p, laser.beta_sp, laser.alpha_H, q,
        paper.sld_power_to_sinj(19.0, laser), 7)
    for got, want in zip((phi, S, samp, k), ref):
        assert np.array_equal(got, want)
    print(f'kernel fidelity OK: {len(phi)} pulses byte-identical to the run_config call '
          f'(r1 = {compute_r1(phi)[0]:.4f})')


def check_injection():
    laser = lab.get_laser('lo2025')
    for P, acc, bw in [(19.0, 8.0, 33.0), (19.0, 3.0, 33.0), (5.0, 1.5, 40.0)]:
        spec = lab.normalise_spec({'injection': {'mode': 'sld', 'P_sld_mW': P, 'acceptance_bw_nm': acc,
                                                 'sld_bw_nm': bw}})
        assert lab.injection_S(spec, laser) == paper.sld_power_to_sinj(P, laser, sld_bw_nm=bw,
                                                                       acceptance_bw_nm=acc)
    assert lab.injection_S(lab.normalise_spec({'injection': {'mode': 'none'}}), laser) == 0.0
    assert lab.injection_S(lab.normalise_spec({'injection': {'mode': 'direct', 'log10_S_inj': 20}}),
                           laser) == 1e20
    print('injection OK: SLD conversion is the study function for every bandwidth')


def check_autocorrelations():
    """FFT results equal the explicit sums they replace; the fixed phase_autocorrelation gives
    r1 at lag 1 and full correlation for a constant phase step."""
    rng = np.random.default_rng(3)
    x = np.cumsum(rng.normal(size=4000)) * 0.1 + rng.normal(size=4000)
    lags, rho = intensity_autocorrelation(x, 40)
    c = x - x.mean()
    for kk in (0, 1, 7, 40):
        explicit = np.sum(c[:len(c) - kk] * c[kk:]) / np.sum(c ** 2)
        assert abs(rho[kk] - explicit) < 1e-12, (kk, rho[kk], explicit)
    assert lags[0] == 0 and rho[0] == 1.0 and len(rho) == 41
    phi = np.cumsum(rng.normal(0.3, 0.8, size=5000))
    g = phase_autocorrelation(phi, 30)
    assert abs(g[1] - compute_r1(phi)[0]) < 1e-12
    for kk in (2, 11, 29):
        explicit = abs(np.mean(np.exp(1j * (phi[kk:] - phi[:-kk]))))
        assert abs(g[kk] - explicit) < 1e-12
    step = phase_autocorrelation(np.arange(10000) * np.pi / 2, 5)
    assert np.allclose(step, 1.0), step
    assert abs(bartlett_bound(10 ** 6) - 2.5758 / 1e3) < 1e-6
    print('autocorrelations OK: FFT equals the explicit Chapter 5 sum to 1e-12, lag 1 is r1, '
          'a 90° constant step now reads 1.0 (it read 0.0 before the fix)')


def check_arcsine():
    """Known answers. Uniform phase with pulse-energy jitter: contrast sqrt(<v^2>) and a KS
    distance to the uniform-phase mixture inside the 99% critical value, while the ideal
    arcsine rejects it. A partially coherent train is rejected by both. The mixture CDF is
    checked against a Monte Carlo draw from its own definition."""
    rng = np.random.default_rng(11)
    n = 400_000
    P = rng.lognormal(0.0, 0.35, n + 1)
    dphi = rng.uniform(-np.pi, np.pi, n)
    phi = np.concatenate([[0.0], np.cumsum(dphi)])
    _, _, eta = amzi_outputs(P, phi, 0.4)
    xg, Fg = eta_null_cdf(P)
    null = lambda x: np.interp(x, xg, Fg)
    ks_null, used = ks_distance(eta, null)
    ks_ideal, _ = ks_distance(eta, arcsine_cdf)
    crit = 1.628 / math.sqrt(used)
    assert ks_null < crit, (ks_null, crit)
    assert ks_ideal > 5 * crit, ks_ideal                        # energy jitter alone breaks the ideal
    assert abs(math.sqrt(8 * eta.var()) - amplitude_contrast(P)) < 0.01
    # independent Monte Carlo draw of the null itself
    v = 2 * np.sqrt(P[:-1] * P[1:]) / (P[:-1] + P[1:])
    mc = 0.5 + 0.5 * v * np.cos(rng.uniform(0, 2 * np.pi, n))
    assert ks_distance(mc, null)[0] < crit
    # equal energies: the null IS the ideal arcsine
    x = np.linspace(0, 1, 501)
    xe, Fe = eta_null_cdf(np.ones(1000))
    assert np.max(np.abs(np.interp(x, xe, Fe) - arcsine_cdf(x))) < 2e-3
    # partially coherent phases are rejected however the energies are accounted for
    _, _, coh = amzi_outputs(P, np.concatenate([[0.0], np.cumsum(rng.normal(0.0, 0.6, n))]), 0.0)
    assert ks_distance(coh, null)[0] > 20 * crit
    edges = np.linspace(0, 1, 129)
    assert abs(np.sum(arcsine_bin_density(edges) * np.diff(edges)) - 1) < 1e-12
    # A degenerate v (alternating pulse energies) puts a square-root singularity mid-range, where
    # a plain cosine grid was off by 0.0056 -- more than the KS test calls significant, i.e. enough
    # to invent a phase departure.
    worst = 0.0
    for vtarget in (0.3, 0.6, 0.9):
        ratio = (1 - math.sqrt(1 - vtarget ** 2)) / vtarget
        Pd = np.where(np.arange(20001) % 2 == 0, 1.0, ratio ** 2)
        vd = 2 * np.sqrt(Pd[:-1] * Pd[1:]) / (Pd[:-1] + Pd[1:])
        xd, Fd = eta_null_cdf(Pd)
        grid = np.linspace(0, 1, 20001)
        exact = arcsine_cdf(grid, (1 - vd[0]) / 2, (1 + vd[0]) / 2)
        worst = max(worst, float(np.max(np.abs(np.interp(grid, xd, Fd) - exact))))
    assert worst < 5e-4, worst
    print(f'arcsine OK: uniform phase with energy jitter gives KS {ks_null:.4f} to the uniform-phase '
          f'expectation (99% critical {crit:.4f}) but {ks_ideal:.4f} to the ideal arcsine; Monte Carlo '
          f'agrees with the mixture CDF; a coherent train is rejected; degenerate-v grid error '
          f'{worst:.1e}')


def check_allan():
    """The overlapping Allan deviation has the textbook value for both noise types a pulse
    source can show: independent arrival times (white PM, sqrt(3) sigma_t / tau) and a
    wandering period (white FM, sigma_y / sqrt(m)). The previous estimator failed the second
    by a factor sqrt(m)."""
    rng = np.random.default_rng(2)
    T, dt = 1e-10, 1e-15
    x = rng.normal(0, 5e-12, 200_001)
    tau, adev = allan_deviation(np.round(x / dt).astype(np.int64), dt, T, max_m=1000)
    pm = adev / (math.sqrt(3) * np.std(np.round(x / dt) * dt) / tau)
    assert np.all(np.abs(pm - 1) < 0.08), pm
    sy = 1e-3
    x = np.cumsum(rng.normal(0, sy, 200_001) * T)
    tau, adev = allan_deviation(np.round(x / dt).astype(np.int64), dt, T, max_m=1000)
    fm = adev / (sy / np.sqrt(tau / T))
    assert np.all(np.abs(fm - 1) < 0.15), fm
    print(f'allan OK: white PM within {100 * np.max(np.abs(pm - 1)):.1f}% of sqrt(3) sigma_t/tau, '
          f'white FM within {100 * np.max(np.abs(fm - 1)):.1f}% of sigma_y/sqrt(m), m = 1..1000')


def check_floors():
    """The finite-sample statistics the verdict relies on, against Monte Carlo."""
    rng = np.random.default_rng(5)
    N = 10_000
    r = np.array([compute_r1(rng.uniform(-np.pi, np.pi, N))[0] for _ in range(600)])
    M = N - 1
    bound = paper.autocorr_ci(M, 0.99)
    frac_above_bound = float(np.mean(r > bound))
    assert frac_above_bound < 0.025, frac_above_bound
    p_theory = math.exp(-M * 1e-4)
    assert abs(np.mean(r > 0.01) - p_theory) < 0.06, (np.mean(r > 0.01), p_theory)
    need = lab.resolve_pulses()
    assert paper.autocorr_ci(need - 1, 0.99) < 0.01 <= paper.autocorr_ci(need - 2, 0.99)
    assert lab.resolve_pulses(0.0062) == 318919          # to be called randomised
    assert lab.resolve_pulses(0.0117) == 1593486         # to be called a failure
    assert lab.resolve_pulses(R1 := 0.01) is None        # exactly on the threshold
    assert lab._verdict(0.008, bound) == 'unresolved'
    assert lab._verdict(0.05, bound) == 'above'
    assert lab._verdict(0.012, bound) == 'unresolved'        # past 0.01 but within random scatter
    b6 = paper.autocorr_ci(10 ** 6 - 1, 0.99)
    assert lab._verdict(0.005, b6) == 'below' and lab._verdict(0.02, b6) == 'above'
    assert lab._verdict(0.0095, b6) == 'unresolved'          # the band straddles the threshold

    # The verdict must decide against the THRESHOLD, not against zero correlation. A source whose
    # true r1 is 0.0105 fails the requirement; judging the measurement alone called it randomised
    # in 24% of million-pulse runs.
    M6 = 10 ** 6 - 1
    sigma = 1 / math.sqrt(2 * M6)
    def measured(rho, trials):
        return np.abs(rho + rng.normal(0, sigma, trials) + 1j * rng.normal(0, sigma, trials))
    bad = measured(0.0105, 2000)
    false_pass = float(np.mean([lab._verdict(x, b6) == 'below' for x in bad]))
    assert false_pass < 0.005, false_pass
    good = measured(0.0, 2000)
    resolved = float(np.mean([lab._verdict(x, b6) == 'below' for x in good]))
    assert resolved > 0.99, resolved
    tenk = measured(0.0, 200)
    assert all(lab._verdict(x, bound) == 'unresolved' for x in tenk)
    print(f'floors OK: a random 10k-pulse train exceeds r1 = 0.01 in {100 * np.mean(r > 0.01):.0f}% of '
          f'trials (theory {100 * p_theory:.0f}%), the 99% bound in {100 * frac_above_bound:.1f}%; '
          f'{need:,} pulses resolve r1 = 0; a true r1 of 0.0105 is never called randomised at 1M '
          f'({100 * false_pass:.1f}%, was 24%), while a random source resolves in {100 * resolved:.0f}%')


def check_stability_guard():
    """A time step near the photon lifetime makes the Euler-Maruyama kernel diverge to inf, and
    every metric downstream would then be meaningless. Measured divergence: dt/tau_p = 0.48 for
    the Chapter 5 laser, 0.45 for the 300 um one."""
    for key, dt_ok, dt_bad in [('lo2025', 0.5, 1.0), ('dfb', 1.0, 1.4)]:
        limit = lab.max_dt_ps(key)
        lab.normalise_spec({'laser': key, 'dt_ps': dt_ok})           # must stay allowed
        try:
            lab.normalise_spec({'laser': key, 'dt_ps': dt_bad})
        except ValueError as exc:
            assert 'photon lifetime' in str(exc) and f'{limit:.2f}' in str(exc), str(exc)
        else:
            raise AssertionError(f'{key} accepted dt = {dt_bad} ps, which diverges')
        # the limit really is below where the kernel blows up
        (phi, S, _, _), _ = lab.simulate_arrays(lab.normalise_spec(
            {'laser': key, 'dt_ps': round(limit, 3), 'n_pulses': 150, 'n_discard': 0, 'f_rep_GHz': 1.0}))
        assert np.isfinite(S).all() and np.isfinite(phi).all(), key

    # the output check catches a divergence the dt limit alone would not
    spec = lab.normalise_spec({'laser': 'lo2025', 'dt_ps': 0.72, 'I_DC_mA': 900, 'I_RF_mA': 900,
                               'n_pulses': 200, 'n_discard': 0, 'f_rep_GHz': 1.0})
    try:
        lab.simulate_arrays(spec)
    except ValueError as exc:
        assert 'diverged' in str(exc)
        caught = True
    else:
        caught = False

    # rubbish in the specification is refused with a reason, not an AttributeError
    for bad, word in [({'injection': 'sld'}, 'injection must be an object'),
                      ({'f_rep_GHz': True}, 'must be a number'),
                      ({'shape': 'fourier', 'shape_params': {'harmonics': 5}}, 'list of 1-4')]:
        try:
            lab.normalise_spec(bad)
        except ValueError as exc:
            assert word in str(exc), (bad, str(exc))
        else:
            raise AssertionError(f'accepted {bad}')
    # the saved-run cache is bounded: without a limit, high-statistics runs accumulate forever
    import tempfile as _tempfile
    with _tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for i in range(5):
            blob = directory / f'run{i}.npz'
            blob.write_bytes(b'0' * 400)
            os.utime(blob, (1000 + i, 1000 + i))          # oldest first
        removed = lab.trim_cache(limit=1000, directory=directory)
        left = sorted(f.name for f in directory.glob('*.npz'))
        assert left == ['run3.npz', 'run4.npz'], left     # newest kept, oldest dropped
        assert sorted(removed) == ['run0.npz', 'run1.npz', 'run2.npz'], removed
    print(f'stability guard OK: dt capped at 0.35 tau_p (0.72 ps Chapter 5, 1.08 ps 300 um, so the '
          f'study\'s 1 ps still runs); a diverged run is '
          f'{"refused by the output check" if caught else "not reachable at the capped dt"}; '
          f'malformed specs give reasons; the saved-run cache evicts oldest-first past its limit')


def check_verdict_band():
    """The band around the measurement must hold the true r1 at 99% whatever the phase
    distribution. The Rayleigh quantile alone assumes uniform phases (Var cos = 1/2); with dphi
    concentrated near 0 and pi -- Chapter 5's partially-locked case -- the radial noise is sqrt(2)
    larger and that quantile covers only 96.8%."""
    rng = np.random.default_rng(4)
    M, trials = 50_000, 400
    rayleigh = paper.autocorr_ci(M, lab.CONFIDENCE)
    for label, draw, true in (
            ('uniform', lambda: rng.uniform(-np.pi, np.pi, M), 0.0),
            ('bimodal', lambda: np.where(rng.random(M) < 0.515, 0.0, np.pi), 0.03)):
        inside = old_inside = 0
        for _ in range(trials):
            d = draw()
            r = abs(np.mean(np.exp(1j * d)))
            inside += abs(r - true) <= lab.verdict_band(d)
            old_inside += abs(r - true) <= rayleigh
        assert inside / trials >= 0.98, (label, inside / trials)
        if label == 'bimodal':
            assert old_inside < inside, 'the bimodal case needs the wider, data-driven band'
            bimodal_old, bimodal_new = old_inside / trials, inside / trials
    # the band never narrows below the bound a purely random source respects
    uniform = rng.uniform(-np.pi, np.pi, M)
    assert lab.verdict_band(uniform) >= rayleigh - 1e-12
    assert lab.resolve_pulses(0.0) == 46053 and lab.resolve_pulses(0.0, 1.0) == 66350
    print(f'verdict band OK: bimodal phases covered {bimodal_new:.1%} of the time against '
          f'{bimodal_old:.1%} for the Rayleigh quantile alone; uniform still >= 99%')


def check_acf_null():
    """Both AMZI series are built from consecutive pulse pairs, so the pulse energies alone put
    structure at lag 1. A white-noise Bartlett bound therefore calls a perfectly random source
    correlated on I_A; the null must come from redrawing the phases with these powers."""
    from gsdfb.analysis import bartlett_bound, intensity_autocorrelation, surrogate_acf_envelope
    # the leak is ~CV_P^2/2 whatever N, while the Bartlett bound falls as 1/sqrt(N): the defect
    # only bites once the train is long enough for the bound to drop below it (~100k here).
    # Straight to the kernel: 200k pulses is past the interactive limit simulate() enforces.
    spec = lab.normalise_spec(dict(CH5, n_pulses=200_000, seed=5))
    (_, S, _, _), _ = lab.simulate_arrays(spec)
    P = lab.get_laser(spec['laser']).output_power(np.maximum(S, 0))
    rng = np.random.default_rng(0)
    random_phase = rng.uniform(-np.pi, np.pi, len(P))
    I_A, _, eta = amzi_outputs(P, random_phase, 0.0)
    bart = bartlett_bound(len(I_A), 0.99)
    _, rho_ia = intensity_autocorrelation(I_A, 5)
    _, env_ia = surrogate_acf_envelope(P, 5, series='I_A')
    assert abs(rho_ia[1]) > bart, (rho_ia[1], bart)        # the white-noise bound is wrong here
    assert abs(rho_ia[1]) <= env_ia[0], (rho_ia[1], env_ia[0])
    cv = float(P.std() / P.mean())
    assert 0.2 * cv ** 2 < env_ia[0] < 3 * cv ** 2, (env_ia[0], cv)   # the leak scales with CV_P
    # a genuinely correlated train still breaks the envelope
    locked = np.cumsum(rng.normal(0.0, 0.05, len(P)))
    I_locked, _, _ = amzi_outputs(P, locked, 0.0)
    _, rho_locked = intensity_autocorrelation(I_locked, 5)
    assert abs(rho_locked[1]) > env_ia[0] * 3, (rho_locked[1], env_ia[0])
    # and the envelope is a 99% bound, not the max of a few surrogates: a fresh random-phase draw
    # must stay inside it at essentially every lag (the max-of-8 version flagged 11 of 100)
    _, rho_100 = intensity_autocorrelation(I_A, 100)
    _, env_100 = surrogate_acf_envelope(P, 100, series='I_A', n_surr=12)
    outside = int(np.sum(np.abs(rho_100[1:]) > env_100))
    assert outside <= 3, outside
    print(f'acf null OK: random phases give rho(1) = {rho_ia[1]:+.4f} on I_A (CV_P = {cv:.3f}), '
          f'outside the Bartlett bound ±{bart:.4f} but inside the redrawn-phase envelope '
          f'±{env_ia[0]:.4f}; {outside}/100 lags outside it for a random train; a locked train exceeds it')


def check_analysis_consistency():
    """analyse() reports the library's numbers, not near-copies of them."""
    ds, _ = lab.simulate(CH5)
    a = lab.analyse(ds, {'psi_deg': 37, 'n_bins': 64, 'max_lag': 50})
    json.dumps(a, allow_nan=False)                            # the browser's JSON.parse rejects NaN
    r1, _ = compute_r1(ds.phi)
    assert abs(a['phase']['r1'] - r1) < 1e-5 * max(r1, 1e-3)
    assert abs(a['phase']['g1']['value'][0] - r1) < 1e-4 * max(r1, 1e-3)
    _, _, eta = lab.amzi_outputs(ds.P, ds.phi, math.radians(37))
    _, rho = intensity_autocorrelation(eta, 50)
    assert np.allclose(a['amzi']['acf']['rho'], rho[1:], rtol=1e-4, atol=1e-9)
    assert len(a['phase']['hist']['density']) == 64 and len(a['amzi']['acf']['lags']) == 50
    # the page falls back to the flat bound when these are absent; they must never be absent here
    assert len(a['amzi']['acf']['envelope']) == 50 and a['phase']['r1_band99'] > 0
    assert a['phase']['r1_band99'] >= a['phase']['r1_bound99'] - 1e-12
    dens = np.array(a['phase']['hist']['density'])
    assert abs(np.sum(dens) * 2 * np.pi / 64 - 1) < 1e-4
    assert sum(a['timing']['hist']['counts']) == ds.n
    t0 = time.perf_counter()
    b = lab.analyse(ds, {'psi_deg': 90, 'n_bins': 64, 'max_lag': 50})
    ms = (time.perf_counter() - t0) * 1e3
    assert b['amzi']['psi_deg'] == 90 and b['phase']['r1'] == a['phase']['r1']
    print(f'analysis consistency OK: r1, |g1(1)|, rho(k) match the library; histograms normalise; '
          f'a psi change re-analyses in {ms:.0f} ms without re-simulating')


def check_background_job():
    """A job run in the worker process is byte-identical to the same run in-process, is saved
    atomically, and reloads from disk; cancelling a running job kills it cleanly."""
    spec = dict(CH5, seed=9001, n_pulses=1500)
    did = lab.spec_id(lab.normalise_spec(spec))
    path = lab.CACHE_DIR / f'{did}.npz'
    if path.exists():
        path.unlink()
    job = lab.JOBS.submit(spec)
    deadline = time.time() + 120
    while time.time() < deadline:
        state = next(j for j in lab.JOBS.list() if j['id'] == job['id'])
        if state['state'] in lab.JobManager.TERMINAL:
            break
        time.sleep(0.2)
    assert state['state'] == 'done', state
    assert path.exists()
    saved = lab.load_npz(path)
    (phi, S, samp, k), _ = lab.simulate_arrays(lab.normalise_spec(spec))
    assert np.array_equal(saved.phi, phi) and np.array_equal(saved.k, k) and np.array_equal(saved.S, S)
    long_job = lab.JOBS.submit(dict(CH5, seed=9002, n_pulses=2_000_000, f_rep_GHz=1.0))
    deadline = time.time() + 60
    while next(j for j in lab.JOBS.list() if j['id'] == long_job['id'])['state'] != 'running':
        assert time.time() < deadline, 'the long job never started'
        time.sleep(0.1)
    time.sleep(1.5)
    running = next(j for j in lab.JOBS.list() if j['id'] == long_job['id'])
    assert running['state'] == 'running' and running['progress'] > 0 and running['elapsed_s'] > 1, running
    cancelled = lab.JOBS.cancel(long_job['id'])
    assert cancelled['state'] == 'cancelled'
    assert not (lab.CACHE_DIR / f"{long_job['dataset_id']}.npz").exists()
    path.unlink()
    lab.JOBS.shutdown()
    print(f'background job OK: worker output byte-identical to in-process ({len(phi)} pulses), '
          f'saved and reloaded; a running job reports progress while its kernel holds the worker GIL, '
          'and cancelling it writes nothing')


def check_http():
    """The routes answer, pages and static files are served, and a 409 steers long runs away."""
    from app import server
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{httpd.server_address[1]}'

    def get(path):
        with urllib.request.urlopen(base + path) as r:
            return r.status, r.read()

    def post(path, body):
        req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read(), parse_constant=_reject)
        except urllib.error.HTTPError as err:
            return err.code, json.loads(err.read())

    try:
        for path in ('/', '/pulses', '/static/plot.js', '/static/theme.css'):
            assert get(path)[0] == 200, path
        status, meta = 200, json.loads(get('/api/pulses/meta')[1])
        assert meta['threshold'] == 0.01
        keys = [pr['key'] for pr in meta['presets']]
        assert keys == ['ch5_sld', 'ch5_sld_pp', 'ch5_free', 'sweep_rc'], keys
        pp = next(pr for pr in meta['presets'] if pr['key'] == 'ch5_sld_pp')
        assert pp['spec']['I_RF_mA'] == 37.0   # 3.7 V peak-to-peak into 50 ohm
        sweep = next(pr for pr in meta['presets'] if pr['key'] == 'sweep_rc')
        assert sweep['spec']['n_discard'] == 500   # as in studies/qkd_sinj_sweep.py
        status, est = post('/api/pulses/estimate', {'spec': CH5})
        assert status == 200 and est['steps_per_period'] == 200
        status, sim = post('/api/pulses/simulate', {'spec': CH5, 'knobs': {'psi_deg': 10}})
        assert status == 200 and sim['analysis']['amzi']['psi_deg'] == 10
        status, ana = post('/api/pulses/analyse', {'dataset_id': sim['analysis']['dataset']['id'],
                                                   'knobs': {'n_bins': 32}})
        assert status == 200 and len(ana['phase']['hist']['x']) == 32
        status, slow = post('/api/pulses/simulate', {'spec': dict(CH5, f_rep_GHz=1.0, n_pulses=500_000)})
        assert status == 409 and slow['background'] is True
        status, bad = post('/api/pulses/estimate', {'spec': dict(CH5, dt_ps=0)})
        assert status == 400 and 'dt_ps' in bad['error']
        assert json.loads(get('/api/pulses/datasets')[1])['datasets']
    finally:
        httpd.shutdown()
    print('http OK: pages, static files, meta/estimate/simulate/analyse/datasets routes; strict JSON; '
          '409 for over-long live runs; validation errors name the field')


def _reject(token):
    raise ValueError(f'non-standard JSON constant {token} in response')


CHECKS = {
    'waveforms': check_waveforms,
    'fidelity': check_kernel_fidelity,
    'injection': check_injection,
    'autocorrelations': check_autocorrelations,
    'arcsine': check_arcsine,
    'allan': check_allan,
    'floors': check_floors,
    'stability': check_stability_guard,
    'band': check_verdict_band,
    'acf_null': check_acf_null,
    'analysis': check_analysis_consistency,
    'job': check_background_job,
    'http': check_http,
}

if __name__ == '__main__':
    lab.warm()
    for name in (sys.argv[1:] or CHECKS):
        CHECKS[name]()
    if not sys.argv[1:]:
        print('\nAll source explorer checks passed.')
