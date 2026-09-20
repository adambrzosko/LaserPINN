"""
Source explorer back end: simulate a gain-switched DFB pulse train once, then analyse it as
often as you like.

The asymmetry that shapes this module, measured on this machine: the pulse kernel
(core.million_pulse_comparison.simulate_pulses_waveform) costs ~0.105 us per time step --
22 us/pulse at 10 GHz and 205 us/pulse at 1 GHz with dt = 0.5 ps, because the cost scales with
steps per period, 1/(f_rep dt) -- while analysing 10k pulses costs ~20-30 ms (1.5 s for 1M
pulses on first open, dominated by the Allan deviation). So a run is simulated once and cached
under an id derived from its specification, and every analysis knob -- AMZI phase, histogram
bins, lag range, detector gate, Allan range -- is re-evaluated against the cached arrays
without touching the kernel. The expensive, knob-independent pieces (Allan deviation, AMZI
fringe, phase correlation, the uniform-phase splitting-ratio null) are memoised per dataset.

No physics is re-implemented here. The Chapter 5 laser, sinusoidal drive, SLD-to-S_inj
conversion and 99% Rayleigh bound come from studies/paper_10ghz_simulation.py; the other
waveforms and the kernel from core/million_pulse_comparison.py; every metric from
gsdfb/analysis.py. This module chooses arguments, caches, and packages JSON.

Two measured constraints set the execution model:
  * The numba kernel holds the GIL for its whole run: a ticker thread was starved for 1553 ms
    by a 1.55 s run. A long run in a server thread would freeze every other request, so
    high-statistics runs go to a separate worker process.
  * The kernel cannot report progress, and chunking it would restart the laser state (N, E)
    at every boundary -- corrupting exactly the inter-pulse phase statistics being measured.
    Background progress is therefore an estimate from the calibrated cost per step.
"""
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import tempfile
import threading
import time
from collections import OrderedDict, deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.dfb_laser import make_laser, q                                  # noqa: E402
from core.million_pulse_comparison import (                               # noqa: E402
    build_fourier, build_gaussian, build_raised_cosine, build_square, build_trapezoid,
    phase_autocorrelation, simulate_pulses_waveform)
from gsdfb.analysis import (                                              # noqa: E402
    absolute_jitter, allan_deviation, amplitude_contrast, amzi_outputs, amzi_splitting_ratio,
    arcsine_bin_density, arcsine_cdf, bartlett_bound, compute_r1, eta_null_cdf,
    intensity_autocorrelation, ks_distance, period_jitter, phase_randomisation_quality,
    surrogate_acf_envelope)
from studies.paper_10ghz_simulation import (                              # noqa: E402
    I_DC as PAPER_I_DC, I_RF as PAPER_I_RF, Z_MATCH, autocorr_ci, build_sine_waveform,
    make_paper_laser, sld_power_to_sinj)

R1_THRESHOLD = 0.01        # decoy-state BB84 requirement, as in studies/qkd_sinj_sweep.py
CONFIDENCE = 0.99          # Chapter 5's confidence level, used for every bound below
LIVE_MAX_S = 8.0           # predicted kernel time above which a run must go to the background
MAX_PULSES = 5_000_000
MEMORY_PULSE_BUDGET = 3_000_000   # ~50 MB of arrays per million pulses
LAG_CAP = 1000
KS_SLICE = 10_000          # phase_randomisation_quality runs its KS test on the first 10k only
# The kernel integrates the field with Euler-Maruyama, so a step comparable with the photon
# lifetime is unstable. Measured divergence to inf at dt/tau_p = 0.48 (150 um DFB), 0.45 (300 um)
# and 0.52 (FP). 0.35 keeps a margin while still allowing the 1 ps that
# studies/qkd_sinj_sweep.py uses on the 300 um laser (1.08 ps there, 0.72 ps for Chapter 5's).
STABILITY_FRACTION = 0.35
MODEL_VERSION = 1          # bump when the kernel or the spec -> argument mapping changes
CACHE_DIR = Path(tempfile.gettempdir()) / 'source_explorer'
CACHE_LIMIT_BYTES = 1 << 30   # 1 GiB of saved runs, oldest evicted first (~40 million-pulse runs)


# ----------------------------------------------------------------- devices and drive
LASERS = OrderedDict([
    ('lo2025', ('DFB, 150 µm, 1547 nm (Lo et al. 2026, Chapter 5)', make_paper_laser)),
    ('dfb', ('DFB, 300 µm, 1550 nm (DFBLaserParams default)', lambda: make_laser('dfb'))),
    ('fp', ('Fabry–Pérot, 300 µm, cleaved facets', lambda: make_laser('fp'))),
])
_lasers = {}


def get_laser(key):
    if key not in _lasers:
        _lasers[key] = LASERS[key][1]()
    return _lasers[key]


def laser_summary(key):
    L = get_laser(key)
    return dict(key=key, name=LASERS[key][0], I_th_mA=L.threshold_current() * 1e3,
                tau_p_ps=L.tau_p * 1e12, lambda_nm=L.lambda0 * 1e9, L_um=L.L * 1e6,
                alpha_H=L.alpha_H, beta_sp=L.beta_sp)


SHAPES = OrderedDict([
    ('sine', dict(name='sine', params=[])),
    ('raised_cosine', dict(name='raised cosine', params=['duty', 't_rise_ps'])),
    ('square', dict(name='square', params=['duty'])),
    ('gaussian', dict(name='Gaussian', params=['center_frac', 'sigma_ps'])),
    ('trapezoid', dict(name='trapezoid', params=['duty', 't_rise_ps', 't_fall_ps'])),
    ('fourier', dict(name='Fourier-shaped', params=['duty', 'harmonics'])),
])


def _num(d, key, default, lo, hi):
    v = d.get(key, default)
    if v is None or v == '':
        v = default
    if isinstance(v, bool):                       # True would silently become 1.0
        raise ValueError(f'{key} must be a number, got {v!r}')
    try:
        v = float(v)
    except (TypeError, ValueError):
        raise ValueError(f'{key} must be a number, got {v!r}')
    if not math.isfinite(v) or v < lo or v > hi:
        raise ValueError(f'{key} = {v:g} is outside the allowed range [{lo:g}, {hi:g}]')
    return v


def _int(d, key, default, lo, hi):
    v = _num(d, key, default, lo, hi)
    if v != int(v):
        raise ValueError(f'{key} must be a whole number, got {v:g}')
    return int(v)


def normalise_spec(raw):
    """Validate a run specification and fill defaults. Only parameters the chosen waveform
    and injection mode actually use are kept, so editing an unused field never changes the
    run's id and never forces a re-simulation."""
    raw = raw or {}
    laser = raw.get('laser', 'lo2025')
    if laser not in LASERS:
        raise ValueError(f'unknown laser {laser!r}; choose from {", ".join(LASERS)}')
    shape = raw.get('shape', 'sine')
    if shape not in SHAPES:
        raise ValueError(f'unknown waveform {shape!r}; choose from {", ".join(SHAPES)}')
    spec = dict(
        laser=laser,
        f_rep_GHz=_num(raw, 'f_rep_GHz', 10.0, 0.1, 40.0),
        I_DC_mA=_num(raw, 'I_DC_mA', PAPER_I_DC * 1e3, 0.0, 1000.0),
        I_RF_mA=_num(raw, 'I_RF_mA', PAPER_I_RF * 1e3, 0.0, 1000.0),
        shape=shape,
        dt_ps=_num(raw, 'dt_ps', 0.5, 0.05, 5.0),
        n_pulses=_int(raw, 'n_pulses', 10_000, 100, MAX_PULSES),
        n_discard=_int(raw, 'n_discard', 200, 0, 100_000),
        seed=_int(raw, 'seed', 42, 0, 2 ** 31 - 1),
    )
    sp, need, params = raw.get('shape_params') or {}, SHAPES[shape]['params'], {}
    if 'duty' in need:
        params['duty'] = _num(sp, 'duty', 0.30, 0.02, 0.95)
    if 't_rise_ps' in need:
        params['t_rise_ps'] = _num(sp, 't_rise_ps', 7.5, 0.0, 1e4)
    if 't_fall_ps' in need:
        params['t_fall_ps'] = _num(sp, 't_fall_ps', 7.5, 0.0, 1e4)
    if 'center_frac' in need:
        params['center_frac'] = _num(sp, 'center_frac', 0.15, 0.0, 1.0)
    if 'sigma_ps' in need:
        params['sigma_ps'] = _num(sp, 'sigma_ps', 7.5, 0.05, 1e4)
    if 'harmonics' in need:
        hs = sp.get('harmonics', [[0.25, 0.0]])
        if (not isinstance(hs, (list, tuple)) or not 1 <= len(hs) <= 4
                or not all(isinstance(h, (list, tuple)) and len(h) == 2 for h in hs)):
            raise ValueError('harmonics must be a list of 1-4 [a, b] pairs')
        pairs = [[_num({'a': h[0]}, 'a', 0, -0.5, 0.5), _num({'b': h[1]}, 'b', 0, -0.5, 0.5)]
                 for h in hs]
        while len(pairs) > 1 and pairs[-1] == [0.0, 0.0]:
            pairs.pop()                     # trailing zero harmonics do not change the waveform
        params['harmonics'] = pairs
    spec['shape_params'] = params

    inj = raw.get('injection') or {}
    if not isinstance(inj, dict):
        raise ValueError("injection must be an object such as {'mode': 'sld', 'P_sld_mW': 19}")
    mode = inj.get('mode', 'sld')
    if mode == 'none':
        spec['injection'] = dict(mode='none')
    elif mode == 'sld':
        spec['injection'] = dict(mode='sld',
                                 P_sld_mW=_num(inj, 'P_sld_mW', 19.0, 0.0, 1000.0),
                                 acceptance_bw_nm=_num(inj, 'acceptance_bw_nm', 8.0, 0.01, 200.0),
                                 sld_bw_nm=_num(inj, 'sld_bw_nm', 33.0, 0.1, 500.0))
    elif mode == 'direct':
        spec['injection'] = dict(mode='direct',
                                 log10_S_inj=_num(inj, 'log10_S_inj', 19.0, 10.0, 25.0))
    else:
        raise ValueError(f'unknown injection mode {mode!r}; choose none, sld or direct')

    limit = max_dt_ps(laser)
    if spec['dt_ps'] > limit:
        raise ValueError(
            f"dt = {spec['dt_ps']:g} ps is too large for this laser: the field integrator is "
            f'Euler-Maruyama and a step near the photon lifetime '
            f'({get_laser(laser).tau_p * 1e12:.2f} ps) diverges. Use {limit:.2f} ps or less.')
    return spec


def max_dt_ps(laser_key):
    """Largest time step that keeps the kernel stable for this laser."""
    return STABILITY_FRACTION * get_laser(laser_key).tau_p * 1e12


def spec_id(spec):
    blob = json.dumps({'v': MODEL_VERSION, 'spec': spec}, sort_keys=True, separators=(',', ':'))
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def spec_label(spec, n=None):
    inj = spec['injection']
    injl = {'none': 'free-running',
            'sld': f"SLD {inj.get('P_sld_mW', 0):g} mW @ {inj.get('acceptance_bw_nm', 0):g} nm",
            'direct': f"S_inj 1e{inj.get('log10_S_inj', 0):g}"}[inj['mode']]
    n = spec['n_pulses'] if n is None else n
    return f"{spec['f_rep_GHz']:g} GHz · {SHAPES[spec['shape']]['name']} · {injl} · {n:,} pulses"


def drive_waveform(spec):
    """Current waveform over one period, and the time grid the kernel runs on.

    The grid follows run_config in studies/paper_10ghz_simulation.py: pts = round(T/dt), at
    least 100, then dt = T/pts. The sine is that study's builder. The pulse-shaped builders
    take I_off/I_on; they are driven with I_off = max(I_DC - I_RF, 0) and I_on = I_DC + I_RF,
    so changing the shape at fixed knobs compares shapes, not amplitudes.

    The Fourier shape is the exception: build_fourier multiplies the on-window by an envelope
    clipped to [0, 1] that starts at 0.5, so it reaches I_on only if the harmonic sum reaches
    0.5. With the default a1 = 0.25 it peaks at I_off + 0.75 (I_on - I_off) -- 89.6 mA against
    119.5 mA for every other shape at the Chapter 5 knobs."""
    f = spec['f_rep_GHz'] * 1e9
    T = 1.0 / f
    pts = max(int(round(T / (spec['dt_ps'] * 1e-12))), 100)
    dt = T / pts
    I_dc, I_rf = spec['I_DC_mA'] * 1e-3, spec['I_RF_mA'] * 1e-3
    shape, p = spec['shape'], spec['shape_params']
    if shape == 'sine':
        w = build_sine_waveform(pts, dt, f, I_dc, I_rf)
    else:
        I_off, I_on = max(I_dc - I_rf, 0.0), I_dc + I_rf
        t_on = p.get('duty', 0.3) * T
        if shape == 'raised_cosine':
            t_rise = p['t_rise_ps'] * 1e-12
            if t_rise > t_on / 2:
                raise ValueError(f'rise time {p["t_rise_ps"]:g} ps is longer than half the on-time '
                                 f'({t_on / 2 * 1e12:.3g} ps) at this duty cycle and rate')
            w = build_raised_cosine(pts, dt, I_off, I_on, t_on, t_rise)
        elif shape == 'square':
            w = build_square(pts, dt, I_off, I_on, t_on)
        elif shape == 'gaussian':
            w = build_gaussian(pts, dt, I_off, I_on, p['center_frac'] * T, p['sigma_ps'] * 1e-12)
        elif shape == 'trapezoid':
            t_rise, t_fall = p['t_rise_ps'] * 1e-12, p['t_fall_ps'] * 1e-12
            if t_rise + t_fall > t_on:
                raise ValueError(f'rise + fall ({p["t_rise_ps"] + p["t_fall_ps"]:g} ps) is longer than '
                                 f'the on-time ({t_on * 1e12:.3g} ps) at this duty cycle and rate')
            w = build_trapezoid(pts, dt, I_off, I_on, t_on, t_rise, t_fall)
        else:
            w = build_fourier(pts, dt, I_off, I_on, t_on,
                              np.asarray(p['harmonics'], dtype=np.float64).reshape(-1, 2))
    return np.ascontiguousarray(np.maximum(w, 0.0), dtype=np.float64), pts, dt, T


def drive_summary(spec, laser, w, pts, dt, T):
    I_th = laser.threshold_current()
    return dict(I_min_mA=float(w.min() * 1e3), I_max_mA=float(w.max() * 1e3),
                I_mean_mA=float(w.mean() * 1e3), I_th_mA=I_th * 1e3,
                above_threshold_frac=float(np.mean(w > I_th)),
                V_rf_amplitude_V=spec['I_RF_mA'] * 1e-3 * Z_MATCH,
                pts=pts, dt_ps=dt * 1e12, T_rep_ps=T * 1e12)


def injection_S(spec, laser):
    inj = spec['injection']
    if inj['mode'] == 'none':
        return 0.0
    if inj['mode'] == 'direct':
        return float(10.0 ** inj['log10_S_inj'])
    return float(sld_power_to_sinj(inj['P_sld_mW'], laser, sld_bw_nm=inj['sld_bw_nm'],
                                   acceptance_bw_nm=inj['acceptance_bw_nm']))


def simulate_arrays(spec):
    """Run the kernel with exactly the argument list of run_config. Returns the four per-pulse
    arrays (peak phase, peak photon density, mid-period density, peak index) and seconds."""
    laser = get_laser(spec['laser'])
    w, pts, dt, T = drive_waveform(spec)
    S_inj = injection_S(spec, laser)
    n, nd = spec['n_pulses'], spec['n_discard']
    t0 = time.perf_counter()
    arrays = simulate_pulses_waveform(
        n + nd, nd, pts, dt, w,
        laser.V, laser.Gamma, laser.v_g, laser.a, laser.N_tr, laser.epsilon,
        laser.A, laser.B, laser.C, laser.tau_p, laser.beta_sp, laser.alpha_H, q,
        S_inj, spec['seed'])
    # Stability depends on how hard the laser is driven as well as on dt, so check the output
    # rather than trusting the dt limit alone: a diverged run is all inf, and every metric
    # downstream would be silently meaningless.
    if not all(np.isfinite(a).all() for a in arrays[:3]):
        raise ValueError(
            f"the integrator diverged at dt = {spec['dt_ps']:g} ps with this drive "
            f"({spec['I_DC_mA']:g} +/- {spec['I_RF_mA']:g} mA); the photon lifetime is "
            f'{laser.tau_p * 1e12:.2f} ps. Reduce dt or the drive amplitude.')
    return arrays, time.perf_counter() - t0


# ----------------------------------------------------------------- cost calibration
_calib = {'s_per_step': 1.05e-7, 'measured': False}
_calib_lock = threading.Lock()
WARM = threading.Event()


def warm():
    """Compile the kernel, then measure its cost per time step on this machine."""
    simulate_arrays(normalise_spec({'n_pulses': 100, 'n_discard': 10}))
    spec = normalise_spec({'n_pulses': 3000, 'n_discard': 0})
    _, el = simulate_arrays(spec)
    with _calib_lock:
        _calib.update(s_per_step=el / (3000 * drive_waveform(spec)[1]), measured=True)
    WARM.set()
    return _calib['s_per_step']


def _record_cost(seconds, steps):
    if steps < 200_000:                     # too short to time reliably
        return
    with _calib_lock:
        s = seconds / steps
        _calib['s_per_step'] = s if not _calib['measured'] else 0.7 * _calib['s_per_step'] + 0.3 * s
        _calib['measured'] = True


def resolve_pulses(r1=0.0, var_cos=0.5):
    """Pulses needed to decide a measured r1 against R1_THRESHOLD at CONFIDENCE, either way: the
    uncertainty band sqrt(-ln(1-C)/M) has to fit in the gap to the threshold. r1 = 0 is the best
    case at 46,053 pulses; a measured 0.0062 needs 318,919 to be called randomised, and 0.0117
    needs 1,593,486 to be called a failure. None only when r1 sits exactly on the threshold,
    where no run length decides it."""
    gap = abs(R1_THRESHOLD - r1)
    if gap == 0:
        return None
    z2 = 2 * _erfinv(CONFIDENCE) ** 2                      # z^2 for the data-driven half-width
    need = max(-math.log(1 - CONFIDENCE), z2 * var_cos) / gap ** 2
    return int(math.floor(need)) + 2


def estimate(raw):
    spec = normalise_spec(raw)
    laser = get_laser(spec['laser'])
    w, pts, dt, T = drive_waveform(spec)
    steps = (spec['n_pulses'] + spec['n_discard']) * pts
    predicted = steps * _calib['s_per_step']
    did = spec_id(spec)
    return dict(id=did, spec=spec, label=spec_label(spec), steps=steps, steps_per_period=pts,
                predicted_s=predicted, live_ok=predicted <= LIVE_MAX_S, live_max_s=LIVE_MAX_S,
                cached=REGISTRY.has(did), calibrated=_calib['measured'],
                s_per_step=_calib['s_per_step'], S_inj=injection_S(spec, laser),
                drive=drive_summary(spec, laser, w, pts, dt, T), laser=laser_summary(spec['laser']),
                max_dt_ps=max_dt_ps(spec['laser']), resolve_pulses=resolve_pulses())


# ----------------------------------------------------------------- datasets
class Dataset:
    def __init__(self, spec, arrays, runtime_s, source, created=None):
        self.spec, self.id = spec, spec_id(spec)
        self.laser = get_laser(spec['laser'])
        self.waveform, self.pts, self.dt, self.T_rep = drive_waveform(spec)
        self.S_inj = injection_S(spec, self.laser)
        self.phi, self.S, self.samp, self.k = (np.asarray(a) for a in arrays)
        self.P = self.laser.output_power(np.maximum(self.S, 0))
        self.n = len(self.phi)
        self.runtime_s, self.source = float(runtime_s), source
        self.created = created or time.time()
        self._memo, self._eta, self._lock = {}, None, threading.Lock()

    def memo(self, key, fn):
        with self._lock:
            if key in self._memo:
                return self._memo[key]
        value = fn()
        with self._lock:
            self._memo[key] = value
        return value

    def amzi_at(self, psi_deg):
        """AMZI outputs at one phase. A single slot, not a memo: a slider would otherwise
        leave three N-length arrays behind for every angle it passed through."""
        with self._lock:
            if self._eta is not None and self._eta[0] == psi_deg:
                return self._eta[1]
        out = amzi_outputs(self.P, self.phi, math.radians(psi_deg))
        with self._lock:
            self._eta = (psi_deg, out)
        return out

    def summary(self):
        return dict(id=self.id, label=spec_label(self.spec, self.n), n=self.n, spec=self.spec,
                    source=self.source, runtime_s=self.runtime_s, created=self.created,
                    S_inj=self.S_inj, in_memory=True)


def trim_cache(limit=CACHE_LIMIT_BYTES, directory=None):
    """Keep the saved-run directory under a limit, dropping the oldest first. Without this it
    grows without bound -- a million-pulse run is ~24 MB and nothing ever removed one."""
    directory = Path(directory or CACHE_DIR)
    if not directory.exists():
        return []
    files = sorted(directory.glob('*.npz'), key=lambda f: f.stat().st_mtime, reverse=True)
    total, removed = 0, []
    for f in files:
        total += f.stat().st_size
        if total > limit:
            removed.append(str(f.name))
            f.unlink(missing_ok=True)
    return removed


def save_npz(path, spec, arrays, runtime_s):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + '.partial.npz')
    phi, S, samp, k = arrays
    np.savez(tmp, phi=phi, S=S, samp=samp, k=k, spec=np.array(json.dumps(spec)),
             runtime_s=np.array(runtime_s), n=np.array(len(phi)), created=np.array(time.time()))
    os.replace(tmp, path)                       # atomic: a half-written run is never visible
    trim_cache()


def load_npz(path):
    with np.load(path) as z:
        spec = json.loads(str(z['spec']))
        arrays = (z['phi'], z['S'], z['samp'], z['k'])
        return Dataset(spec, arrays, float(z['runtime_s']), 'saved', float(z['created']))


class Registry:
    """Datasets in memory, least recently used evicted past a pulse budget. Background runs
    are also on disk, so an evicted one reloads instead of re-running for minutes."""

    def __init__(self, budget=MEMORY_PULSE_BUDGET):
        self._d, self._lock, self.budget = OrderedDict(), threading.Lock(), budget

    def put(self, ds):
        with self._lock:
            self._d[ds.id] = ds
            self._d.move_to_end(ds.id)
            total = sum(d.n for d in self._d.values())
            while total > self.budget and len(self._d) > 1:
                _, old = self._d.popitem(last=False)
                total -= old.n

    def get(self, did):
        with self._lock:
            ds = self._d.get(did)
            if ds is not None:
                self._d.move_to_end(did)
                return ds
        path = CACHE_DIR / f'{did}.npz'
        if path.exists():
            ds = load_npz(path)
            self.put(ds)
            return ds
        raise KeyError(f'dataset {did} is no longer in memory and was never saved; simulate it again')

    def has(self, did):
        with self._lock:
            if did in self._d:
                return True
        return (CACHE_DIR / f'{did}.npz').exists()

    def listing(self):
        with self._lock:
            rows = [d.summary() for d in reversed(self._d.values())]
        seen = {r['id'] for r in rows}
        if CACHE_DIR.exists():
            for path in sorted(CACHE_DIR.glob('*.npz'), key=lambda p: -p.stat().st_mtime):
                if path.stem in seen or path.stem.endswith('.partial'):
                    continue
                try:
                    with np.load(path) as z:
                        spec, n = json.loads(str(z['spec'])), int(z['n'])
                        rows.append(dict(id=path.stem, label=spec_label(spec, n), n=n, spec=spec,
                                         source='saved', runtime_s=float(z['runtime_s']),
                                         created=float(z['created']), in_memory=False))
                except Exception:               # an unreadable file is skipped, not fatal
                    continue
        return rows


REGISTRY = Registry()
_live_lock = threading.Lock()


class TooSlow(Exception):
    def __init__(self, predicted):
        super().__init__(f'this run is predicted to take {predicted:.0f} s, beyond the {LIVE_MAX_S:g} s '
                         f'live limit; run it in the background instead')
        self.predicted = predicted


def simulate(raw):
    """Return (dataset, reused). Identical specifications reuse the cached run: the kernel is
    seeded, so a repeat would be byte-identical anyway."""
    spec = normalise_spec(raw)
    did = spec_id(spec)
    if REGISTRY.has(did):
        return REGISTRY.get(did), True
    steps = (spec['n_pulses'] + spec['n_discard']) * drive_waveform(spec)[1]
    predicted = steps * _calib['s_per_step']
    if predicted > LIVE_MAX_S:
        raise TooSlow(predicted)
    with _live_lock:                    # the kernel holds the GIL: running two at once gains nothing
        if REGISTRY.has(did):
            return REGISTRY.get(did), True
        arrays, el = simulate_arrays(spec)
        _record_cost(el, steps)
        ds = Dataset(spec, arrays, el, 'live')
        REGISTRY.put(ds)
        return ds, False


# ----------------------------------------------------------------- analysis
def _f(v, sig=6):
    v = float(v)
    return float(f'{v:.{sig}g}') if math.isfinite(v) else None


def _arr(a, sig=5, cap=None):
    """JSON-safe list: significant-figure rounding (fixed decimals would zero an Allan
    deviation of 1e-5) and NaN/inf as null, which JSON.parse accepts and NaN it does not."""
    a = np.asarray(a, dtype=np.float64).ravel()
    if cap and a.size > cap:
        a = a[np.linspace(0, a.size - 1, cap).astype(int)]
    return [float(f'{x:.{sig}g}') if math.isfinite(x) else None for x in a.tolist()]


def normalise_knobs(raw, ds):
    raw = raw or {}
    series = raw.get('acf_series', 'eta')
    if series not in ('eta', 'I_A'):
        raise ValueError("acf_series must be 'eta' or 'I_A'")
    return dict(
        psi_deg=round(_num(raw, 'psi_deg', 0.0, -1e6, 1e6) % 360.0, 3),
        n_bins=_int(raw, 'n_bins', 128, 8, 1024),
        max_lag=min(_int(raw, 'max_lag', 100, 1, LAG_CAP), max(1, ds.n // 4)),
        gate_ps=_num(raw, 'gate_ps', 30.0, 0.01, 1e6),
        allan_max_m=_int(raw, 'allan_max_m', 1000, 1, 1_000_000),
        acf_series=series,
    )


def verdict_band(dphi, confidence=CONFIDENCE):
    """Half-width of the interval on the TRUE r1 around the measurement.

    Two estimates, whichever is larger. The Rayleigh quantile sqrt(-ln(1-C)/M) assumes uniform
    phases, where Var(cos dphi) = 1/2; when dphi is concentrated near 0 and pi -- Chapter 5's
    partially-locked case -- the radial noise is sqrt(2) larger and that quantile is only a 96.8%
    interval, not 99%. So it is taken together with a data-driven half-width
    z * sqrt(Var(cos(dphi - arg r1)) / M), which widens exactly when the phase distribution
    demands it and never narrows the conservative case.
    """
    M = len(dphi)
    if M < 2:
        return float('inf')
    centred = np.cos(dphi - np.angle(np.mean(np.exp(1j * dphi))))
    z = math.sqrt(2) * _erfinv(confidence)
    return max(float(autocorr_ci(M, confidence)), float(z * math.sqrt(centred.var() / M)))


def _erfinv(x):
    from scipy.special import erfinv
    return float(erfinv(x))


def _verdict(r1, bound):
    """Decide against the threshold, not against zero correlation.

    `bound` is verdict_band(): the true r1 lies within +/- bound of the measurement with
    probability CONFIDENCE. A verdict is only given when that whole band sits on one side of
    R1_THRESHOLD.

    Comparing the measurement alone with the threshold (what this function did first) ignores
    the band: a source whose true r1 is 0.0105 -- a source that FAILS the requirement -- reads
    below 0.01 in 24% of million-pulse runs, and was called 'randomised, resolved' every time.
    """
    if r1 + bound < R1_THRESHOLD:
        return 'below'
    if r1 - bound > R1_THRESHOLD:
        return 'above'
    return 'unresolved'


def _timing_hist(k, dt_ps, n_bins):
    """Histogram of peak arrival on bins aligned to the kernel's dt: arrival is quantised to
    multiples of dt, and bins that are not aligned alias into a comb of empty bins."""
    k = np.asarray(k)
    k0, k1 = int(k.min()), int(k.max())
    width = max(1, math.ceil((k1 - k0 + 1) / n_bins))
    counts = np.bincount((k - k0) // width)
    centres = (k0 + (np.arange(len(counts)) + 0.5) * width - 0.5) * dt_ps
    return centres, counts, width * dt_ps


def analyse(ds, raw_knobs=None):
    t0 = time.perf_counter()
    kn = normalise_knobs(raw_knobs, ds)
    nb, max_lag = kn['n_bins'], kn['max_lag']
    n, M = ds.n, ds.n - 1

    # -- inter-pulse phase ---------------------------------------------------------
    r1, dphi = ds.memo('r1', lambda: compute_r1(ds.phi))
    prq = ds.memo(('prq', nb), lambda: {key: val for key, val in
                                         phase_randomisation_quality(ds.phi, n_bins=nb).items()
                                         if key != 'dphi'})
    bound = float(autocorr_ci(M, CONFIDENCE))          # the bound a perfectly random source respects
    band = verdict_band(dphi)                          # the interval on the true r1
    counts, edges = np.histogram(dphi, bins=nb, range=(-np.pi, np.pi))
    ks_n = min(M, KS_SLICE)
    g1_all = ds.memo('g1', lambda: phase_autocorrelation(ds.phi, min(LAG_CAP, n - 1) + 1))
    lags = np.arange(1, max_lag + 1)
    g1 = g1_all[1:max_lag + 1]
    g1_bound = np.sqrt(-np.log(1 - CONFIDENCE) / (n - lags))
    trace_n = min(n, 2000)
    phase = dict(
        r1=_f(r1), threshold=R1_THRESHOLD, r1_bound99=_f(bound),
        r1_mean_random=_f(math.sqrt(math.pi / (4 * M))),
        p_random_exceeds=_f(math.exp(-M * R1_THRESHOLD ** 2)),
        r1_band99=_f(band),
        verdict=_verdict(r1, band), significant=bool(r1 > bound),
        resolve_pulses=resolve_pulses(r1, float(np.cos(dphi - np.angle(np.mean(np.exp(1j * dphi)))).var())),
        resolve_pulses_ideal=resolve_pulses(0.0),
        sigma_phi=_f(prq['sig_phi']), sigma_phi_uniform=_f(math.pi / math.sqrt(3)),
        kl=_f(prq['kl']), kl_bias=_f((nb - 1) / (2 * M * math.log(2))),
        ks=_f(prq['ks_stat']), ks_n=ks_n, ks_crit99=_f(1.628 / math.sqrt(ks_n)),
        hist=dict(x=_arr(0.5 * (edges[1:] + edges[:-1])), density=_arr(counts / (M * np.diff(edges))),
                  bin=_f(2 * math.pi / nb), uniform=_f(1 / (2 * math.pi))),
        g1=dict(lags=lags.tolist(), value=_arr(g1), bound=_arr(g1_bound),
                outside=int(np.sum(g1 > g1_bound))),
        trace=dict(index=list(range(trace_n)), phi=_arr(np.angle(np.exp(1j * ds.phi[:trace_n])), 4)),
    )

    # -- AMZI -----------------------------------------------------------------------
    I_A, I_B, eta = ds.amzi_at(kn['psi_deg'])
    V, psi_opt, psi_vals, IA_mean = ds.memo('fringe', lambda: amzi_splitting_ratio(ds.P, ds.phi))
    counts, edges = np.histogram(eta, bins=nb, range=(0.0, 1.0))
    # Arcsine comparison. The reference Chapter 5 draws is the ideal Beta(1/2, 1/2) on [0, 1];
    # the sharper test is against the distribution uniform phase would give with THESE pulse
    # energies (eta_null_cdf), which leaves only the phase to explain any difference. Both
    # nulls are independent of psi, so they are memoised; only eta itself moves with psi.
    xg, Fg = ds.memo('eta_null', lambda: eta_null_cdf(ds.P))
    ks_ideal, ks_n = ks_distance(eta, arcsine_cdf)
    ks_null, _ = ks_distance(eta, lambda x: np.interp(x, xg, Fg))
    arcsine = dict(contrast=_f(math.sqrt(8.0 * float(eta.var()))),
                   contrast_amplitude=_f(ds.memo('v_amp', lambda: amplitude_contrast(ds.P))),
                   ks_ideal=_f(ks_ideal), ks_uniform_phase=_f(ks_null), ks_n=ks_n,
                   ks_crit99=_f(1.628 / math.sqrt(ks_n)))
    series = eta if kn['acf_series'] == 'eta' else I_A
    _, rho = intensity_autocorrelation(series, max_lag)
    bart = bartlett_bound(len(series), CONFIDENCE)
    # The white-noise bound is the wrong null here: both eta and I_A are built from consecutive
    # pulse pairs, so the pulse energies alone put structure at lag 1 (+0.0096 for I_A at the
    # Chapter 5 point, against a Bartlett bound of 0.0058). Compare with phases redrawn instead.
    envelope = ds.memo(('acf_null', kn['acf_series'], max_lag, kn['psi_deg']),
                       lambda: surrogate_acf_envelope(ds.P, max_lag, series=kn['acf_series'],
                                                      n_surr=12,
                                                      psi=math.radians(kn['psi_deg']))[1])
    amzi = dict(
        psi_deg=kn['psi_deg'], V=_f(V), psi_opt_deg=_f(math.degrees(psi_opt)),
        eta_mean=_f(eta.mean()), eta_std=_f(eta.std()),
        fringe=dict(psi_deg=_arr(np.degrees(psi_vals)), IA_mean_mW=_arr(IA_mean * 1e3)),
        eta_hist=dict(x=_arr(0.5 * (edges[1:] + edges[:-1])), density=_arr(counts / (len(eta) / nb)),
                      bin=_f(1.0 / nb),
                      ideal=_arr(arcsine_bin_density(edges)),
                      expected=_arr(np.diff(np.interp(edges, xg, Fg)) / np.diff(edges))),
        arcsine=arcsine,
        acf=dict(series=kn['acf_series'], lags=lags.tolist(), rho=_arr(rho[1:]),
                 bound=_f(bart), envelope=_arr(envelope),
                 outside=int(np.sum(np.abs(rho[1:]) > envelope)),
                 outside_bartlett=int(np.sum(np.abs(rho[1:]) > bart)),
                 expected_outside=_f((1 - CONFIDENCE) * max_lag)),
    )

    # -- timing -----------------------------------------------------------------------
    sigma_t, t_peak = absolute_jitter(ds.k, ds.dt)
    sigma_T, _ = period_jitter(ds.k, ds.dt, ds.T_rep)
    t_ps = t_peak * 1e12
    t_med = float(np.median(t_ps))
    centres, tcounts, twidth = _timing_hist(ds.k, ds.dt * 1e12, nb)
    tau, adev = ds.memo(('allan', kn['allan_max_m']),
                        lambda: allan_deviation(ds.k, ds.dt, ds.T_rep, max_m=kn['allan_max_m']))
    timing = dict(
        sigma_t_ps=_f(sigma_t * 1e12), sigma_T_ps=_f(sigma_T * 1e12),
        quant_floor_ps=_f(ds.dt * 1e12 / math.sqrt(12)), mean_ps=_f(t_ps.mean()), median_ps=_f(t_med),
        gate_ps=_f(kn['gate_ps']),
        in_gate=_f(np.mean(np.abs(t_ps - t_med) <= kn['gate_ps'] / 2)),
        hist=dict(x=_arr(centres), counts=[int(c) for c in tcounts], bin_ps=_f(twidth)),
        allan=dict(tau_s=_arr(tau), adev=_arr(adev), max_m=kn['allan_max_m'],
                   m_used=int(round(float(tau[-1]) / ds.T_rep)),
                   white_pm=_arr(math.sqrt(3.0) * sigma_t / np.asarray(tau)),
                   resolution=_f(ds.dt / ds.T_rep)),
    )

    # -- power -------------------------------------------------------------------------
    P_mW = ds.P * 1e3
    # a run saved before the stability guard can hold inf; report it rather than crashing here
    finite = P_mW[np.isfinite(P_mW)]
    pc, pe = (np.histogram(finite, bins=nb) if finite.size
              else (np.zeros(nb, dtype=int), np.arange(nb + 1, dtype=float)))
    mean_S, mean_samp = float(np.nanmean(ds.S)), float(np.nanmean(ds.samp))
    power = dict(
        mean_mW=_f(finite.mean()) if finite.size else None,
        cv=_f(finite.std() / finite.mean()) if finite.size and finite.mean() > 0 else None,
        non_finite=int(P_mW.size - finite.size),
        peak_to_mid_dB=_f(10 * math.log10(mean_S / mean_samp)) if mean_samp > 0 and mean_S > 0 else None,
        hist=dict(x=_arr(0.5 * (pe[1:] + pe[:-1])), counts=[int(c) for c in pc],
                  bin_mW=_f(pe[1] - pe[0])),
    )

    L = ds.laser
    drive = drive_summary(ds.spec, L, ds.waveform, ds.pts, ds.dt, ds.T_rep)
    drive.update(t_ps=_arr(np.arange(ds.pts) * ds.dt * 1e12, cap=800),
                 I_mA=_arr(ds.waveform * 1e3, cap=800))

    return dict(dataset=ds.summary(), knobs=kn, phase=phase, amzi=amzi, timing=timing,
                power=power, drive=drive, laser=laser_summary(ds.spec['laser']),
                timings=dict(analysis_ms=_f((time.perf_counter() - t0) * 1e3, 4)))


# ----------------------------------------------------------------- background runs
def _worker_main(inbox, outbox):
    """Worker process: compile once, then run jobs one at a time, writing each to disk.

    Messages travel over pipes, not multiprocessing.Queue. A Queue hands each message to a
    feeder thread, and the kernel holds this process's GIL for its whole run, so a 'started'
    put just before the kernel was delivered only after the kernel finished: a 1M-pulse job
    showed 'starting' with no progress for its entire 18 s. Connection.send writes at once."""
    try:
        outbox.send(('ready', warm()))
    except Exception as exc:                                   # pragma: no cover
        outbox.send(('fatal', f'{type(exc).__name__}: {exc}'))
        return
    while True:
        try:
            msg = inbox.recv()
        except EOFError:                                       # the server went away
            return
        if msg[0] == 'stop':
            return
        _, jid, spec, path = msg
        try:
            outbox.send(('started', jid, time.time()))
            arrays, el = simulate_arrays(spec)
            save_npz(Path(path), spec, arrays, el)
            outbox.send(('done', jid, el))
        except Exception as exc:
            outbox.send(('failed', jid, f'{type(exc).__name__}: {exc}'))


class JobManager:
    """One worker process, one job at a time, the rest queued. Cancelling a running job kills
    the process (the kernel cannot be interrupted) and a fresh worker starts with the next job."""

    TERMINAL = ('done', 'failed', 'cancelled')

    def __init__(self):
        self._jobs, self._pending = OrderedDict(), deque()
        self._lock = threading.Lock()
        self._proc = self._inbox = None
        self._busy, self._gen, self._seq = None, 0, 0
        self._worker_state, self._s_per_step = 'stopped', None

    def submit(self, raw):
        spec = normalise_spec(raw)
        did = spec_id(spec)
        steps = (spec['n_pulses'] + spec['n_discard']) * drive_waveform(spec)[1]  # validates the shape
        with self._lock:
            for job in self._jobs.values():
                if job['dataset_id'] == did and job['state'] not in self.TERMINAL:
                    return self._public(job)
            self._seq += 1
            job = dict(id=f'j{self._seq}', dataset_id=did, label=spec_label(spec), steps=steps,
                       n_pulses=spec['n_pulses'], state='queued', submitted=time.time(),
                       started=None, finished=None, runtime_s=None, error=None, note=None)
            self._jobs[job['id']] = job
            if REGISTRY.has(did):
                job.update(state='done', finished=time.time(), note='already simulated; reused')
            else:
                self._pending.append((job['id'], spec))
                self._dispatch_locked()
            return self._public(job)

    def list(self):
        with self._lock:
            return [self._public(j) for j in reversed(self._jobs.values())]

    def cancel(self, jid):
        with self._lock:
            job = self._jobs.get(jid)
            if job is None:
                raise KeyError(f'no job {jid}')
            if job['state'] in self.TERMINAL:
                return self._public(job)
            job.update(state='cancelled', finished=time.time())
            if self._busy == jid:
                proc, self._proc, self._busy = self._proc, None, None
                self._gen += 1                         # orphan the old worker's monitor
                self._worker_state = 'stopped'
                self._inbox.close()
                proc.terminate()
                threading.Thread(target=proc.join, daemon=True).start()
                self._dispatch_locked()
            else:
                self._pending = deque(p for p in self._pending if p[0] != jid)
            return self._public(job)

    def shutdown(self):
        with self._lock:
            if self._proc is not None and self._proc.is_alive():
                self._proc.terminate()

    # -- internals, all called with self._lock held -----------------------------------------
    def _public(self, job):
        out = dict(job)
        s = self._s_per_step or _calib['s_per_step']
        out['predicted_s'] = job['steps'] * s
        out['worker'] = self._worker_state
        if job['state'] == 'running' and job['started']:
            el = time.time() - job['started']
            out.update(elapsed_s=el, progress=min(0.99, el / out['predicted_s']),
                       eta_s=max(0.0, out['predicted_s'] - el))
        else:
            out['progress'] = 1.0 if job['state'] == 'done' else 0.0
        return out

    def _dispatch_locked(self):
        if self._busy is not None or not self._pending:
            return
        if self._proc is None or not self._proc.is_alive():
            self._start_worker_locked()
        jid, spec = self._pending.popleft()
        self._busy = jid
        self._jobs[jid]['state'] = 'starting'
        self._inbox.send(('run', jid, spec, str(CACHE_DIR / f"{self._jobs[jid]['dataset_id']}.npz")))

    def _start_worker_locked(self):
        ctx = mp.get_context('spawn')
        jobs_r, jobs_w = ctx.Pipe(duplex=False)
        events_r, events_w = ctx.Pipe(duplex=False)
        self._proc = ctx.Process(target=_worker_main, args=(jobs_r, events_w), daemon=True,
                                 name='source-explorer-worker')
        self._proc.start()
        jobs_r.close()
        events_w.close()                 # only the worker holds the write end: EOF means it died
        self._inbox = jobs_w
        self._gen += 1
        self._worker_state = 'starting'
        threading.Thread(target=self._monitor, args=(self._proc, events_r, self._gen), daemon=True).start()

    def _finish_locked(self, jid, **fields):
        job = self._jobs.get(jid)
        if job is not None and job['state'] not in self.TERMINAL:
            job.update(finished=time.time(), **fields)
        if self._busy == jid:
            self._busy = None
        self._dispatch_locked()

    def _monitor(self, proc, events, gen):
        while True:
            try:
                if not events.poll(0.5):
                    if proc.is_alive():
                        continue
                    raise EOFError
                msg = events.recv()
            except (EOFError, OSError):
                proc.join(1.0)
                events.close()
                with self._lock:
                    if gen == self._gen:
                        self._worker_state, self._proc = 'stopped', None
                        if self._busy is not None:
                            self._finish_locked(self._busy, state='failed',
                                                error=f'worker process exited (code {proc.exitcode})')
                return
            with self._lock:
                if gen != self._gen:
                    return
                kind = msg[0]
                if kind == 'ready':
                    self._worker_state, self._s_per_step = 'ready', msg[1]
                elif kind == 'started' and msg[1] in self._jobs:
                    self._jobs[msg[1]].update(state='running', started=msg[2])
                elif kind == 'done':
                    self._finish_locked(msg[1], state='done', runtime_s=msg[2])
                elif kind == 'failed':
                    self._finish_locked(msg[1], state='failed', error=msg[2])
                elif kind == 'fatal' and self._busy is not None:
                    self._finish_locked(self._busy, state='failed', error=msg[1])


JOBS = JobManager()


def meta():
    """Everything the page needs to build its controls."""
    dfb_th = get_laser('dfb').threshold_current() * 1e3
    presets = [
        dict(key='ch5_sld', label='Chapter 5: 10 GHz sine + 19 mW SLD',
             note='The drive Chapter 5 simulates: 45.5 mA bias, I_RF = 74 mA, taking 3.7 V as an '
                  'amplitude into 50 Ω, with 19 mW of SLD at the study default 8 nm acceptance '
                  'bandwidth. The experimental section quotes 3.7 V peak-to-peak, which is 37 mA — '
                  'see the next preset.',
             spec=normalise_spec({'laser': 'lo2025', 'shape': 'sine',
                                  'injection': {'mode': 'sld', 'P_sld_mW': 19.0}})),
        dict(key='ch5_sld_pp', label='Chapter 5: 10 GHz sine + 19 mW SLD, 3.7 V peak-to-peak',
             note='The same point read the other way: 3.7 V peak-to-peak into 50 Ω is a 37 mA '
                  'amplitude, so I(t) spans 8.5–82.5 mA and never reaches zero current. The thesis '
                  'is inconsistent between its experimental text and its parameter table; this '
                  'preset exists so the two readings can be compared.',
             spec=normalise_spec({'laser': 'lo2025', 'shape': 'sine', 'I_RF_mA': 37.0,
                                  'injection': {'mode': 'sld', 'P_sld_mW': 19.0}})),
        dict(key='ch5_free', label='Chapter 5: 10 GHz sine, free-running',
             note='The same drive with the SLD off.',
             spec=normalise_spec({'laser': 'lo2025', 'shape': 'sine', 'injection': {'mode': 'none'}})),
        dict(key='sweep_rc', label='S_inj sweep: 10 GHz raised cosine',
             note='studies/qkd_sinj_sweep.py drive: 300 µm DFB, 0.9 → 5 × I_th, 30% duty, '
                  'dt = 1 ps, 500 settling pulses dropped.',
             spec=normalise_spec({'laser': 'dfb', 'shape': 'raised_cosine', 'dt_ps': 1.0,
                                  'n_discard': 500,
                                  'I_DC_mA': round(2.95 * dfb_th, 3), 'I_RF_mA': round(2.05 * dfb_th, 3),
                                  'shape_params': {'duty': 0.30, 't_rise_ps': 7.5},
                                  'injection': {'mode': 'direct', 'log10_S_inj': 19.0}})),
    ]
    return dict(
        lasers=[laser_summary(key) for key in LASERS],
        shapes=[dict(key=key, **val) for key, val in SHAPES.items()],
        presets=presets, threshold=R1_THRESHOLD, confidence=CONFIDENCE,
        live_max_s=LIVE_MAX_S, max_pulses=MAX_PULSES, lag_cap=LAG_CAP,
        cache_limit_bytes=CACHE_LIMIT_BYTES,
        max_dt_ps={key: max_dt_ps(key) for key in LASERS},
        calibration=dict(_calib), warm=WARM.is_set(), resolve_pulses=resolve_pulses(),
        cache_dir=str(CACHE_DIR),
    )
