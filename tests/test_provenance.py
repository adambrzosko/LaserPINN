"""Validation of gsdfb.provenance and the run browser's back end.

    python tests/test_provenance.py                 # all checks
    python tests/test_provenance.py recording index # selected checks
"""
import json
import shutil
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                                                       # noqa: E402

from gsdfb import provenance as prov                                     # noqa: E402
from gsdfb.io import save_run                                            # noqa: E402


def _figure(path, text='x'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.plot([0, 1], [0, 1])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def check_recording():
    """A manifest records the parameters passed, the caller's constants and laser, and a
    fingerprint that moves when the source moves."""
    from core.dfb_laser import make_laser
    with tempfile.TemporaryDirectory() as tmp:
        figure = Path(tmp) / 'fig.png'
        _figure(figure)
        namespace = {'N_PULSES': 10_000, 'DT': 0.5e-12, 'FREQS': [1, 2, 3], 'laser': make_laser('dfb'),
                     'lowercase': 5, '_private': 1, 'plt': object(),
                     'BIG': np.arange(10_000), 'NOT_FINITE': float('nan')}
        record = prov.write_manifest(figure, params={'f_rep_GHz': 10, 'seed': 42}, namespace=namespace)
        again = prov.read_manifest(figure)
        assert again == record
        assert record['params'] == {'f_rep_GHz': 10, 'seed': 42}
        constants = record['constants']
        assert constants['N_PULSES'] == 10_000 and constants['DT'] == 0.5e-12
        assert constants['FREQS'] == [1, 2, 3]
        assert 'lowercase' not in constants and '_private' not in constants
        assert constants['BIG']['shape'] == [10_000] and 'min' in constants['BIG']   # summarised
        assert 'NOT_FINITE' not in constants                                          # NaN dropped
        assert record['laser']['I_th_mA'] > 0 and record['laser']['variable'] == 'laser'
        assert record['versions']['numpy'] == np.__version__
        assert len(record['code_fingerprint']) == 12 and record['code_files']

        # The fingerprint is the source AS THIS PROCESS IMPORTED IT. A study saves its figures
        # hours after importing, so hashing the file at save time would name code that never ran.
        target = ROOT / 'gsdfb' / 'provenance.py'
        before, _ = prov.code_fingerprint([target])
        text = target.read_text()
        try:
            target.write_text(text + '\n# fingerprint probe\n')
            during, _ = prov.code_fingerprint([target])
            assert during == before, 'a mid-run edit changed the recorded fingerprint'
            edited = prov.write_manifest(figure, namespace={})['edited_during_run']
            assert 'gsdfb/provenance.py' in edited, edited      # but it is reported
            prov._SNAPSHOT.pop(str(target), None)               # a fresh process sees the new file
            assert prov.code_fingerprint([target])[0] != before
        finally:
            target.write_text(text)
            prov._SNAPSHOT.pop(str(target), None)
        assert prov.code_fingerprint([target])[0] == before
        assert prov.write_manifest(figure, namespace={})['edited_during_run'] == []
    print('recording OK: params, constants, laser and versions recorded; oversized and non-finite '
          'values summarised or dropped; the fingerprint tracks the source')


def check_laser_capture():
    """The laser must be recorded even though the CLASS is imported alongside the instance, which
    it is in every script: the class passes the same duck-type test and calling it unbound raises,
    so the first implementation recorded laser = null everywhere."""
    from core.dfb_laser import DFBLaserParams, make_laser
    laser = make_laser('dfb')
    for namespace in ({'DFBLaserParams': DFBLaserParams, 'laser': laser},
                      {'laser': laser, 'DFBLaserParams': DFBLaserParams},
                      {'Params': DFBLaserParams, 'broken': object(), 'dfb': laser}):
        got = prov.capture_laser(namespace)
        assert got and abs(got['I_th_mA'] - laser.threshold_current() * 1e3) < 1e-9, namespace.keys()
    assert prov.capture_laser({'DFBLaserParams': DFBLaserParams}) is None      # class only: nothing
    assert prov.capture_laser({}) is None
    print('laser capture OK: an instance is found past the class and past unusable candidates')


def check_complex_and_odd_values():
    """A manifest must never be the thing that breaks a run."""
    assert prov._jsonable(np.array([1 + 2j, 3 + 4j] * 50))['dtype'].startswith('complex')
    assert prov._jsonable(np.array([], dtype=float)) is None
    assert prov._jsonable(np.float64(2.5)) == 2.5
    assert prov._jsonable({'a': float('inf'), 'b': 1}) == {'b': 1}
    # a summarised array (over MAX_LIST) must not smuggle NaN into the manifest: /api/runs would
    # then answer with invalid JSON and the run browser would fail to parse anything at all
    summary = prov._jsonable(np.array([1.0, np.nan, np.inf] + [2.0] * 40))
    json.dumps(summary, allow_nan=False)
    assert summary['min'] == 1.0 and summary['max'] == 2.0 and summary['non_finite'] == 2, summary
    json.dumps(prov._jsonable(np.full(50, np.nan)), allow_nan=False)
    assert prov._jsonable(object()) is None
    deep = {'a': {'b': {'c': {'d': 1}}}}
    json.dumps(prov._jsonable(deep))
    print('odd values OK: complex arrays, empty arrays, infinities and unserialisable objects '
          'are summarised or dropped, never raised')


def check_run_parameters():
    """The scalars gsdfb.io.save_run already writes into a run .npz are readable without loading
    the arrays beside them."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'run.npz'
        rng = np.random.default_rng(0)
        # just under save_run's 200k-element limit, so they are stored raw rather than
        # replaced by a histogram, and incompressible so the file really is megabytes
        save_run({'N_PULSES': 1_000_000, 'I_DC': 0.0455, 'label': 'ch5',
                  'big': rng.normal(size=199_000), 'big2': rng.normal(size=199_000),
                  'big3': rng.normal(size=199_000), 'freqs': np.arange(10) * 1e9}, str(path))
        size_mb = path.stat().st_size / 1e6
        assert size_mb > 4, size_mb
        t0 = time.perf_counter()
        params = prov.run_parameters(path)
        elapsed = time.perf_counter() - t0
        assert params['N_PULSES'] == 1_000_000 and abs(params['I_DC'] - 0.0455) < 1e-12
        assert not any(k.startswith('big') for k in params), 'a 199k-element array was read in'
        assert elapsed < 0.5, elapsed
    print(f'run parameters OK: scalars recovered from a {size_mb:.0f} MB run file in {elapsed*1e3:.0f} ms, '
          f'without reading the arrays')


def check_attribution():
    """Attribution finds the script behind a directory, from the code and from the README."""
    attribution = prov.attribution_map()
    assert 'studies/timing_jitter_analysis.py' in attribution.get('timing_jitter', [])
    assert 'core/million_pulse_comparison.py' in attribution.get('1M_pulse_comparison', [])
    assert 'studies/acceptance_bw_comparison.py' in attribution.get('acceptance_bw_comparison_v2', [])
    documented = prov.documented_attribution()
    assert 'core/gain_switched_interference.py' in documented.get('gain_switched', []), documented.get('gain_switched')
    for scripts in attribution.values():
        for s in scripts:
            assert (ROOT / s).exists(), s
    print(f'attribution OK: {len(attribution)} directories mapped to scripts that exist, from '
          f'img_dir()/path literals and the README table')


def check_index():
    """The index separates what was recorded from what was inferred, and flags a figure whose
    code changed after it was written."""
    with tempfile.TemporaryDirectory() as tmp:
        images = Path(tmp) / 'images'
        recorded, inferred = images / 'demo' / 'a.png', images / 'demo' / 'b.png'
        _figure(recorded)
        _figure(inferred)
        prov.write_manifest(recorded, params={'k': 1}, namespace={'N': 2})
        result = prov.index(images)
        assert result['totals']['figures'] == 2 and result['totals']['recorded'] == 1
        study = result['studies'][0]
        by_name = {f['name']: f for f in study['files']}
        assert by_name['a.png']['source'] == 'recorded'
        assert by_name['a.png']['manifest']['params'] == {'k': 1}
        assert by_name['b.png']['source'] == 'inferred' and by_name['b.png']['manifest'] is None
        assert not by_name['a.png']['stale']

        # a manifest whose recorded fingerprint no longer matches the source is flagged
        manifest = json.loads(prov.manifest_path(recorded).read_text())
        manifest['code_fingerprint'] = 'deadbeefcafe'
        prov.manifest_path(recorded).write_text(json.dumps(manifest))
        flagged = prov.index(images)['studies'][0]['files']
        assert any(f['name'] == 'a.png' and f['stale'] for f in flagged)
    print('index OK: recorded and inferred are distinguished, and a fingerprint that no longer '
          'matches the source is flagged stale')


def check_archive():
    """Archiving moves, never deletes, and leaves an index of what went where."""
    with tempfile.TemporaryDirectory() as tmp:
        source = ROOT / 'images' / '_archive_probe'
        source.mkdir(parents=True, exist_ok=True)
        (source / 'junk.png').write_bytes(b'x' * 100)
        try:
            moved = prov.archive_paths(['images/_archive_probe/junk.png'], Path(tmp) / 'arch')
            assert len(moved) == 1 and moved[0]['bytes'] == 100
            assert not (source / 'junk.png').exists()
            landed = Path(tmp) / 'arch' / 'images/_archive_probe/junk.png'
            assert landed.exists() and landed.read_bytes() == b'x' * 100
            written = json.loads((Path(tmp) / 'arch' / 'ARCHIVE_INDEX.json').read_text())
            assert written['total_bytes'] == 100 and written['moved'][0]['from'].endswith('junk.png')
        finally:
            shutil.rmtree(source, ignore_errors=True)
    print('archive OK: files move with their layout, an index records the move, nothing is deleted')


def check_http():
    """The run-browser routes answer, and /figures cannot be walked out of images/."""
    from app import server
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{httpd.server_address[1]}'

    def get(path):
        try:
            with urllib.request.urlopen(base + path) as r:
                return r.status, r.read(), r.headers.get('Content-Type')
        except urllib.error.HTTPError as err:
            return err.code, err.read(), None

    try:
        assert get('/runs')[0] == 200
        status, body, _ = get('/api/runs')
        index = json.loads(body)
        assert status == 200 and index['totals']['figures'] > 0
        first = next(f for s in index['studies'] for f in s['files'] if f['kind'] == 'figure')
        status, body, ctype = get('/figures/' + first['path'][len('images/'):])
        assert status == 200 and ctype == 'image/png' and body[:4] == b'\x89PNG'
        for escape in ('/figures/../README.md', '/figures/..%2f..%2fREADME.md',
                       '/figures/../../gsdfb/provenance.py'):
            code, _, _ = get(escape)
            assert code in (403, 404), (escape, code)
        assert json.loads(get('/api/runs/cleanup')[1])['total_bytes'] >= 0
    finally:
        httpd.shutdown()
    print('http OK: /runs, /api/runs and /figures serve; path traversal out of images/ is refused')


CHECKS = {
    'recording': check_recording,
    'laser': check_laser_capture,
    'odd_values': check_complex_and_odd_values,
    'run_parameters': check_run_parameters,
    'attribution': check_attribution,
    'index': check_index,
    'archive': check_archive,
    'http': check_http,
}

if __name__ == '__main__':
    for name in (sys.argv[1:] or CHECKS):
        CHECKS[name]()
    if not sys.argv[1:]:
        print('\nAll provenance checks passed.')
