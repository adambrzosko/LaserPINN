"""
Provenance for saved results: what produced a figure, from which parameters, and whether the
code has moved on since.

The problem this solves, measured on this repository: images/ held 155 figures and 792 MB of
data with no record beyond file timestamps, in directories whose neighbouring files came from
different runs -- images/timing_jitter/jitter_vs_freq.png was two days newer than the
allan_deviation.png beside it, with nothing to say whether they shared parameters. 26 of those
figures are pasted verbatim into the thesis (content-hash matched; index()['totals']['in_thesis']
is the live count), so "which run made this?" is a question about published numbers.

Two levels of evidence, never conflated:

  RECORDED   gsdfb.plotting.save_fig writes a sidecar <figure>.json at save time: the script and
             command, the parameters, the laser, library versions, and a fingerprint of the
             project source that was imported. This is a record.

  INFERRED   for figures saved before this module existed, attribution comes from scanning the
             scripts for the images/ directory they write to, and staleness from comparing file
             times against the source files. This is evidence, and is labelled as such
             everywhere it is shown. It never claims to know the parameters, because nothing on
             disk records them.

    from gsdfb.provenance import write_manifest, read_manifest, index
    write_manifest('images/demo/fig.png', params={'f_rep_GHz': 10})
    index()['studies']          # everything the run browser shows
"""
import ast
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / 'images'
MANIFEST_VERSION = 1
FIGURE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.svg'}
DATA_SUFFIXES = {'.npz', '.npy', '.csv', '.json', '.txt', '.pdf'}
MAX_LIST = 24                      # longer sequences are summarised, not stored whole
THESIS = Path(os.environ.get('PHD_THESIS_DIR', Path.home() / 'Documents/Work/PhD/Thesis/PhD-Thesis'))


# ----------------------------------------------------------------- recording
def _relative(path):
    """Repository-relative where possible; a path outside the tree (a test's temporary
    directory, an images/ folder elsewhere) is kept as it is rather than raising."""
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


def _jsonable(value, depth=0):
    """Convert a value to something JSON can hold, or return None to drop it. Numpy scalars and
    short arrays come through; arrays of data do not (a manifest is a record, not a dataset)."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        # NaN and inf serialise as NaN/Infinity, which JSON.parse in the browser rejects
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if depth > 2:
        return None
    if hasattr(value, 'item') and getattr(value, 'shape', None) == ():
        return _jsonable(value.item(), depth + 1)
    if hasattr(value, 'tolist') and hasattr(value, 'shape'):
        if value.size == 0 or value.size > MAX_LIST:
            if not value.size or value.dtype.kind not in 'fiub':   # complex/object: shape only
                return {'shape': list(value.shape), 'dtype': str(value.dtype)} if value.size else None
            summary = {'shape': list(value.shape), 'dtype': str(value.dtype)}
            good = value[np.isfinite(value)] if value.dtype.kind == 'f' else value
            if good.size:               # an unguarded NaN here reaches the browser as invalid JSON
                summary.update(min=float(good.min()), max=float(good.max()))
            if good.size != value.size:
                summary['non_finite'] = int(value.size - good.size)
            return summary
        return [_jsonable(v, depth + 1) for v in value.tolist()]
    if isinstance(value, (list, tuple, set)):
        seq = list(value)
        if len(seq) > MAX_LIST:
            return {'length': len(seq), 'first': _jsonable(seq[0], depth + 1),
                    'last': _jsonable(seq[-1], depth + 1)}
        return [_jsonable(v, depth + 1) for v in seq]
    if isinstance(value, dict):
        return {str(k): _jsonable(v, depth + 1) for k, v in value.items()
                if _jsonable(v, depth + 1) is not None}
    if hasattr(value, '__dataclass_fields__'):
        return {f: _jsonable(getattr(value, f), depth + 1) for f in value.__dataclass_fields__}
    return None


def capture_constants(namespace):
    """Module-level CONSTANTS of the calling script, so an unmodified script still records what it
    was configured with: N_PULSES, DT, DUTY, freqs... Upper-case names only, values JSON can hold.
    This is a convenience, not a substitute for passing params explicitly."""
    out = {}
    for name, value in (namespace or {}).items():
        if name.startswith('_') or not name.isupper():
            continue
        converted = _jsonable(value)
        if converted is not None:
            out[name] = converted
    return out


def capture_laser(namespace):
    """Any laser parameter set in the caller's namespace, with its derived quantities.

    Skips classes and keeps looking after a failure. The DFBLaserParams CLASS passes the
    duck-type test -- lambda0 is a class attribute and threshold_current a method -- and calling
    it unbound raises, so matching the class and giving up recorded laser = null in every script
    that imports the class alongside its instance, which is all of them.
    """
    for name, value in (namespace or {}).items():
        if isinstance(value, type) or not (hasattr(value, 'threshold_current')
                                           and hasattr(value, 'lambda0')):
            continue
        try:
            return {'variable': name, 'lambda0_nm': value.lambda0 * 1e9, 'L_um': value.L * 1e6,
                    'I_th_mA': value.threshold_current() * 1e3, 'tau_p_ps': value.tau_p * 1e12,
                    'alpha_H': value.alpha_H, 'beta_sp': value.beta_sp,
                    'cavity_type': getattr(value, 'cavity_type', None)}
        except Exception:
            continue                       # not a usable laser; try the next name
    return None


_SNAPSHOT = {}          # source path -> digest, as first seen in this process
RUN_STARTED = time.time()


def _sha1(path):
    h = hashlib.sha1()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def snapshot_digest(path):
    """The digest of a source file as this process first saw it.

    write_manifest runs when a figure is saved, which in a long study is hours after the code was
    imported and executed. Hashing the file then records whatever is on disk at that moment, so
    editing a module mid-run made the manifest name code that never produced the figure -- and
    the run browser then called that figure recorded and current.
    """
    key = str(path)
    if key not in _SNAPSHOT:
        _SNAPSHOT[key] = _sha1(path)
    return _SNAPSHOT[key]


def project_sources():
    """Project source files currently imported: the code that is actually producing the figure."""
    files = set()
    for module in list(sys.modules.values()):
        f = getattr(module, '__file__', None)
        if not f:
            continue
        p = Path(f)
        try:
            p.relative_to(ROOT)
        except ValueError:
            continue
        if p.suffix == '.py' and '.venv' not in p.parts:
            files.add(p)
    return sorted(files)


def code_fingerprint(files=None):
    """One hash over the project source in play, plus the per-file hashes. A figure whose
    fingerprint differs from today's was made by different code -- which is the honest question,
    rather than guessing from timestamps."""
    files = files or project_sources()
    per_file = {}
    combined = hashlib.sha1()
    for p in files:
        try:
            digest = snapshot_digest(p)
        except OSError:
            continue
        rel = str(p.relative_to(ROOT))
        per_file[rel] = digest[:12]
        combined.update(rel.encode())
        combined.update(digest.encode())
    return combined.hexdigest()[:12], per_file


def versions():
    out = {'python': platform.python_version()}
    for name in ('numpy', 'scipy', 'numba', 'matplotlib'):
        module = sys.modules.get(name)
        out[name] = getattr(module, '__version__', None) if module else None
    return out


def manifest_path(figure_path):
    p = Path(figure_path)
    return p.with_suffix(p.suffix + '.json')


def write_manifest(figure_path, params=None, namespace=None, extra=None, runtime_s=None):
    """Record what produced `figure_path`, beside it as <figure>.png.json."""
    figure_path = Path(figure_path)
    main = sys.modules.get('__main__')
    namespace = namespace if namespace is not None else getattr(main, '__dict__', {})
    script = getattr(main, '__file__', None)
    fingerprint, per_file = code_fingerprint()
    edited = []
    for rel in per_file:
        try:
            if _sha1(ROOT / rel) != _SNAPSHOT.get(str(ROOT / rel)):
                edited.append(rel)
        except OSError:
            continue
    record = {
        'manifest_version': MANIFEST_VERSION,
        'figure': _relative(figure_path),
        'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'created_unix': time.time(),
        'script': _relative(script) if script else None,
        'command': ' '.join(sys.argv),
        'params': _jsonable(params or {}) or {},
        'constants': capture_constants(namespace),
        'laser': capture_laser(namespace),
        'versions': versions(),
        'code_fingerprint': fingerprint,     # the source as imported, not as it sits on disk now
        'code_files': per_file,
        'run_started': time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime(RUN_STARTED)),
        'edited_during_run': edited,         # sources changed after this process imported them
        'runtime_s': runtime_s,
    }
    if extra:
        record.update(_jsonable(extra) or {})
    path = manifest_path(figure_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True))
    return record


def read_manifest(figure_path):
    path = manifest_path(figure_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


# ----------------------------------------------------------------- inference for older results
def _script_files():
    return [p for folder in ('core', 'studies', 'pinn', 'fiber', 'app', 'tests')
            for p in sorted((ROOT / folder).rglob('*.py'))
            if '__pycache__' not in p.parts]


def attribution_map(include_readme=True):
    """{images subdirectory -> [scripts that write there]}, by reading the source: every
    img_dir('name'), 'images/name/...' literal and IMG_DIR-style constant. Inference, not a
    record: it attributes a directory, and says nothing about which run filled it."""
    out = {}
    for path in _script_files():
        try:
            tree = ast.parse(path.read_text())
        except (OSError, SyntaxError):
            continue
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, 'id', '') == 'img_dir' and node.args:
                if isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    names.add(node.args[0].value)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if 'images/' in node.value:
                    part = node.value.split('images/', 1)[1].strip('/')
                    if part:
                        names.add(part.split('/')[0])
            elif isinstance(node, ast.JoinedStr):        # f'images/{tag}/...': take the literal head
                head = ''
                for piece in node.values:
                    if isinstance(piece, ast.Constant) and isinstance(piece.value, str):
                        head += piece.value
                    else:
                        break
                if 'images/' in head:
                    part = head.split('images/', 1)[1].strip('/')
                    if part and '/' in head.split('images/', 1)[1]:
                        names.add(part.split('/')[0])
        for name in names:
            out.setdefault(name, []).append(str(path.relative_to(ROOT)))
    if include_readme:
        for name, scripts in documented_attribution().items():
            for script in scripts:
                if (ROOT / script).exists() and script not in out.get(name, []):
                    out.setdefault(name, []).append(script)
    return out


def _imports_of(script):
    """Project modules a script imports, one level: enough to know which physics it depends on."""
    try:
        tree = ast.parse((ROOT / script).read_text())
    except (OSError, SyntaxError):
        return []
    roots = ('core', 'gsdfb', 'fiber', 'studies', 'pinn')
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(roots):
            found.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(roots):
                    found.add(alias.name)
    paths = []
    for module in sorted(found):
        p = ROOT / (module.replace('.', '/') + '.py')
        if p.exists():
            paths.append(str(p.relative_to(ROOT)))
    return paths


def run_parameters(npz_path, max_bytes=4096):
    """Scalars already saved on disk by gsdfb.io.save_run, which flattens a script's namespace
    into the run's .npz -- I_DC, V_RF_AMP, N_PULSES and the rest are in there. Only members
    under max_bytes are read, so this costs nothing on a 73 MB file of raw arrays.

    These belong to the RUN, not to one figure: a directory whose figures were written days
    apart does not have one parameter set, and the browser says so."""
    import zipfile
    import numpy.lib.format as npformat
    out = {}
    try:
        with zipfile.ZipFile(npz_path) as archive:
            for info in archive.infolist():
                if not info.filename.endswith('.npy') or info.file_size > max_bytes:
                    continue
                try:
                    with archive.open(info) as handle:
                        value = _jsonable(npformat.read_array(handle, allow_pickle=False))
                except Exception:
                    continue
                if value is not None:
                    out[info.filename[:-4]] = value
    except (OSError, zipfile.BadZipFile):
        return {}
    return out


def documented_attribution(readme=None):
    """Script -> images/ directory as DOCUMENTED in README.md. Some scripts write their figures
    to the working directory and the results were filed by hand, so the code cannot say where
    they went; the README table can."""
    readme = Path(readme or ROOT / 'README.md')
    if not readme.exists():
        return {}
    out = {}
    for line in readme.read_text().splitlines():
        if 'images/' not in line or '|' not in line:
            continue
        scripts = [w.strip('`') for w in line.split('`') if w.endswith('.py')]
        for piece in line.split('images/')[1:]:
            name = piece.split('/')[0].split('`')[0].strip().strip('/')
            for script in scripts if name else ():
                if '/' in script:
                    out.setdefault(name, []).append(script)
                    continue
                for folder in ('studies', 'core', 'pinn', 'fiber', 'tests'):
                    if (ROOT / folder / script).exists():
                        out.setdefault(name, []).append(f'{folder}/{script}')
                        break
    return {k: sorted(set(v)) for k, v in out.items()}


def thesis_usage():
    """{md5 -> [thesis figure paths]}: which working figures were pasted into the thesis. A
    figure used in the thesis is one whose provenance actually matters."""
    if not THESIS.exists():
        return {}
    out = {}
    for p in THESIS.rglob('Figs/*'):
        if p.suffix.lower() in FIGURE_SUFFIXES | {'.pdf'}:
            try:
                out.setdefault(hashlib.md5(p.read_bytes()).hexdigest(), []).append(
                    str(p.relative_to(THESIS)))
            except OSError:
                continue
    return out


# ----------------------------------------------------------------- the index the browser shows
def _file_record(path, thesis, attribution, script_times):
    stat = path.stat()
    rel = _relative(path)
    manifest = read_manifest(path) if path.suffix in FIGURE_SUFFIXES else None
    scripts = attribution.get(path.parent.name, [])
    record = {
        'path': rel,
        'name': path.name,
        'kind': 'figure' if path.suffix in FIGURE_SUFFIXES else 'data',
        'bytes': stat.st_size,
        'modified': stat.st_mtime,
        'scripts': scripts,
        'source': 'recorded' if manifest else 'inferred',
        'manifest': manifest,
        'stale': [],
    }
    if path.suffix in FIGURE_SUFFIXES and thesis:
        try:
            record['thesis'] = thesis.get(hashlib.md5(path.read_bytes()).hexdigest(), [])
        except OSError:
            record['thesis'] = []
    else:
        record['thesis'] = []
    if manifest:
        current, _ = code_fingerprint([ROOT / f for f in manifest.get('code_files', {})
                                       if (ROOT / f).exists()])
        if manifest.get('code_fingerprint') and current != manifest['code_fingerprint']:
            record['stale'].append('the recorded code fingerprint no longer matches this source tree')
    else:
        newer = [s for s in script_times.get(path.parent.name, [])
                 if script_times['mtimes'].get(s, 0) > stat.st_mtime]
        if newer:
            record['stale'].append('code changed after this file was written: ' + ', '.join(newer))
    return record


def index(images_dir=None):
    """Everything the run browser needs: one entry per images/ subdirectory, with its files,
    attribution, thesis usage and staleness."""
    images_dir = Path(images_dir or IMAGES)
    thesis = thesis_usage()
    attribution = attribution_map()
    script_times = {'mtimes': {}}
    for study, scripts in attribution.items():
        deps = []
        for s in scripts:
            deps.extend([s] + _imports_of(s))
        deps = sorted(set(deps))
        script_times[study] = deps
        for d in deps:
            p = ROOT / d
            if p.exists():
                script_times['mtimes'][d] = p.stat().st_mtime

    studies = []
    for folder in sorted(p for p in images_dir.iterdir() if p.is_dir()):
        files = [f for f in sorted(folder.rglob('*')) if f.is_file() and not f.name.startswith('.')]
        records = [_file_record(f, thesis, attribution, script_times) for f in files
                   if f.suffix in FIGURE_SUFFIXES | DATA_SUFFIXES and not f.name.endswith('.png.json')]
        if not records:
            continue
        figures = [r for r in records if r['kind'] == 'figure']
        data_files = [Path(r['path']) if Path(r['path']).is_absolute() else ROOT / r['path']
                      for r in records
                      if r['path'].endswith('.npz') and '_old' not in Path(r['path']).stem]
        run_data = None
        if data_files:
            newest = max(data_files, key=lambda p: p.stat().st_mtime)
            params = run_parameters(newest)
            if params:
                run_data = {'file': _relative(newest),
                            'modified': newest.stat().st_mtime, 'params': params,
                            'covers': [r['name'] for r in figures
                                       if abs(r['modified'] - newest.stat().st_mtime) < 3600]}
        studies.append({
            'study': folder.name,
            'scripts': attribution.get(folder.name, []),
            'run_data': run_data,
            'files': records,
            'n_figures': len(figures),
            'n_data': len(records) - len(figures),
            'bytes': sum(r['bytes'] for r in records),
            'modified': max(r['modified'] for r in records),
            'spread_days': (max(r['modified'] for r in records)
                            - min(r['modified'] for r in records)) / 86400,
            'recorded': sum(1 for r in figures if r['source'] == 'recorded'),
            'in_thesis': sum(1 for r in records if r['thesis']),
            'stale': sum(1 for r in records if r['stale']),
        })
    return {
        'root': str(ROOT), 'images': str(images_dir),
        'thesis': str(THESIS) if THESIS.exists() else None,
        'studies': studies,
        'totals': {
            'studies': len(studies),
            'figures': sum(s['n_figures'] for s in studies),
            'data': sum(s['n_data'] for s in studies),
            'bytes': sum(s['bytes'] for s in studies),
            'recorded': sum(s['recorded'] for s in studies),
            'in_thesis': sum(s['in_thesis'] for s in studies),
            'stale': sum(s['stale'] for s in studies),
        },
        'generated': time.time(),
    }


def cleanup_candidates(images_dir=None):
    """Files that are safe to archive, with the reason and the space each frees. Nothing is
    deleted here: the caller decides, and archive_paths() moves rather than removes."""
    images_dir = Path(images_dir or IMAGES)
    out = []
    for path in sorted(images_dir.rglob('*')):
        if not path.is_file():
            continue
        rel = _relative(path)
        size = path.stat().st_size
        if path.name == '.DS_Store':
            out.append({'path': rel, 'bytes': size, 'reason': 'macOS Finder metadata'})
        elif '_old' in path.stem:
            live = path.with_name(path.name.replace('_old', ''))
            if live.exists():
                out.append({'path': rel, 'bytes': size,
                            'reason': f'superseded by {live.name}, which still exists'})
    seen = {}
    for path in sorted(images_dir.rglob('*')):
        if path.is_file() and path.suffix in FIGURE_SUFFIXES:
            digest = hashlib.md5(path.read_bytes()).hexdigest()
            if digest in seen:
                out.append({'path': _relative(path), 'bytes': path.stat().st_size,
                            'reason': f'byte-identical to {seen[digest]}'})
            else:
                seen[digest] = _relative(path)
    return out


def archive_paths(paths, destination):
    """Move paths under `destination`, keeping their relative layout, and write an index of what
    moved. Reversible on purpose: simulation output that took an hour to make is not deleted on a
    tidy-up."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    moved = []
    for rel in paths:
        src = ROOT / rel
        if not src.exists():
            continue
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        size = sum(f.stat().st_size for f in src.rglob('*') if f.is_file()) if src.is_dir() \
            else src.stat().st_size
        os.replace(src, target) if src.is_file() else subprocess.run(
            ['mv', str(src), str(target)], check=True)
        moved.append({'from': rel, 'to': _relative(target), 'bytes': size})
    (destination / 'ARCHIVE_INDEX.json').write_text(json.dumps(
        {'moved': moved, 'when': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
         'total_bytes': sum(m['bytes'] for m in moved)}, indent=2))
    return moved
