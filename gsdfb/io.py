"""
Run-data persistence for the analysis scripts.

The study scripts historically plotted straight from in-memory arrays and exited,
so restyling a figure meant re-running the whole simulation.  `save_run` dumps a
script's namespace to a single .npz so the figures can be redrawn from disk.

Usage, at the very end of a script's __main__ block:

    from gsdfb.io import save_run
    save_run(locals(), 'images/<study>/data.npz')

and to read it back:

    from gsdfb.io import load_run
    d = load_run('images/<study>/data.npz')
    d['results.10GHz_free.r1']

Nested dicts are flattened with '.' separators, so a value originally at
results['10GHz_free']['r1'] is stored under the key 'results.10GHz_free.r1'.
Non-data objects (modules, functions, figures, laser parameter objects) are skipped.

Arrays with more than 200k elements are raw per-pulse ensembles.  Keeping them
verbatim makes the archive hundreds of megabytes, and every figure that uses one
draws a histogram of it, so they are stored as

    <name>__hist_counts   512 bin counts
    <name>__hist_edges    513 bin edges
    <name>__stats         [mean, std, min, max, n]

which is enough to redraw the distribution figures and to re-bin them coarser.
"""
import numpy as np

_SCALARS = (int, float, bool, np.integer, np.floating, np.bool_)
_MAXDEPTH = 4
_MAX_ELEMS = 200_000     # arrays larger than this are stored as histograms
_HIST_BINS = 512


def _clean(key):
    """Make a dict key usable as an npz field name."""
    if isinstance(key, tuple):
        return '_'.join(_clean(k) for k in key)
    if isinstance(key, float):
        return repr(key).replace('.', 'p').replace('-', 'm').replace('+', '')
    return str(key).replace('.', 'p').replace(' ', '_').replace('/', '_')


def _walk(obj, prefix, out, depth=0):
    if depth > _MAXDEPTH:
        return

    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return
        if obj.size > _MAX_ELEMS and obj.dtype.kind in 'fiu':
            # A raw per-pulse ensemble.  Storing 10^6 doubles per frequency per
            # case makes the archive hundreds of MB, and every figure that uses
            # one plots a histogram of it, so keep the distribution and the
            # moments rather than the samples.
            f = obj.ravel()
            if f.dtype.kind == 'f':
                f = f[np.isfinite(f)]
            if f.size == 0:
                return
            counts, edges = np.histogram(f, bins=_HIST_BINS)
            out[prefix + '__hist_counts'] = counts.astype(np.int64)
            out[prefix + '__hist_edges'] = edges
            out[prefix + '__stats'] = np.array(
                [f.mean(), f.std(), f.min(), f.max(), f.size])
            return
        out[prefix] = obj
        return

    if isinstance(obj, _SCALARS):
        out[prefix] = np.asarray(obj)
        return

    if isinstance(obj, str):
        out[prefix] = np.asarray(obj)
        return

    if isinstance(obj, dict):
        for k, v in obj.items():
            _walk(v, f'{prefix}.{_clean(k)}' if prefix else _clean(k), out, depth + 1)
        return

    if isinstance(obj, (list, tuple)):
        # a homogeneous numeric sequence is data; anything else is structure
        try:
            arr = np.asarray(obj)
        except Exception:
            arr = None
        if arr is not None and arr.dtype != object and arr.size:
            out[prefix] = arr
            return
        for i, v in enumerate(obj):
            _walk(v, f'{prefix}.{i}', out, depth + 1)
        return

    # anything else (modules, functions, figures, dataclasses) is not run data


def save_run(namespace, path, skip=()):
    """Flatten a script namespace into a .npz of every array and scalar in it.

    Parameters
    ----------
    namespace : dict   usually `locals()` from the script's __main__ block
    path      : str    output .npz path; parent directories are created
    skip      : iterable of names to exclude in addition to the defaults
    """
    import os
    skip = set(skip) | {'np', 'plt', 'os', 'sys', 'matplotlib', 'save_run',
                        'load_run', 'laser', 'sld', 'inj', 'fiber', 'params'}
    out = {}
    for name, val in namespace.items():
        if name.startswith('_') or name in skip:
            continue
        if callable(val) or type(val).__module__ not in (
                'builtins', 'numpy', 'numpy.core.multiarray'):
            # allow plain containers and numpy data through, drop everything else
            if not isinstance(val, (dict, list, tuple, np.ndarray) + _SCALARS + (str,)):
                continue
        _walk(val, name, out)

    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    np.savez_compressed(path, **out)
    print(f"  Saved run data: {path}  ({len(out)} arrays)")
    return sorted(out)


def load_run(path):
    """Load a .npz written by save_run into a plain dict."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
