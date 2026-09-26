"""
Shared plotting utilities for gain-switched DFB laser simulations.

Provides consistent matplotlib configuration and figure saving across
all analysis scripts. Replaces 24 copies of matplotlib.use('Agg') +
style boilerplate.
"""
import os
import sys
import matplotlib
# Scripts only write files, so they use the non-interactive Agg backend and
# run headless. Inside IPython or Jupyter the session's backend is left alone;
# forcing Agg there would stop figures displaying inline in a notebook.
if 'IPython' not in sys.modules:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── Style dictionary ─────────────────────────────────────────────────────────

STYLE = {
    'font.family': 'DejaVu Serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
}


def setup_plotting():
    """Apply consistent matplotlib style. Call once at script start."""
    plt.rcParams.update(STYLE)


def save_fig(fig, path, close=True, params=None, manifest=True, **savefig_kwargs):
    """Save a figure and, beside it, a record of what produced it.

    The sidecar <path>.json holds the script and command, the parameters passed here, the
    upper-case constants and laser of the calling module, library versions and a fingerprint of
    the project source that was imported (see gsdfb.provenance). Without it a figure on disk
    carries nothing but its timestamp, which is how images/ came to hold figures whose
    parameters nobody could state. The run browser at /runs reads these.

    Parameters
    ----------
    fig : matplotlib Figure
    path : str — output file path (e.g. 'images/multimode/fig1.png')
    close : bool — close figure after saving (default True)
    params : dict — the parameters that matter for THIS figure; recorded verbatim
    manifest : bool — set False for throwaway figures
    **savefig_kwargs — passed to fig.savefig (dpi, bbox_inches, ...)
    """
    path = str(path)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    savefig_kwargs.setdefault('bbox_inches', 'tight')
    savefig_kwargs.setdefault('pad_inches', 0.1)
    fig.savefig(path, **savefig_kwargs)
    if close:
        plt.close(fig)
    if manifest:
        try:
            from gsdfb.provenance import write_manifest
            import sys
            frame = sys._getframe(1)
            # locals as well as globals: a script that keeps its configuration inside main() has
            # nothing at module level to capture, and recorded an empty manifest
            caller = dict(frame.f_globals)
            caller.update({k: v for k, v in frame.f_locals.items()
                           if k.isupper() or 'laser' in k.lower()})
            write_manifest(path, params=params, namespace=caller)
        except Exception as exc:                  # a figure must never be lost to its own bookkeeping
            print(f"  (provenance not recorded for {path}: {type(exc).__name__}: {exc})")
    print(f"  Saved: {path}")


def img_dir(name):
    """Return the images subdirectory for a given analysis, creating it.

    Usage: out = img_dir('multimode')  # returns 'images/multimode'
    """
    d = os.path.join('images', name)
    os.makedirs(d, exist_ok=True)
    return d
