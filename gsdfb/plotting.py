"""
Shared plotting utilities for gain-switched DFB laser simulations.

Provides consistent matplotlib configuration and figure saving across
all analysis scripts. Replaces 24 copies of matplotlib.use('Agg') +
style boilerplate.
"""
import os
import matplotlib
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


def save_fig(fig, path, close=True):
    """Save figure to path, creating directories as needed.

    Parameters
    ----------
    fig : matplotlib Figure
    path : str — output file path (e.g. 'images/multimode/fig1.png')
    close : bool — close figure after saving (default True)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches='tight', pad_inches=0.1)
    if close:
        plt.close(fig)
    print(f"  Saved: {path}")


def img_dir(name):
    """Return the images subdirectory for a given analysis, creating it.

    Usage: out = img_dir('multimode')  # returns 'images/multimode'
    """
    d = os.path.join('images', name)
    os.makedirs(d, exist_ok=True)
    return d
