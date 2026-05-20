"""
gsdfb — Gain-Switched DFB Laser Simulation Toolkit.

Shared utilities for the simulation suite. Import commonly used functions:

    from gsdfb import compute_r1, compute_metrics, setup_plotting, save_fig
    from gsdfb.pinn_utils import Scales, physics_residuals, add_noise_snr
"""
from gsdfb.analysis import (
    compute_r1,
    compute_metrics,
    phase_randomisation_quality,
    absolute_jitter,
    period_jitter,
    allan_deviation,
    amzi_outputs,
    amzi_splitting_ratio,
)
from gsdfb.plotting import setup_plotting, save_fig, STYLE
