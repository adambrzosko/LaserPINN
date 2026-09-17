"""
Fibre propagation workbench: a local GUI over fiber/grin_modes.py and fiber/gmmnlse.py.

Launch a beam into a fibre and get back what the solver computes -- mode profiles, output
spectrum, temporal envelope, per-mode power against distance, and the spontaneous-Raman
noise floor. Nothing here re-implements physics: every number comes from FibreModes and
GMMNLSE, and this module only marshals JSON in and arrays out.

Standard library only (the project depends on numpy/scipy/numba/matplotlib and nothing
else), so it runs wherever the package runs:

    python app/server.py [--port 8787]

then open http://localhost:8787.

Costs, measured on this machine, which set the interaction model: a mode solve is 0.02 s
for SMF-28, 0.24 s for OM3 (55 modes) and 0.95 s for OM1 (153 modes), so solves are cached
and the fibre panel responds immediately. A single-mode 10 km propagation on a 2048-point
grid is 0.08 s, so runs are synchronous -- except when noise is accumulated over every
guided mode, which is the expensive case (tens of seconds), and is therefore capped and
clearly labelled in the UI.
"""
import argparse
import json
import math
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fiber.constants import c, hbar
from fiber.gmmnlse import GMMNLSE, ModeCoupling, ModeLoss, TimeGrid
from fiber.grin_modes import DESIGNS, FibreDesign, FibreModes
from fiber.raman_models import BlowWood, LinAgrawal
from fiber.vector_modes import constituents, label as vector_label, radial_azimuthal, vector_field

HERE = Path(__file__).resolve().parent
MAX_POINTS = 1200          # downsample every trace to this before sending
MODE_GRID = 121            # 2D profile resolution (121x121 keeps the payload ~120 kB)
_cache = {}
_cache_lock = threading.Lock()


# ----------------------------------------------------------------- fibre + modes
def design_from(spec):
    """A FibreDesign from a preset name, or from explicit geometry for a custom fibre."""
    if spec.get('preset') in DESIGNS and not spec.get('custom'):
        return DESIGNS[spec['preset']]
    alpha = float(spec.get('alpha_profile', 2.05))
    return FibreDesign(
        name=spec.get('name', 'custom'),
        core_radius=float(spec['core_radius_um']) * 1e-6,
        NA=float(spec['NA']),
        alpha_profile=math.inf if alpha >= 50 else alpha,
        clad_radius=float(spec.get('clad_radius_um', 62.5)) * 1e-6,
        lambda_NA=float(spec.get('lambda_NA_nm', 850.0)) * 1e-9,
    )


def required_span_THz(n_points, dt_s):
    """Fit span the dispersion polynomial must cover for a given time grid.

    The grid reaches +-pi/dt in angular frequency, i.e. +-1/(2 dt) in Hz, independent of
    the point count. Returned with a 5% margin so rounding never trips the solver's own
    bounds check."""
    return 1.05 / (2 * dt_s) / 1e12


def solve_modes(spec, wavelength_nm, span_THz):
    """Cached mode solve. The key includes everything that changes the result."""
    design = design_from(spec)
    dr = 0.02e-6 if design.alpha_profile == math.inf else 0.05e-6
    key = (design.name, design.core_radius, design.NA, design.alpha_profile,
           design.clad_radius, design.lambda_NA, round(wavelength_nm, 6), round(span_THz, 6))
    with _cache_lock:
        if key in _cache:
            return _cache[key], design
    modes = FibreModes(design, wavelength_nm * 1e-9, span_Hz=span_THz * 1e12, dr=dr)
    with _cache_lock:
        _cache[key] = modes
    return modes, design


def beta_deriv_at(modes, p, order, omega):
    """n-th derivative of beta_p at arbitrary omega, taken ANALYTICALLY from the fitted
    polynomial. Finite-differencing the fit collapses by the third derivative -- it gave
    beta3 the wrong sign and 3400x the magnitude -- so differentiate the polynomial
    itself and rescale by the fit's x-scaling."""
    mode = modes.modes[p]
    fit = modes._fits[(mode.l, mode.m)]
    x = (np.asarray(omega, dtype=float) - modes.omega0) / modes._x_scale
    return fit.deriv(order)(x) / modes._x_scale ** order


def mode_field_diameters(modes, p):
    """MFD under the three conventions in use, with 1/e^2 reported only where it means
    something: a mode with a radial node (LP02, LP03) has no monotonic outward decay, and
    interpolating one anyway returns nonsense (it produced 0.05 um before this guard)."""
    r, R = modes.radial_profile(p)
    A = modes.effective_area(p)
    d_aeff = 2 * np.sqrt(A / np.pi)
    dRdr = np.gradient(R, r)
    num, den = np.trapezoid(R ** 2 * r, r), np.trapezoid(dRdr ** 2 * r, r)
    d_pet = 2 * np.sqrt(2 * num / den) if den > 0 else float('nan')
    I = R ** 2
    peak = int(np.argmax(I))
    outward = I[peak:]
    monotonic = bool(np.all(np.diff(outward) <= 1e-12 * I.max()))
    d_e2 = (2 * float(np.interp(I.max() / np.e ** 2, outward[::-1], r[peak:][::-1]))
            if monotonic else float('nan'))
    return d_aeff, d_pet, d_e2


def mode_table(modes, n2=2.6e-20):
    w0, k0 = modes.omega0, 2 * np.pi / modes.wavelength
    rows = []
    for p in range(len(modes)):
        cutoff = modes.cutoff_wavelength(p)
        A = modes.effective_area(p)
        d_aeff, d_pet, d_e2 = mode_field_diameters(modes, p)
        rows.append({
            'index': p,
            'label': modes.labels[p],
            'l': modes.modes[p].l,
            'm': modes.modes[p].m,
            'group': modes.principal_group(p),
            'beta': float(modes.beta0[p]),
            'n_eff': float(modes.beta0[p] / k0),
            'A_eff_um2': A * 1e12,
            'mfd_aeff_um': d_aeff * 1e6,
            'mfd_petermann_um': d_pet * 1e6,
            'mfd_1e2_um': None if np.isnan(d_e2) else d_e2 * 1e6,
            'gamma_W_km': 2 * np.pi * n2 / (modes.wavelength * A) * 1e3,
            'beta2_ps2_km': float(modes.beta_derivative(p, 2)) * 1e27,
            'beta3_ps3_km': float(modes.beta_derivative(p, 3)) * 1e39,
            'D_ps_nm_km': modes.dispersion_parameter(p),
            'dmd_ps_km': (modes.beta_derivative(p, 1) - modes.beta_derivative(0, 1)) * 1e15,
            'cladding_fraction': modes.cladding_power_fraction(p),
            'cutoff_nm': None if np.isnan(cutoff) else cutoff * 1e9,
            'vector': [vector_label(f, o, mm, i) for f, o, mm, i
                       in constituents(modes.modes[p].l, modes.modes[p].m)],
        })
    return rows


def dispersion_curves(modes, indices, n2=2.6e-20, n_points=160):
    """n_eff, beta2, beta3, D, A_eff and gamma across the fitted band, for each mode.

    Everything comes from the one mode solve already cached: beta(omega) is a fitted
    polynomial, so its derivatives are analytic, and A_eff(omega) follows the solver's own
    area_scale. The band is the fit's validity range -- quoting values outside it would be
    extrapolation dressed as physics."""
    half = modes.span * 0.98
    omega = modes.omega0 + np.linspace(-half, half, n_points)
    omega = omega[omega > 0]
    lam_nm = 2 * np.pi * c / omega * 1e9
    order = np.argsort(lam_nm)
    lam_nm, omega = lam_nm[order], omega[order]
    k0 = omega / c
    area0 = {p: modes.effective_area(p) for p in indices}
    scale = modes.area_scale(omega)                     # A_eff(w) = A_eff(w0)/scale
    out = {'wavelength_nm': downsample(lam_nm, n_points).tolist(), 'modes': []}
    for p in indices:
        beta = modes.beta(p, omega)
        b2 = beta_deriv_at(modes, p, 2, omega)
        b3 = beta_deriv_at(modes, p, 3, omega)
        A = area0[p] / scale
        guided = modes.is_guided(p, omega)
        def mask(v):
            return [None if not g else round(float(x), 9) for x, g in zip(np.asarray(v), guided)]
        out['modes'].append({
            'label': modes.labels[p],
            'n_eff': mask(beta / k0),
            'beta2_ps2_km': mask(b2 * 1e27),
            'beta3_ps3_km': mask(b3 * 1e39),
            'D_ps_nm_km': mask(-2 * np.pi * c * b2 / (2 * np.pi * c / omega) ** 2 * 1e6),
            'A_eff_um2': mask(A * 1e12),
            'gamma_W_km': mask(2 * np.pi * n2 / ((2 * np.pi * c / omega) * A) * 1e3),
        })
    return out


def profile_payload(modes, p, box_um=None, vector=None):
    """2D field of one mode for the profile viewer.

    Scalar (LP) view returns the signed field, from which the page renders either
    amplitude or intensity. `vector` selects a constituent instead -- 'TE', 'TM', 'HE' or
    'EH' plus an orientation index -- and then the payload carries both transverse
    components so the page can draw the polarisation map."""
    core_um = modes.design.core_radius * 1e6
    box = box_um or max(1.6 * core_um, 12.0)
    g = np.linspace(-box, box, MODE_GRID) * 1e-6
    X, Y = np.meshgrid(g, g)
    mode = modes.modes[p]
    r, R = modes.radial_profile(p)
    keep = r <= box * 1e-6
    d_aeff, d_pet, d_e2 = mode_field_diameters(modes, p)

    out = {
        'box_um': box,
        'core_um': core_um,
        'grid': MODE_GRID,
        'label': modes.labels[p],
        'mfd_aeff_um': d_aeff * 1e6,
        'mfd_petermann_um': d_pet * 1e6,
        'mfd_1e2_um': None if np.isnan(d_e2) else d_e2 * 1e6,
        'radial_r_um': downsample(r[keep] * 1e6, 400).tolist(),
        'radial_I': downsample((R[keep] ** 2) / float((R[keep] ** 2).max() or 1), 400).tolist(),
        'constituents': [vector_label(f, o, mm, i) for f, o, mm, i
                         in constituents(mode.l, mode.m)],
    }

    if not vector:
        field = modes.field(p, X, Y)
        scale = float(np.abs(field).max()) or 1.0
        out.update({'kind': 'scalar', 'field': np.round(field / scale, 4).ravel().tolist()})
        return out

    family = vector.get('family', 'HE')
    index = int(vector.get('index', 0))
    ex, ey = vector_field(modes, mode.l, mode.m, family, index, X, Y)
    amp = np.sqrt(ex ** 2 + ey ** 2)
    scale = float(amp.max()) or 1.0
    er, ep = radial_azimuthal(ex, ey, X, Y)
    dA = (g[1] - g[0]) ** 2
    step = max(1, MODE_GRID // 18)                 # quiver lattice, not every pixel
    qx, qy, qu, qv = [], [], [], []
    for i in range(step // 2, MODE_GRID, step):
        for j in range(step // 2, MODE_GRID, step):
            if amp[i, j] < 0.12 * scale:
                continue
            qx.append(round(float(g[j] * 1e6), 3)); qy.append(round(float(g[i] * 1e6), 3))
            qu.append(round(float(ex[i, j] / scale), 3)); qv.append(round(float(ey[i, j] / scale), 3))
    out.update({
        'kind': 'vector',
        'family': family, 'index': index,
        'vector_label': vector_label(family, mode.l - 1 if family == 'EH' else
                                     (0 if family in ('TE', 'TM') else mode.l + 1),
                                     mode.m, index),
        'field': np.round(amp / scale, 4).ravel().tolist(),          # |E| for the heat map
        'ex': np.round(ex / scale, 4).ravel().tolist(),
        'ey': np.round(ey / scale, 4).ravel().tolist(),
        'quiver': {'x': qx, 'y': qy, 'u': qu, 'v': qv},
        'radial_power_frac': float(np.sum(er ** 2) * dA),
        'azimuthal_power_frac': float(np.sum(ep ** 2) * dA),
    })
    return out


# ----------------------------------------------------------------- launch fields
def launch_field(grid, spec):
    """Build a launch envelope. TimeGrid provides only cw(); pulses are built here from
    grid.t so the library keeps no GUI-specific helpers."""
    power = float(spec.get('power_W', 1e-3))
    shape = spec.get('shape', 'cw')
    offset = float(spec.get('offset_GHz', 0.0)) * 1e9
    if shape == 'cw':
        return grid.cw(power, snap_offset(grid, offset))
    T0 = float(spec.get('width_ps', 10.0)) * 1e-12
    t = grid.t
    if shape == 'gaussian':
        env = np.exp(-t ** 2 / (2 * T0 ** 2))
    elif shape == 'sech':
        env = 1.0 / np.cosh(t / T0)
    else:
        raise ValueError(f'unknown pulse shape {shape!r}')
    field = np.sqrt(power) * env.astype(complex)
    if offset:
        field = field * (grid.cw(1.0, snap_offset(grid, offset)))
    return field


def snap_offset(grid, offset_Hz):
    """cw() requires the offset to land on a frequency bin."""
    return round(offset_Hz / grid.df) * grid.df


# ----------------------------------------------------------------- propagation
def run_propagation(req):
    wl = float(req['wavelength_nm'])
    n_points = int(req.get('n_points', 2048))
    dt = float(req.get('dt_fs', 50.0)) * 1e-15

    # The fit span must cover the time grid, so widen it rather than failing: the grid
    # reaches +-1/(2 dt), which the user sets through dt, not through this control.
    span = max(float(req.get('span_THz', 8.0)), required_span_THz(n_points, dt))
    modes, design = solve_modes(req['fibre'], wl, span)

    grid = TimeGrid(n_points, dt, wl * 1e-9)

    beams = [b for b in req.get('beams', []) if b.get('enabled', True)]
    if not beams:
        raise ValueError('add at least one beam')
    labels, A0 = [], []
    for b in beams:
        label = b.get('mode') or ''
        if not label:
            raise ValueError('a beam has no mode selected')
        if label not in modes.labels:
            raise ValueError(f'{label} is not guided in this fibre at {wl:.0f} nm — '
                             f'this fibre guides {modes.labels[0]}…{modes.labels[-1]} '
                             f'({len(modes)} modes)')
        labels.append(label)
        A0.append(launch_field(grid, b))

    length = float(req['length_km']) * 1e3
    dz = float(req.get('dz_m', 50.0))
    n_save = max(2, min(int(req.get('n_save', 12)), 40))
    z_save = np.linspace(length / n_save, length, n_save)

    noise = req.get('noise', 'none')
    ext = req.get('extensions', {})
    loss = float(req.get('alpha_dB_km', 0.3))
    if ext.get('mode_loss'):
        loss = ModeLoss(base_dB_km=loss,
                        dma_dB_km=float(ext.get('dma_dB_km', 0.0)),
                        bend_radius=(float(ext['bend_radius_mm']) * 1e-3
                                     if ext.get('bend_radius_mm') else None))

    coupling = None
    if ext.get('mode_coupling') and float(ext.get('kappa', 0.0)) > 0:
        coupling = ModeCoupling(kappa=float(ext['kappa']),
                                correlation_length=float(ext.get('L_c_mm', 1.0)) * 1e-3)

    # Noise over every guided mode is the expensive path; cap it so the UI stays usable.
    noise_modes = None
    if noise == 'mean':
        noise_modes = [labels[0]] if req.get('noise_first_mode_only', True) else None

    # Nonlinear term selection. 'phase-matched' is the solver default (0.1/dz: SPM, XPM,
    # intermodal Raman gain, degenerate-pair coupling); 'complete' admits the near-matched
    # intergroup four-wave-mixing terms too, at the cost of a much larger tensor.
    nl = req.get('nonlinear_terms', 'phase_matched')
    coherence_tol = {'phase_matched': None, 'complete': float('inf')}.get(nl, None)
    raman_model = (None if nl == 'kerr_only'
                   else BlowWood() if req.get('raman_model') == 'blow_wood' else LinAgrawal())
    if raman_model is None and noise == 'mean':
        raise ValueError("noise='mean' needs a Raman model; choose Kerr + Raman, or turn noise off")
    disp_order = req.get('dispersion_order')
    disp_order = None if disp_order in (None, '', 'full') else int(disp_order)

    gmm = GMMNLSE(
        modes, grid, labels,
        n2=float(req.get('n2', 2.6e-20)),
        raman=raman_model,
        alpha_dB_km=loss,
        coherence_tol=coherence_tol,
        dispersion_order=disp_order,
        self_steepening=bool(req.get('self_steepening', True)),
        noise=noise,
        temperature=float(req.get('temperature_K', 295.0)),
        noise_modes=noise_modes,
        mode_coupling=coupling,
        dispersive_overlaps=bool(ext.get('dispersive_overlaps', False)),
        vacuum_seed=bool(ext.get('vacuum_seed', False)) and noise == 'stochastic',
        backward=bool(ext.get('backward', False)) and noise == 'mean',
        seed=int(req.get('seed', 0)),
    )
    adaptive = float(ext['adaptive_tol']) if ext.get('adaptive') else None
    res = gmm.propagate(np.stack(A0), length, dz=dz, z_save=z_save, adaptive=adaptive)

    return pack_result(modes, grid, res, gmm, labels, design, wl)


def pack_result(modes, grid, res, gmm, labels, design, wl):
    # Sort by ascending WAVELENGTH, not ascending frequency: the spectrum plots against
    # wavelength, and an omega-ascending order would run the axis backwards.
    order = np.argsort(2 * np.pi * c / grid.omega)
    wavelength_nm = (2 * np.pi * c / grid.omega)[order] * 1e9
    psd = res.field_psd()[-1][:, order]                  # (K, N) at the output
    launch_psd = None

    t_ps = grid.t * 1e12
    power_t = np.abs(res.fields[-1]) ** 2

    # --- retarded frame -------------------------------------------------------------
    # The solver integrates in the co-moving frame of the reference mode: its beta0 and
    # beta1 are removed, so the time axis is retarded time T = t - beta1_ref * z. Every
    # other mode therefore DRIFTS across the window at its own walk-off, and in OM3 that
    # is fast: 87 ps/km for LP11 and 683 ps/km across the full 55-mode set, against a
    # window of ~100 ps. Report it, because a pulse that leaves the window wraps around
    # and a plot of it is worse than no plot.
    window_ps = float((grid.t[-1] - grid.t[0]) * 1e12)
    # gmm.ref indexes the PROPAGATED list, not the mode table: with propagate=['LP11a',
    # 'LP01'] and reference=0 it is 0, meaning LP11a, while modes.labels[0] is LP01.
    # Resolve through gmm.prop or both the frame label and every walk-off baseline are
    # taken from the wrong mode.
    ref_mode = gmm.prop[gmm.ref]
    b1_ref = modes.beta_derivative(ref_mode, 1)
    L_km = res.z[-1] / 1e3
    walk = []
    for row, p in enumerate(gmm.prop):
        dmd_ps_km = (modes.beta_derivative(p, 1) - b1_ref) * 1e15
        drift_ps = dmd_ps_km * L_km
        walk.append({
            'label': modes.labels[p],
            'dmd_ps_km': dmd_ps_km,
            'drift_ps': drift_ps,
            'wrapped': bool(abs(drift_ps) > window_ps / 2),
        })
    frame = {
        'reference_mode': modes.labels[ref_mode],
        'window_ps': window_ps,
        'walk_off': walk,
        'any_wrapped': any(w['wrapped'] for w in walk),
        'window_fills_at_km': (window_ps / max((abs(w['dmd_ps_km']) for w in walk), default=0.0)
                               if any(abs(w['dmd_ps_km']) > 0 for w in walk) else None),
    }

    # a time window that actually shows the pulse, rather than the whole period
    total = power_t.sum(axis=0)
    if total.max() > 0:
        centre = float((t_ps * total).sum() / total.sum())
        spread = float(np.sqrt((total * (t_ps - centre) ** 2).sum() / total.sum()))
        half = max(5 * spread, 8 * (t_ps[1] - t_ps[0]))
        tmask = np.abs(t_ps - centre) <= half
        if tmask.sum() < 32:
            tmask = np.ones_like(t_ps, dtype=bool)
    else:
        tmask = np.ones_like(t_ps, dtype=bool)

    # a wavelength window around the launch, so the plot is not mostly empty grid
    wmask = np.abs(wavelength_nm - wl) <= max(60.0, 0.02 * wl)
    if wmask.sum() < 64:
        wmask = np.ones_like(wavelength_nm, dtype=bool)

    out = {
        'fibre': {
            'name': design.name,
            'core_radius_um': design.core_radius * 1e6,
            'NA': design.NA,
            'alpha_profile': 'step' if design.alpha_profile == math.inf else design.alpha_profile,
            'n_modes': len(modes),
            'n_groups': len({modes.principal_group(p) for p in range(len(modes))}),
            'V': 2 * np.pi / (wl * 1e-9) * design.core_radius * design.NA,
        },
        'labels': labels,
        'span_THz': modes.span / 2 / np.pi / 1e12,
        'z_km': (res.z / 1e3).tolist(),
        'steps': int(getattr(gmm, 'n_steps', 0)),
        'rejected': int(getattr(gmm, 'n_rejected', 0)),
        # mean_power() is a window average, which for a short pulse in a long window is far
        # below the peak the user set -- send both so the UI can say which is which.
        'power_vs_z_mW': (res.mean_power() * 1e3).T.tolist(),
        'peak_vs_z_mW': (np.max(np.abs(res.fields) ** 2, axis=-1) * 1e3).T.tolist(),
        'window_ps': float((grid.t[-1] - grid.t[0]) * 1e12),
        'frame': frame,
        'solver': {
            'n_terms': int(getattr(gmm, 'n_terms', 0)),
            'raman': None if gmm.raman is None else gmm.raman.name,
            'coherence_tol': (None if gmm.coherence_tol is None
                              else ('inf' if np.isinf(gmm.coherence_tol) else gmm.coherence_tol)),
            'dispersion_order': gmm.dispersion_order,
            'self_steepening': bool(gmm.self_steepening),
        },
        'time': {
            't_ps': downsample(t_ps[tmask], MAX_POINTS).tolist(),
            'power_W': [downsample(row[tmask], MAX_POINTS).tolist() for row in power_t],
            'axis_label': f'retarded time T = t - z/v_g({modes.labels[ref_mode]})  (ps)',
        },
        'spectrum': {
            'wavelength_nm': downsample(wavelength_nm[wmask], MAX_POINTS).tolist(),
            'dBm_per_nm': [downsample(to_dbm_nm(row[wmask], wavelength_nm[wmask]),
                                      MAX_POINTS).tolist() for row in psd],
        },
    }

    if res.noise_psd is not None:
        nz = res.noise_psd[-1]
        out['noise'] = {
            'labels': list(res.noise_labels),
            'wavelength_nm': downsample(wavelength_nm[wmask], MAX_POINTS).tolist(),
            'forward_dBm_per_nm': [downsample(to_dbm_nm(row[order][wmask],
                                                        wavelength_nm[wmask]),
                                              MAX_POINTS).tolist() for row in nz],
        }
        if res.noise_psd_backward is not None:
            out['noise']['backward_dBm_per_nm'] = [
                downsample(to_dbm_nm(row[order][wmask], wavelength_nm[wmask]),
                           MAX_POINTS).tolist() for row in res.noise_psd_backward[-1]]
    return out


def to_dbm_nm(psd_W_per_Hz, wavelength_nm):
    """PSD in W/Hz -> dBm per nm, the way an OSA reads a smooth spectrum."""
    lam = np.asarray(wavelength_nm) * 1e-9
    per_nm = np.asarray(psd_W_per_Hz) * c * 1e-9 / lam ** 2
    return 10 * np.log10(np.maximum(per_nm, 1e-30) * 1e3)


def downsample(arr, n):
    arr = np.asarray(arr, dtype=float)
    if arr.size <= n:
        return np.round(arr, 6)
    idx = np.linspace(0, arr.size - 1, n).astype(int)
    return np.round(arr[idx], 6)


# ----------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, fmt, *args):
        pass                                   # quiet; errors still surface in responses

    def _send(self, code, body, ctype='application/json'):
        payload = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            html = (HERE / 'index.html').read_bytes()
            return self._send(200, html, 'text/html; charset=utf-8')
        if self.path == '/api/fibres':
            return self._send(200, {
                'presets': [{
                    'key': k,
                    'name': k.upper(),
                    'core_radius_um': d.core_radius * 1e6,
                    'NA': d.NA,
                    'alpha_profile': 'step' if d.alpha_profile == math.inf else d.alpha_profile,
                    'clad_radius_um': d.clad_radius * 1e6,
                    'lambda_NA_nm': d.lambda_NA * 1e9,
                } for k, d in DESIGNS.items()],
            })
        return self._send(404, {'error': 'not found'})

    def do_POST(self):
        try:
            n = int(self.headers.get('Content-Length', 0))
            req = json.loads(self.rfile.read(n) or b'{}')
            if self.path == '/api/modes':
                modes, design = solve_modes(req['fibre'], float(req['wavelength_nm']),
                                            float(req.get('span_THz', 8.0)))
                n2 = float(req.get('n2', 2.6e-20))
                body = {
                    'modes': mode_table(modes, n2=n2),
                    'n_modes': len(modes),
                    'n_groups': len({modes.principal_group(p) for p in range(len(modes))}),
                    'V': 2 * np.pi / (float(req['wavelength_nm']) * 1e-9)
                         * design.core_radius * design.NA,
                    'span_THz': modes.span / 2 / np.pi / 1e12,
                }
                if req.get('profile_index') is not None:
                    body['profile'] = profile_payload(modes, int(req['profile_index']),
                                                      vector=req.get('vector'))
                return self._send(200, body)
            if self.path == '/api/dispersion':
                modes, _ = solve_modes(req['fibre'], float(req['wavelength_nm']),
                                       float(req.get('span_THz', 8.0)))
                wanted = req.get('labels') or [modes.labels[0]]
                idx = [modes.index_of(l) for l in wanted if l in modes.labels]
                if not idx:
                    raise ValueError('none of the requested modes are guided here')
                return self._send(200, dispersion_curves(modes, idx,
                                                         n2=float(req.get('n2', 2.6e-20))))
            if self.path == '/api/propagate':
                return self._send(200, run_propagation(req))
            return self._send(404, {'error': 'not found'})
        except Exception as exc:                          # surface the real reason in the UI
            return self._send(400, {'error': f'{type(exc).__name__}: {exc}',
                                    'trace': traceback.format_exc(limit=3)})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--port', type=int, default=8787)
    ap.add_argument('--host', default='127.0.0.1')
    args = ap.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'Fibre workbench on http://{args.host}:{args.port}  (Ctrl-C to stop)')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nstopped')


if __name__ == '__main__':
    main()
