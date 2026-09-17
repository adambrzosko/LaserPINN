"""
Vector (HE / EH / TE / TM) constituents of the scalar LP modes.

fiber.grin_modes solves the scalar wave equation, which is the weakly-guiding limit: it
returns LP_lm modes, each a near-degenerate superposition of true vector modes of the
fibre. This module performs the standard decomposition in that same limit, so the field
PATTERNS and their polarisation structure are available without a full vector solver.

Construction
------------
Write the two scalar azimuthal solutions of LP_lm as

    psi_c = R(r) cos(l phi) / sqrt(pi)        psi_s = R(r) sin(l phi) / sqrt(pi)

(the normalisation FibreModes.field already applies). Combining them with the two
transverse polarisations gives the vector constituents, in Cartesian components
(E_x, E_y), each divided by sqrt(2) so that integral |E|^2 dA = 1:

    l = 0:  HE_1m           (psi, 0) and (0, psi)          -- the two polarisations
    l = 1:  TE_0m           (-psi_s,  psi_c)  / sqrt(2)    -- purely azimuthal
            TM_0m           ( psi_c,  psi_s)  / sqrt(2)    -- purely radial
            HE_2m           ( psi_c, -psi_s)  / sqrt(2), ( psi_s,  psi_c) / sqrt(2)
    l >= 2: EH_(l-1)m       ( psi_c,  psi_s)  / sqrt(2), ( psi_s, -psi_c) / sqrt(2)
            HE_(l+1)m       ( psi_c, -psi_s)  / sqrt(2), ( psi_s,  psi_c) / sqrt(2)

This reproduces the textbook table of LP constituents (Gloge, Appl. Opt. 10, 2252, 1971;
Snyder & Love, *Optical Waveguide Theory*): LP_0m -> HE_1m; LP_1m -> TE_0m, TM_0m, HE_2m;
LP_lm -> EH_(l-1)m, HE_(l+1)m.

What this does and does not give you
------------------------------------
Exact, within the weakly-guiding approximation: the transverse field patterns, their
polarisation maps, orthonormality, and the fact that TE_0m is purely azimuthal while
TM_0m is purely radial (both verified in tests/test_vector_modes.py).

NOT given: the small propagation-constant SPLITTING between constituents of one LP mode.
That splitting is a higher-order vector correction of order Delta (~0.01 here), and every
constituent returned by this module carries the beta of its parent LP mode. If you need
the splitting -- for example to model polarisation-mode coupling within a group -- a full
vector mode solver is required; this module is not a substitute for one.

    from fiber.vector_modes import constituents, vector_field, CONSTITUENT_TABLE
"""
import numpy as np

_SQRT2 = np.sqrt(2.0)


def constituents(l, m):
    """Vector constituents of LP_lm, as a list of (family, order, m, degeneracy_index).

    family is 'HE', 'EH', 'TE' or 'TM'; order is the azimuthal index of the vector mode
    (so HE_(l+1)m has order l+1). Two entries with the same (family, order, m) are the
    two orientations of a degenerate pair."""
    if l == 0:
        return [('HE', 1, m, 0), ('HE', 1, m, 1)]
    if l == 1:
        return [('TE', 0, m, 0), ('TM', 0, m, 0), ('HE', 2, m, 0), ('HE', 2, m, 1)]
    return [('EH', l - 1, m, 0), ('EH', l - 1, m, 1),
            ('HE', l + 1, m, 0), ('HE', l + 1, m, 1)]


def label(family, order, m, index=0):
    """'TE01', 'HE21a', 'EH11b' ... -- a/b suffixes only where the pair is degenerate."""
    suffix = '' if family in ('TE', 'TM') else ('a', 'b')[index]
    return f'{family}{order}{m}{suffix}'


def vector_field(modes, l, m, family, index=0, x=None, y=None):
    """Transverse field (E_x, E_y) of one vector constituent of LP_lm, sampled on x, y.

    Normalised to integral (|E_x|^2 + |E_y|^2) dA = 1. `modes` is a FibreModes instance
    that must already contain the parent LP mode."""
    if l == 0:
        psi = modes.field(modes.index_of(f'LP0{m}'), x, y)
        zero = np.zeros_like(psi)
        return (psi, zero) if index == 0 else (zero, psi)

    psi_c = modes.field(modes.index_of(f'LP{l}{m}a'), x, y)
    psi_s = modes.field(modes.index_of(f'LP{l}{m}b'), x, y)

    if family == 'TE':                      # azimuthal: E = R phi-hat
        ex, ey = -psi_s, psi_c
    elif family == 'TM':                    # radial: E = R r-hat
        ex, ey = psi_c, psi_s
    elif family == 'HE':
        ex, ey = (psi_c, -psi_s) if index == 0 else (psi_s, psi_c)
    elif family == 'EH':
        ex, ey = (psi_c, psi_s) if index == 0 else (psi_s, -psi_c)
    else:
        raise ValueError(f'unknown family {family!r}; expected HE, EH, TE or TM')
    return ex / _SQRT2, ey / _SQRT2


def radial_azimuthal(ex, ey, x, y):
    """Project a Cartesian transverse field onto (E_r, E_phi). TE modes have E_r = 0 and
    TM modes E_phi = 0, which is the cleanest check that a construction is right."""
    phi = np.arctan2(y, x)
    cos, sin = np.cos(phi), np.sin(phi)
    return ex * cos + ey * sin, -ex * sin + ey * cos


def CONSTITUENT_TABLE(modes, max_group=None):
    """The LP -> vector table for every LP mode in `modes`, in the form the textbook
    table is written: one row per LP_lm with its constituent vector-mode names."""
    rows, seen = [], set()
    for p in range(len(modes)):
        mode = modes.modes[p]
        key = (mode.l, mode.m)
        if key in seen:
            continue
        seen.add(key)
        group = modes.principal_group(p)
        if max_group is not None and group > max_group:
            continue
        names, ordered = [], constituents(mode.l, mode.m)
        for family, order, mm, idx in ordered:
            nm = f'{family}_{order}{mm}'
            if nm not in names:
                names.append(nm)
        rows.append({
            'lp': f'LP{mode.l}{mode.m}',
            'l': mode.l, 'm': mode.m, 'group': group,
            'constituents': names,
            'n_states': len(ordered),
            'beta': float(modes.beta0[p]),
        })
    return sorted(rows, key=lambda r: (r['group'], r['l'], r['m']))
