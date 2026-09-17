"""Validation of fiber.vector_modes against the defining properties of the vector modes.

    python tests/test_vector_modes.py
"""
import sys

import numpy as np

from fiber.grin_modes import DESIGNS, FibreModes
from fiber.vector_modes import (CONSTITUENT_TABLE, constituents, label,
                                radial_azimuthal, vector_field)

LAM = 1550e-9
_MODES = FibreModes(DESIGNS['om3'], LAM, span_Hz=5e12)
_G = np.linspace(-45e-6, 45e-6, 221)
_X, _Y = np.meshgrid(_G, _G)
_DA = (_G[1] - _G[0]) ** 2


def _norm(F):
    return float(np.sum(F[0] ** 2 + F[1] ** 2) * _DA)


def _dot(F, G):
    return float(np.sum(F[0] * G[0] + F[1] * G[1]) * _DA)


def check_normalisation():
    """Every constituent carries unit power over the sampling box."""
    worst = 0.0
    for (l, m) in [(0, 1), (1, 1), (2, 1), (0, 2), (3, 1)]:
        for family, order, mm, idx in constituents(l, m):
            F = vector_field(_MODES, l, m, family, idx, _X, _Y)
            worst = max(worst, abs(_norm(F) - 1))
    assert worst < 5e-3, worst
    print(f'vector normalisation OK: every constituent unit power to {worst:.1e}')


def check_orthogonality():
    """The constituents of one LP mode form an orthogonal set -- they are a basis for the
    same degenerate subspace, not overlapping copies of it."""
    worst = 0.0
    for (l, m) in [(1, 1), (2, 1), (3, 1), (1, 2)]:
        fields = [vector_field(_MODES, l, m, f, i, _X, _Y)
                  for f, o, mm, i in constituents(l, m)]
        for a in range(len(fields)):
            for b in range(a + 1, len(fields)):
                worst = max(worst, abs(_dot(fields[a], fields[b])))
    assert worst < 1e-6, worst
    print(f'vector orthogonality OK: largest cross term {worst:.1e} across LP11, LP21, LP31, LP12')


def check_te_tm_polarisation():
    """The defining property: TE_0m is purely azimuthal, TM_0m purely radial. This is what
    distinguishes a correct construction from a plausible-looking wrong one."""
    for m in (1, 2):
        te = vector_field(_MODES, 1, m, 'TE', 0, _X, _Y)
        tm = vector_field(_MODES, 1, m, 'TM', 0, _X, _Y)
        te_r, te_p = radial_azimuthal(*te, _X, _Y)
        tm_r, tm_p = radial_azimuthal(*tm, _X, _Y)
        te_radial = float(np.sum(te_r ** 2) * _DA)
        tm_azim = float(np.sum(tm_p ** 2) * _DA)
        assert te_radial < 1e-9, (m, te_radial)
        assert tm_azim < 1e-9, (m, tm_azim)
        assert abs(float(np.sum(te_p ** 2) * _DA) - 1) < 5e-3
        assert abs(float(np.sum(tm_r ** 2) * _DA) - 1) < 5e-3
    print('TE/TM polarisation OK: TE azimuthal and TM radial to <1e-9 of stray component')


def check_textbook_table():
    """Reproduce the standard LP -> constituent table (Gloge 1971; the table quoted in the
    multimode-fibre chapter): LP0m -> HE1m; LP1m -> TE0m, TM0m, HE2m; LPlm -> EH(l-1)m,
    HE(l+1)m."""
    expected = {
        'LP01': ['HE_11'], 'LP02': ['HE_12'], 'LP03': ['HE_13'],
        'LP11': ['TE_01', 'TM_01', 'HE_21'],
        'LP12': ['TE_02', 'TM_02', 'HE_22'],
        'LP21': ['EH_11', 'HE_31'],
        'LP22': ['EH_12', 'HE_32'],
    }
    table = {r['lp']: r['constituents'] for r in CONSTITUENT_TABLE(_MODES)}
    for lp, want in expected.items():
        assert lp in table, f'{lp} missing from the computed table'
        assert table[lp] == want, (lp, table[lp], want)
    # degeneracy count: l=0 gives 2 states, l>=1 gives 4
    rows = {r['lp']: r['n_states'] for r in CONSTITUENT_TABLE(_MODES)}
    assert rows['LP01'] == 2 and rows['LP11'] == 4 and rows['LP21'] == 4
    print(f'textbook table OK: {len(expected)} LP modes reproduce their published '
          f'constituents, with 2-fold (l=0) and 4-fold (l>=1) degeneracy')


def check_labels_and_beta():
    """Labels are well formed, and every constituent inherits its parent LP beta -- the
    module must not invent a splitting it cannot compute."""
    assert label('TE', 0, 1) == 'TE01'
    assert label('HE', 2, 1, 0) == 'HE21a' and label('HE', 2, 1, 1) == 'HE21b'
    rows = {r['lp']: r for r in CONSTITUENT_TABLE(_MODES)}
    p = _MODES.index_of('LP11a')
    assert abs(rows['LP11']['beta'] - _MODES.beta0[p]) < 1e-9
    print('labels and beta inheritance OK: constituents carry the parent LP propagation '
          'constant, no fabricated splitting')


CHECKS = {
    'normalisation': check_normalisation,
    'orthogonality': check_orthogonality,
    'te_tm': check_te_tm_polarisation,
    'table': check_textbook_table,
    'labels': check_labels_and_beta,
}

if __name__ == '__main__':
    for name in (sys.argv[1:] or CHECKS):
        CHECKS[name]()
    if not sys.argv[1:]:
        print('\nAll fiber.vector_modes checks passed.')
