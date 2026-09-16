"""Validation of fiber.grin_modes against analytic and datasheet values.

    python tests/test_grin_modes.py
"""
from dataclasses import dataclass

import numpy as np

from fiber.grin_modes import DESIGNS, FibreDesign, FibreModes, _radial_solve, silica_index


@dataclass(frozen=True)
class _ExactParabola(FibreDesign):
    """n^2 = n1^2 (1 - 2 Delta r^2/a^2) with no cladding step: the 2D harmonic oscillator,
    for which k^2 n1^2 - beta^2 = 2 kappa G exactly, G = 2m + l - 1 and
    kappa = k n1 sqrt(2 Delta) / a."""
    n1: float = 1.46
    Delta: float = 0.009

    def index(self, r, wavelength):
        return np.sqrt(self.n1 ** 2 * (1 - 2 * self.Delta * (np.asarray(r) / self.core_radius) ** 2))


PARABOLA = _ExactParabola('parabola', core_radius=25e-6, NA=0.2, alpha_profile=2.0, clad_radius=60e-6)
LAM = 1.55e-6


def _parabola_errors(dr):
    k = 2 * np.pi / LAM
    kappa = k * PARABOLA.n1 * np.sqrt(2 * PARABOLA.Delta) / PARABOLA.core_radius
    errs = []
    for l, m in [(0, 1), (1, 1), (2, 1), (0, 2), (1, 2)]:
        _, beta, _ = _radial_solve(PARABOLA, LAM, l, dr)
        errs.append(abs(((k * PARABOLA.n1) ** 2 - beta[m - 1] ** 2) / (2 * kappa * (2 * m + l - 1)) - 1))
    return max(errs)


def check_silica_index():
    n = float(silica_index(1550e-9))
    assert abs(n - 1.44402) < 1e-5, n
    print(f'silica index OK: n(1550 nm) = {n:.5f}')


def check_parabola_eigenvalues():
    coarse, fine = _parabola_errors(0.1e-6), _parabola_errors(0.05e-6)
    assert fine < 3e-5, fine
    assert 3.0 < coarse / fine < 5.0, coarse / fine
    print(f'parabolic eigenvalues OK: max error {fine:.1e} at dr = 50 nm, '
          f'convergence ratio {coarse / fine:.2f} (2nd order -> 4)')


def check_parabola_effective_area():
    dr = 0.025e-6
    r, _, R = _radial_solve(PARABOLA, LAM, 0, dr)
    I = R[:, 0] ** 2
    A_eff = 2 * np.pi * (np.sum(I * r) * dr) ** 2 / (np.sum(I ** 2 * r) * dr)
    kappa = 2 * np.pi / LAM * PARABOLA.n1 * np.sqrt(2 * PARABOLA.Delta) / PARABOLA.core_radius
    err = abs(A_eff * kappa / (2 * np.pi) - 1)
    assert err < 1e-3, err
    print(f'parabolic LP01 effective area OK: {A_eff * 1e12:.2f} um^2 vs 2 pi/kappa, error {err:.1e}')


def check_smf28_datasheet():
    m = FibreModes(DESIGNS['smf28'], 1550e-9, span_Hz=15e12, dr=0.02e-6)
    A_eff, D = m.effective_area(0) * 1e12, m.dispersion_parameter(0)
    assert m.labels == ['LP01'], m.labels
    assert 82 < A_eff < 88, A_eff
    assert 16.0 < D < 17.5, D
    single_at_1270 = FibreModes(DESIGNS['smf28'], 1270e-9, span_Hz=2e12, dr=0.02e-6).labels == ['LP01']
    assert single_at_1270
    print(f'SMF-28 OK: single mode, A_eff = {A_eff:.1f} um^2, D = {D:.2f} ps/nm/km, '
          f'still single-mode at 1270 nm')


def check_om3_overlaps():
    m = FibreModes(DESIGNS['om3'], 1551.72e-9, span_Hz=5e12)
    i01, ia, ib = m.index_of('LP01'), m.index_of('LP11a'), m.index_of('LP11b')
    S = m.overlap_tensor([i01, ia, ib])
    assert m.beta0[ia] == m.beta0[ib]
    assert abs(S[1, 1, 2, 2] / S[1, 1, 1, 1] - 1 / 3) < 1e-10
    assert abs(S[0, 0, 0, 1]) < 1e-12 * S[0, 0, 0, 0]
    for perm in [(1, 0, 2, 3), (2, 1, 0, 3), (0, 3, 2, 1), (3, 2, 1, 0)]:
        assert np.allclose(S, S.transpose(perm))
    io = m.intensity_overlaps([i01, ia, ib], [i01, ia, ib])
    assert np.allclose(io, [[S[q, l, q, l] for l in range(3)] for q in range(3)])
    assert abs(m.effective_area(i01) * S[0, 0, 0, 0] - 1) < 1e-12
    print(f'OM3 overlaps OK: {len(m)} guided modes, A_eff(LP01) = {m.effective_area(i01) * 1e12:.1f} um^2, '
          f'S(01,11a,01,11a)/S(0000) = {S[0, 1, 0, 1] / S[0, 0, 0, 0]:.3f}, '
          f'LP11 a/b coherent ratio exactly 1/3')


def check_dispersion_fit():
    m = FibreModes(DESIGNS['om3'], 1551.72e-9, span_Hz=30e12)
    worst = max(abs(float(m.beta(p, m.omega0)) - m.beta0[p]) for p in range(12))
    assert worst < 0.1, worst
    print(f'dispersion fit OK: beta(omega0) reproduces the solver to {worst:.1e} rad/m')


def check_smf28_lp11_cutoff():
    """Step-index LP11 cuts off at V = 2.405. The guided mask must switch there, not at the
    nearest dispersion-scan frequency."""
    from scipy.optimize import brentq

    from fiber.constants import c

    design = DESIGNS['smf28']
    m = FibreModes(design, 1200e-9, span_Hz=20e12, dr=0.02e-6)
    p = m.index_of('LP11a')
    lam_c = m.cutoff_wavelength(p)

    def V(lam):
        n_cl = silica_index(lam)
        return 2 * np.pi / lam * design.core_radius * np.sqrt((n_cl + design.dn0) ** 2 - n_cl ** 2)

    expected = brentq(lambda lam: V(lam) - 2.4048, 1100e-9, 1400e-9)
    assert abs(lam_c / expected - 1) < 5e-3, (lam_c, expected)
    assert np.isnan(m.cutoff_wavelength(m.index_of('LP01')))
    assert m.is_guided(p, 2 * np.pi * c / (0.999 * lam_c))
    assert not m.is_guided(p, 2 * np.pi * c / (1.001 * lam_c))
    print(f'LP11 cutoff OK: {lam_c * 1e9:.1f} nm vs V = 2.405 at {expected * 1e9:.1f} nm')


def check_mode_fields():
    """The 2D field sampler: unit power over a box, orthogonal between distinct modes, and
    carrying the structure the LP label claims (2l azimuthal sign changes, m radial maxima)."""
    m = FibreModes(DESIGNS['om3'], LAM, span_Hz=5e12)
    g = np.linspace(-50e-6, 50e-6, 501)
    X, Y = np.meshgrid(g, g)
    dA = (g[1] - g[0]) ** 2
    for label in ['LP01', 'LP02', 'LP11a', 'LP21a']:
        psi = m.field(m.index_of(label), X, Y)
        assert abs(np.sum(psi ** 2) * dA - 1) < 5e-3, (label, np.sum(psi ** 2) * dA)
    a = m.field(m.index_of('LP11a'), X, Y)
    b = m.field(m.index_of('LP11b'), X, Y)
    assert abs(np.sum(a * b) * dA) < 1e-9, np.sum(a * b) * dA

    phi = np.linspace(0, 2 * np.pi, 1024, endpoint=False)
    for label, l in [('LP01', 0), ('LP11a', 1), ('LP21a', 2), ('LP31a', 3)]:
        p = m.index_of(label)
        r, R = m.radial_profile(p)
        r_peak = r[np.argmax(R ** 2)]
        ring = m.field(p, r_peak * np.cos(phi), r_peak * np.sin(phi))
        sign_changes = np.count_nonzero(np.diff(np.sign(ring), append=np.sign(ring[0])))
        assert sign_changes == 2 * l, (label, sign_changes)
    assert m.principal_group(m.index_of('LP01')) == 1
    assert m.principal_group(m.index_of('LP21a')) == 3 == m.principal_group(m.index_of('LP02'))
    print('mode fields OK: unit-power sampling, LP11a/LP11b orthogonal, 2l azimuthal sign '
          'changes for l = 0..3, group number 2m + l - 1')


if __name__ == '__main__':
    check_silica_index()
    check_parabola_eigenvalues()
    check_parabola_effective_area()
    check_smf28_datasheet()
    check_smf28_lp11_cutoff()
    check_om3_overlaps()
    check_dispersion_fit()
    check_mode_fields()
    print('\nAll fiber.grin_modes checks passed.')
