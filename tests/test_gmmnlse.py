"""Validation of fiber.gmmnlse against analytic limits and the single-mode solver.

    python tests/test_gmmnlse.py                   # all checks
    python tests/test_gmmnlse.py raman_gain ...    # selected checks
"""
import sys

import numpy as np

from fiber.constants import c, hbar
from fiber.fiber_params import FiberParams
from fiber.gmmnlse import GMMNLSE, TimeGrid
from fiber.grin_modes import DESIGNS, FibreModes
from fiber.propagator import FiberPropagator
from fiber.raman_models import BlowWood, LinAgrawal, gain_shape, spontaneous_shape

LAM = 1550e-9
N2 = 2.6e-20


def check_single_mode():
    """LP01 alone, beta2/beta3 only, Blow-Wood Raman, no self-steepening: must reproduce
    the validated split-step FiberPropagator (N = 2 soliton over one dispersion length)."""
    modes = FibreModes(DESIGNS['om3'], LAM, span_Hz=12e12, max_modes=1)
    grid = TimeGrid(2 ** 11, 50e-15, LAM)
    gamma = N2 * grid.omega0 / c / modes.effective_area(0)
    beta2, beta3 = modes.beta_derivative(0, 2), modes.beta_derivative(0, 3)
    T0 = 0.3e-12
    L = T0 ** 2 / abs(beta2)
    A0 = np.sqrt(4 / (gamma * L)) / np.cosh(grid.t / T0)

    out = GMMNLSE(modes, grid, ['LP01'], raman=BlowWood(), alpha_dB_km=0.2,
                  self_steepening=False, dispersion_order=3).propagate(A0[None, :], L, dz=L / 2000)
    fp = FiberParams(lambda0=LAM)
    fp.alpha = 0.2 / (10 * np.log10(np.e)) / 1e3
    fp.beta2, fp.beta3, fp.gamma = beta2, beta3, gamma
    ref = FiberPropagator(fp, include_raman=True).propagate(A0.astype(complex), grid.dt, L, n_steps=20000)

    err = np.linalg.norm(out.fields[-1, 0] - ref) / np.linalg.norm(ref)
    assert err < 2e-3, err
    print(f'single-mode reduction OK: relative field difference vs split-step = {err:.1e}')


def check_self_steepening():
    """Dispersionless Kerr with self-steepening: intensity obeys I_z + (3 gamma/omega0) I I_t = 0,
    so the peak is delayed by exactly 3 gamma P0 z / omega0."""
    modes = FibreModes(DESIGNS['om3'], LAM, span_Hz=12e12, max_modes=1)
    grid = TimeGrid(2 ** 12, 50e-15, LAM)
    gamma = N2 * grid.omega0 / c / modes.effective_area(0)
    P0, T0, z = 1.0 / gamma, 1e-12, 10.0
    A0 = np.sqrt(P0) * np.exp(-grid.t ** 2 / (2 * T0 ** 2))
    out = GMMNLSE(modes, grid, ['LP01'], raman=None, alpha_dB_km=0.0, self_steepening=True,
                  dispersion_order=1).propagate(A0[None, :], z, dz=0.02)
    I = np.abs(out.fields[-1, 0]) ** 2
    k = int(np.argmax(I))
    t_peak = grid.t[k] + 0.5 * grid.dt * (I[k - 1] - I[k + 1]) / (I[k - 1] - 2 * I[k] + I[k + 1])
    expected = 3 * gamma * P0 * z / grid.omega0
    assert abs(t_peak / expected - 1) < 0.02, (t_peak, expected)
    print(f'self-steepening OK: peak delay {t_peak * 1e15:.2f} fs vs analytic {expected * 1e15:.2f} fs')


def check_degenerate_coherent_term():
    """LP11a/LP11b CW, Kerr only: the circular combinations A+- = (Aa +- i Ab)/sqrt2 decouple,
    conserving |A+-|^2 with phase rates (2/3) g (|A+-|^2 + 2|A-+|^2). Fails if the coherent
    S_aabb term is missing, mis-scaled or has the wrong sign."""
    modes = FibreModes(DESIGNS['om3'], LAM, span_Hz=1e12, max_modes=3)
    grid = TimeGrid(64, 1e-12, LAM)
    g = N2 * grid.omega0 / c * modes.overlap_tensor([modes.index_of('LP11a')])[0, 0, 0, 0]
    P, theta, psi, z = 1.0 / g, 0.4, 0.9, 5.0
    Aa = np.full(64, np.sqrt(P) * np.cos(theta), complex)
    Ab = np.full(64, np.sqrt(P) * np.sin(theta) * np.exp(1j * psi), complex)
    out = GMMNLSE(modes, grid, ['LP11a', 'LP11b'], raman=None, alpha_dB_km=0.0,
                  self_steepening=False, dispersion_order=1,
                  reference='LP11a').propagate(np.stack([Aa, Ab]), z, dz=0.01).fields[-1]

    def circular(a, b):
        return (a + 1j * b) / np.sqrt(2), (a - 1j * b) / np.sqrt(2)

    p0, m0 = circular(Aa[0], Ab[0])
    p1, m1 = circular(out[0, 0], out[1, 0])
    power_err = max(abs(abs(p1) ** 2 / abs(p0) ** 2 - 1), abs(abs(m1) ** 2 / abs(m0) ** 2 - 1))
    phase_err = max(
        abs(np.angle(p1 / p0 * np.exp(-1j * (2 / 3) * g * (abs(p0) ** 2 + 2 * abs(m0) ** 2) * z))),
        abs(np.angle(m1 / m0 * np.exp(-1j * (2 / 3) * g * (abs(m0) ** 2 + 2 * abs(p0) ** 2) * z))))
    assert power_err < 1e-9, power_err
    assert phase_err < 1e-6, phase_err
    print(f'degenerate-group coherent coupling OK: |A+-|^2 conserved to {power_err:.1e}, '
          f'phase error {phase_err:.1e} rad after {g * P * z:.0f} rad of SPM')


def check_raman_gain():
    """CW pump in LP01, weak CW probe in LP11a at the 13.2 THz Stokes shift: probe power
    grows as exp[(n2 omega_s/c) S_(11a,01,11a,01) gain_shape P0 L]. The old intensity-overlap
    model gave zero here."""
    modes = FibreModes(DESIGNS['om3'], LAM, span_Hz=26e12, max_modes=3)
    grid = TimeGrid(2 ** 10, 20e-15, LAM)
    shift = round(13.2e12 / grid.df) * grid.df
    P0, L = 1.0, 1000.0
    A0 = np.stack([grid.cw(P0), grid.cw(1e-6, -shift)])
    out = GMMNLSE(modes, grid, ['LP01', 'LP11a'], raman=LinAgrawal(),
                  alpha_dB_km=0.0).propagate(A0, L, dz=10.0).fields[-1, 1]
    k = int(np.argmax(np.abs(np.fft.fft(A0[1]))))
    gain_num = np.log(np.abs(np.fft.fft(out)[k]) ** 2 / np.abs(np.fft.fft(A0[1])[k]) ** 2)
    S = modes.overlap_tensor([0, modes.index_of('LP11a')])[1, 0, 1, 0]
    omega_s = grid.omega0 - 2 * np.pi * shift
    gain_ana = N2 * omega_s / c * S * gain_shape(LinAgrawal(), 2 * np.pi * shift) * P0 * L
    assert abs(gain_num / gain_ana - 1) < 2e-3, (gain_num, gain_ana)
    print(f'intermodal Raman gain OK: ln G = {gain_num:.5f} vs analytic {gain_ana:.5f}')


def _noise_reference(grid, modes, receivers, P0, alpha_dB, L):
    alpha = alpha_dB / (10 * np.log10(np.e)) / 1e3
    S = modes.intensity_overlaps(receivers, [0])[:, 0]
    omega = grid.omega
    shape = spontaneous_shape(LinAgrawal(), grid.Omega, 300.0)
    return np.array([hbar * omega * (N2 * omega / c) * s * shape * P0 * np.exp(-alpha * L) * L for s in S])


def check_mean_noise():
    """Mean spontaneous-Raman PSD from a CW LP01 pump with loss equal in pump and noise:
    S_q = hbar omega g_q P0 exp(-alpha L) L (Stokes and anti-Stokes)."""
    modes = FibreModes(DESIGNS['om3'], LAM, span_Hz=26e12, max_modes=3)
    grid = TimeGrid(2 ** 11, 20e-15, LAM)
    P0, L, alpha_dB = 1e-3, 5000.0, 0.3
    res = GMMNLSE(modes, grid, ['LP01'], raman=LinAgrawal(), alpha_dB_km=alpha_dB, noise='mean',
                  noise_modes=['LP01', 'LP11a']).propagate(grid.cw(P0)[None, :], L, dz=50.0)
    ref = _noise_reference(grid, modes, [0, modes.index_of('LP11a')], P0, alpha_dB, L)
    band = (np.abs(grid.Omega) > 2 * np.pi * 1e12) & (np.abs(grid.Omega) < 2 * np.pi * 20e12)
    err = np.max(np.abs(res.noise_psd[-1][:, band] / ref[:, band] - 1))
    assert err < 2e-3, err
    print(f'mean spontaneous-Raman PSD OK: max deviation from analytic {err:.1e} over 1-20 THz, '
          f'both LP01 and LP11a')


def check_stochastic_noise():
    """Ensemble PSD of the Langevin noise must match the same analytic reference (this is
    the check the old frequency-domain recipe fails by a factor N)."""
    modes = FibreModes(DESIGNS['om3'], LAM, span_Hz=26e12, max_modes=3)
    grid = TimeGrid(2 ** 11, 20e-15, LAM)
    P0, L, alpha_dB, runs = 1e-3, 5000.0, 0.3, 8
    i11 = modes.index_of('LP11a')
    psd = np.zeros(grid.n_points)
    for run in range(runs):
        A0 = np.stack([grid.cw(P0), np.zeros(grid.n_points, complex)])
        out = GMMNLSE(modes, grid, ['LP01', 'LP11a'], pumps=['LP01'], raman=LinAgrawal(),
                      alpha_dB_km=alpha_dB, noise='stochastic',
                      seed=run).propagate(A0, L, dz=50.0).fields[-1, 1]
        psd += np.abs(np.fft.fft(out)) ** 2 * grid.dt / grid.n_points / runs
    ref = _noise_reference(grid, modes, [i11], P0, alpha_dB, L)[0]
    band = (grid.Omega > 2 * np.pi * 8e12) & (grid.Omega < 2 * np.pi * 18e12)
    ratio = psd[band].mean() / ref[band].mean()
    assert abs(ratio - 1) < 0.06, ratio
    print(f'stochastic Langevin noise OK: ensemble PSD / analytic = {ratio:.3f} '
          f'({runs} runs, {band.sum()} bins)')


def check_photon_number():
    """Lossless three-mode propagation with Kerr, Raman, self-steepening and the complete
    coherent tensor (coherence_tol = inf): photon number sum |A(omega)|^2/omega is conserved."""
    modes = FibreModes(DESIGNS['om3'], LAM, span_Hz=26e12, max_modes=3)
    grid = TimeGrid(2 ** 11, 20e-15, LAM)
    T0 = 0.2e-12
    A0 = np.stack([np.sqrt(P) / np.cosh((grid.t - t0) / T0)
                   for P, t0 in [(1e5, 0.0), (5e4, 0.3e-12), (2e4, -0.2e-12)]]).astype(complex)
    gmm = GMMNLSE(modes, grid, ['LP01', 'LP11a', 'LP11b'], raman=LinAgrawal(), alpha_dB_km=0.0,
                  coherence_tol=np.inf)
    out = gmm.propagate(A0, 0.02, dz=5e-5).fields[-1]

    def photons(F):
        return np.sum(np.abs(np.fft.fft(F, axis=-1)) ** 2 / grid.omega[None, :])

    drift = abs(photons(out) / photons(A0) - 1)
    spectral_change = np.linalg.norm(np.abs(np.fft.fft(out)) - np.abs(np.fft.fft(A0))) / np.linalg.norm(np.abs(np.fft.fft(A0)))
    assert drift < 1e-7, drift
    assert spectral_change > 1e-2, spectral_change
    print(f'photon number OK: drift {drift:.1e} with {gmm.n_terms} tensor terms '
          f'(spectrum changed by {spectral_change:.0%})')


CHECKS = {
    'single_mode': check_single_mode,
    'self_steepening': check_self_steepening,
    'degenerate': check_degenerate_coherent_term,
    'raman_gain': check_raman_gain,
    'mean_noise': check_mean_noise,
    'stochastic_noise': check_stochastic_noise,
    'photon_number': check_photon_number,
}

if __name__ == '__main__':
    for name in (sys.argv[1:] or CHECKS):
        CHECKS[name]()
    if not sys.argv[1:]:
        print('\nAll fiber.gmmnlse checks passed.')
