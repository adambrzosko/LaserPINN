"""Validation of fiber.gmmnlse against analytic limits and the single-mode solver.

    python tests/test_gmmnlse.py                   # all checks
    python tests/test_gmmnlse.py raman_gain ...    # selected checks
"""
import sys

import numpy as np

from fiber.constants import c, hbar
from fiber.fiber_params import FiberParams
from fiber.gmmnlse import GMMNLSE, ModeCoupling, ModeLoss, TimeGrid
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
    shape = spontaneous_shape(LinAgrawal(), grid.Omega, 295.0)   # the GMMNLSE default
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


def check_mode_coupling():
    """Random linear mode coupling: exactly unitary (total power conserved to machine
    precision), diffusive (coupled power grows as kappa^2 w z), and suppressed between
    non-degenerate modes by the Lorentzian w = 1/(1 + (dbeta0 L_c)^2)."""
    modes = FibreModes(DESIGNS['om3'], LAM, span_Hz=1e12, max_modes=3)
    grid = TimeGrid(64, 1e-12, LAM)
    i11a = modes.index_of('LP11a')
    dbeta = abs(modes.beta0[0] - modes.beta0[i11a])
    L_c = 1.0 / dbeta                      # by construction (dbeta L_c) = 1, so w = 1/2
    w_cross = 1.0 / (1.0 + (dbeta * L_c) ** 2)
    kappa = 3e-3

    # (1) exactly unitary: total power over the propagated modes is conserved
    A0 = np.stack([grid.cw(P) for P in (1.0, 0.3, 0.1)])
    strong = GMMNLSE(modes, grid, ['LP01', 'LP11a', 'LP11b'], raman=None, n2=0.0,
                     alpha_dB_km=0.0, dispersion_order=1,
                     mode_coupling=ModeCoupling(kappa=0.05, correlation_length=L_c),
                     seed=0).propagate(A0, 50.0, dz=0.5).fields[-1]
    drift = abs(np.sum(np.abs(strong) ** 2) / np.sum(np.abs(A0) ** 2) - 1)
    assert drift < 1e-12, drift

    # (2) per-step transfer, straight from the coupling operator. |a|^2 after a random
    # walk is exponentially distributed, so a propagation-level ensemble converges as
    # 1/sqrt(runs); sampling the step itself gives the same expectation far more cheaply.
    def per_step_transfer(labels, seed, draws=5000, h=1.0):
        gmm = GMMNLSE(modes, grid, labels, raman=None, n2=0.0, alpha_dB_km=0.0,
                      dispersion_order=1,
                      mode_coupling=ModeCoupling(kappa=kappa, correlation_length=L_c), seed=seed)
        gmm.propagate(np.zeros((2, grid.n_points), complex), h, dz=h)   # sets up _cpl_sigma
        e = np.array([1.0, 0.0], complex)
        return np.mean([abs(gmm._coupling_step(e, h)[1]) ** 2 for _ in range(draws)])

    cross = per_step_transfer(['LP01', 'LP11a'], 1)
    degen = per_step_transfer(['LP11a', 'LP11b'], 1)
    assert abs(cross / (kappa ** 2 * w_cross) - 1) < 0.06, cross
    assert abs(degen / kappa ** 2 - 1) < 0.06, degen
    # same seed -> same random draws, so their ratio isolates the Lorentzian weight
    assert abs((cross / degen) / w_cross - 1) < 1e-3, cross / degen

    # (3) coupled power accumulates linearly with distance (loose: 64-run ensemble)
    z, runs = 200.0, 64
    halves, ends = [], []
    for seed in range(runs):
        A = np.zeros((2, grid.n_points), complex)
        A[0] = grid.cw(1.0)
        out = GMMNLSE(modes, grid, ['LP01', 'LP11a'], raman=None, n2=0.0, alpha_dB_km=0.0,
                      dispersion_order=1,
                      mode_coupling=ModeCoupling(kappa=kappa, correlation_length=L_c),
                      seed=seed).propagate(A, z, dz=5.0, z_save=[z / 2])
        p = np.mean(np.abs(out.fields) ** 2, axis=-1)
        halves.append(p[0, 1] / p[0].sum())
        ends.append(p[1, 1] / p[1].sum())
    grown = np.mean(ends) / np.mean(halves)
    assert abs(np.mean(ends) / (kappa ** 2 * w_cross * z) - 1) < 0.3, np.mean(ends)
    assert abs(grown / 2 - 1) < 0.35, grown
    print(f'mode coupling OK: unitary to {drift:.1e}; per-step transfer {cross:.3e} vs analytic '
          f'{kappa ** 2 * w_cross:.3e} (cross-group) and {degen:.3e} vs {kappa ** 2:.3e} '
          f'(degenerate); weight ratio exact to {abs((cross / degen) / w_cross - 1):.1e}; '
          f'power grows x{grown:.2f} over 2x distance')


def check_adaptive_stepping():
    """Step-doubling error control reaches a fine fixed-step reference to well within the
    requested tolerance while taking an order of magnitude fewer steps, and beats a fixed
    run at the same (maximum) step size."""
    modes = FibreModes(DESIGNS['om3'], LAM, span_Hz=12e12, max_modes=1)
    grid = TimeGrid(2 ** 11, 50e-15, LAM)
    gamma = N2 * grid.omega0 / c / modes.effective_area(0)
    beta2 = modes.beta_derivative(0, 2)
    T0 = 0.3e-12
    L = T0 ** 2 / abs(beta2)
    A0 = (np.sqrt(4 / (gamma * L)) / np.cosh(grid.t / T0))[None, :]   # N = 2 soliton

    gmm = GMMNLSE(modes, grid, ['LP01'], raman=None, alpha_dB_km=0.0, dispersion_order=3)
    ref = gmm.propagate(A0, L, dz=L / 5000).fields[-1, 0]
    n_ref = gmm.n_steps
    adapt = gmm.propagate(A0, L, dz=L / 50, adaptive=1e-9)
    n_adapt, n_rej = gmm.n_steps, gmm.n_rejected
    coarse = gmm.propagate(A0, L, dz=L / 50).fields[-1, 0]

    err_adapt = np.linalg.norm(adapt.fields[-1, 0] - ref) / np.linalg.norm(ref)
    err_fixed = np.linalg.norm(coarse - ref) / np.linalg.norm(ref)
    assert err_adapt < 1e-5, err_adapt
    assert n_adapt < n_ref / 10, (n_adapt, n_ref)
    assert err_fixed > 10 * err_adapt, (err_fixed, err_adapt)
    print(f'adaptive stepping OK: {n_adapt} steps ({n_rej} rejected) reach {err_adapt:.1e} vs the '
          f'{n_ref}-step reference, where {n_ref // 100} fixed steps of the same maximum size '
          f'give {err_fixed:.1e}')


def check_mode_loss():
    """ModeLoss: reduces to the scalar loss for a straight fibre with no DMA; differential
    mode attenuation follows the cladding power fraction, which grows with mode order; a
    bend strips the weakly guided high-order modes first and tightens as the radius falls;
    and the propagated powers follow exp(-alpha_p L) mode by mode."""
    modes = FibreModes(DESIGNS['om3'], LAM, span_Hz=5e12)
    grid = TimeGrid(64, 1e-12, LAM)
    low, high = 'LP01', modes.labels[-1]
    i_low, i_high = modes.index_of(low), modes.index_of(high)

    plain = GMMNLSE(modes, grid, [low, high], alpha_dB_km=ModeLoss(base_dB_km=0.3))
    scalar = 0.3 / (10 * np.log10(np.e)) / 1e3
    assert all(abs(plain._alpha(p) / scalar - 1) < 1e-12 for p in plain.prop)

    f_low, f_high = modes.cladding_power_fraction(i_low), modes.cladding_power_fraction(i_high)
    assert f_high > f_low, (f_low, f_high)
    assert modes.guidance_margin(i_high) < modes.guidance_margin(i_low)

    dma = ModeLoss(base_dB_km=0.3, dma_dB_km=5.0)
    assert dma.dB_km(modes, i_high) > dma.dB_km(modes, i_low) > 0.3

    tight, loose = ModeLoss(bend_radius=5e-3), ModeLoss(bend_radius=20e-3)
    assert tight.dB_km(modes, i_high) > tight.dB_km(modes, i_low)
    assert tight.dB_km(modes, i_high) > loose.dB_km(modes, i_high)
    assert abs(loose.dB_km(modes, i_low) / 0.3 - 1) < 1e-6, loose.dB_km(modes, i_low)

    L = 200.0
    gmm = GMMNLSE(modes, grid, [low, high], raman=None, n2=0.0, alpha_dB_km=dma,
                  dispersion_order=1)
    out = gmm.propagate(np.stack([grid.cw(1.0), grid.cw(1.0)]), L, dz=10.0)
    for row, p in enumerate(gmm.prop):
        expected = np.exp(-gmm._alpha(p) * L)
        assert abs(out.mean_power()[-1, row] / expected - 1) < 1e-9
    print(f'mode loss OK: cladding fraction {f_low:.4f} ({low}) -> {f_high:.4f} ({high}); '
          f'DMA {dma.dB_km(modes, i_low):.3f} -> {dma.dB_km(modes, i_high):.3f} dB/km; '
          f'5 mm bend {tight.dB_km(modes, i_low):.3f} -> {tight.dB_km(modes, i_high):.1f} dB/km; '
          f'propagated powers match exp(-alpha_p L)')


def check_dispersive_overlaps():
    """A_eff(omega) taken from the dispersion scan reproduces independently solved modes
    across the C+L band; area_scale is exactly 1 at the design frequency; and switching
    dispersive_overlaps on changes the spontaneous-Raman PSD by exactly that factor."""
    modes = FibreModes(DESIGNS['om3'], LAM, span_Hz=26e12)
    p = modes.index_of('LP01')
    assert abs(modes.area_scale(modes.omega0) - 1) < 1e-12

    for lam in (1450e-9, 1650e-9, 1750e-9):
        direct = FibreModes(DESIGNS['om3'], lam, span_Hz=2e12)
        scaled = modes.effective_area(p, omega=2 * np.pi * c / lam)
        rel = abs(scaled / direct.effective_area(direct.index_of('LP01')) - 1)
        assert rel < 0.01, (lam, rel)
    # modes shrink as frequency rises, so the scale rises above 1 on the anti-Stokes side
    assert modes.area_scale(modes.omega0 * 1.01) > 1 > modes.area_scale(modes.omega0 * 0.99)

    grid = TimeGrid(2 ** 11, 20e-15, LAM)
    P0, L = 1e-3, 2000.0
    psd = {}
    for flag in (False, True):
        res = GMMNLSE(modes, grid, ['LP01'], raman=LinAgrawal(), alpha_dB_km=0.0,
                      noise='mean', noise_modes=['LP01'], dispersive_overlaps=flag,
                      ).propagate(grid.cw(P0)[None, :], L, dz=100.0)
        psd[flag] = res.noise_psd[-1, 0]
    k = int(np.argmin(np.abs(grid.Omega - 2 * np.pi * 13.2e12)))
    ratio = psd[True][k] / psd[False][k]
    expected = float(modes.area_scale(grid.omega[k]))
    assert abs(ratio / expected - 1) < 2e-3, (ratio, expected)
    print(f'dispersive overlaps OK: A_eff(omega) matches direct solves to <1% over '
          f'1450-1750 nm; Stokes-peak PSD scales by {ratio:.4f} vs predicted {expected:.4f} '
          f'({100 * (expected - 1):+.1f}% at a 13.2 THz shift)')


def check_vacuum_seed():
    """The vacuum seed carries exactly half a photon per bin per mode, is rejected without
    stochastic noise, and lets a strong pump generate spontaneous four-wave mixing in bins
    where nothing was launched -- which a seedless run cannot do."""
    modes = FibreModes(DESIGNS['om3'], LAM, span_Hz=12e12, max_modes=1)
    grid = TimeGrid(2 ** 10, 50e-15, LAM)

    gmm = GMMNLSE(modes, grid, ['LP01'], raman=None, alpha_dB_km=0.0, noise='stochastic',
                  vacuum_seed=True, seed=0)
    seed_f = gmm._vacuum_seed()
    # PSD = |Af|^2 dt/N and photons = PSD df/(hbar omega), so photons = |Af|^2/(N^2 hbar omega)
    photons = np.abs(seed_f[0]) ** 2 / (grid.n_points ** 2 * hbar * grid.omega)
    guided = modes.is_guided(0, grid.omega)
    assert np.allclose(photons[guided], 0.5, rtol=1e-9), photons[guided][:3]

    try:
        GMMNLSE(modes, grid, ['LP01'], noise='mean', vacuum_seed=True)
    except ValueError:
        pass
    else:
        raise AssertionError('vacuum_seed must require stochastic noise')

    # Spontaneous FWM, isolated: a CW pump has constant intensity, so SPM is pure phase
    # rotation and creates no new frequencies at all. Any sideband is therefore seeded by
    # the vacuum, amplified by modulation instability (beta2 < 0 here), and must peak near
    # the analytic Omega_max = sqrt(2 gamma P / |beta2|).
    gamma = N2 * grid.omega0 / c / modes.effective_area(0)
    beta2 = modes.beta_derivative(0, 2)
    P0, L = 500.0, 20.0
    A0 = grid.cw(P0)[None, :]

    def spectrum(seed=None):
        kw = dict(raman=None, alpha_dB_km=0.0)
        if seed is not None:
            kw.update(noise='stochastic', vacuum_seed=True, seed=seed)
        out = GMMNLSE(modes, grid, ['LP01'], **kw).propagate(A0, L, dz=0.005).fields[-1, 0]
        return np.abs(np.fft.fft(out)) ** 2

    bare = spectrum()
    seeded = np.mean([spectrum(s) for s in range(4)], axis=0)
    f = np.fft.fftfreq(grid.n_points, d=grid.dt)
    side = (np.abs(f) > 0.1e12) & (np.abs(f) < 4e12)
    assert bare[side].max() < 1e-20 * P0, bare[side].max()      # no FWM without a seed
    assert seeded[side].mean() > 1e6 * max(bare[side].mean(), 1e-300)
    f_peak = abs(f[int(np.argmax(np.where(side, seeded, 0.0)))])
    f_mi = np.sqrt(2 * gamma * P0 / abs(beta2)) / (2 * np.pi)
    assert abs(f_peak / f_mi - 1) < 0.15, (f_peak, f_mi)
    print(f'vacuum seed OK: {photons[guided].mean():.4f} photons/bin (half a photon), rejected '
          f'without stochastic noise; a CW pump gives exactly zero sidebands unseeded and '
          f'{seeded[side].mean():.2e} seeded, peaking at {f_peak / 1e12:.3f} THz vs the analytic '
          f'MI frequency {f_mi / 1e12:.3f} THz')


def check_backward_noise():
    """Counter-propagating spontaneous Raman: equal to the forward PSD when alpha L << 1,
    saturating as (1-exp(-2 alpha L))/(2 alpha) while the forward term turns over at
    L = 1/alpha, so the two diverge with length exactly as the analytic forms require."""
    modes = FibreModes(DESIGNS['om3'], LAM, span_Hz=26e12, max_modes=3)
    grid = TimeGrid(2 ** 11, 20e-15, LAM)
    P0, alpha_dB = 1e-3, 0.3
    alpha = alpha_dB / (10 * np.log10(np.e)) / 1e3
    lengths = np.array([1e3, 5e3, 14e3, 50e3])

    res = GMMNLSE(modes, grid, ['LP01'], raman=LinAgrawal(), alpha_dB_km=alpha_dB,
                  noise='mean', noise_modes=['LP01'], backward=True,
                  ).propagate(grid.cw(P0)[None, :], lengths[-1], dz=100.0, z_save=lengths[:-1])

    k = int(np.argmin(np.abs(grid.Omega - 2 * np.pi * 13.2e12)))
    fwd, bwd = res.noise_psd[:, 0, k], res.noise_psd_backward[:, 0, k]
    coef = fwd[0] / (P0 * lengths[0] * np.exp(-alpha * lengths[0]))   # calibrate once
    for i, L in enumerate(lengths):
        want_f = coef * P0 * L * np.exp(-alpha * L)
        want_b = coef * P0 * (1 - np.exp(-2 * alpha * L)) / (2 * alpha)
        assert abs(fwd[i] / want_f - 1) < 5e-3, (L, fwd[i], want_f)
        assert abs(bwd[i] / want_b - 1) < 5e-3, (L, bwd[i], want_b)
    assert abs(bwd[0] / fwd[0] - 1) < 0.01            # equal while alpha L << 1
    assert bwd[-1] / fwd[-1] > 4                       # forward has turned over by 50 km
    print(f'backward noise OK: backward/forward = ' +
          ', '.join(f'{L / 1e3:g} km {b / f:.2f}' for L, f, b in zip(lengths, fwd, bwd)) +
          f'; backward saturates at {bwd[-1]:.3e} W/Hz (1/alpha = {1 / alpha / 1e3:.1f} km)')


def check_polarisation():
    """Vector propagation: a single populated polarisation reduces to the scalar solver
    exactly; a weak orthogonal probe picks up gamma*(2/3)*P*L of XPM against the factor 2
    a co-polarised channel sees -- the isotropic-chi(3) ratio fiber.polarization also
    verifies; and PMD is unitary, conserving total power over both components."""
    modes = FibreModes(DESIGNS['om3'], LAM, span_Hz=5e12, max_modes=3)
    grid = TimeGrid(256, 1e-12, LAM)
    gamma = N2 * grid.omega0 / c / modes.effective_area(0)
    P0, probe, L = 1.0, 1e-6, 20.0

    scalar = GMMNLSE(modes, grid, ['LP01'], raman=None, alpha_dB_km=0.0, dispersion_order=1,
                     ).propagate(grid.cw(P0)[None, :], L, dz=0.5).fields[-1, 0]
    vector = GMMNLSE(modes, grid, ['LP01', 'LP01'], polarisation=['x', 'y'], raman=None,
                     alpha_dB_km=0.0, dispersion_order=1,
                     ).propagate(np.stack([grid.cw(P0), np.zeros(grid.n_points, complex)]),
                                 L, dz=0.5).fields[-1]
    assert np.max(np.abs(vector[0] - scalar)) < 1e-12 * np.abs(scalar).max()
    assert np.max(np.abs(vector[1])) == 0.0

    def probe_phase(pol_labels):
        gmm = GMMNLSE(modes, grid, ['LP01', 'LP01'], pumps=[0], polarisation=pol_labels,
                      raman=None, alpha_dB_km=0.0, dispersion_order=1)
        A0 = np.stack([grid.cw(P0), grid.cw(probe)])
        out = gmm.propagate(A0, L, dz=0.5).fields[-1, 1]
        return float(np.angle(out[0] / np.sqrt(probe)))

    cross = probe_phase(['x', 'y']) / (gamma * P0 * L)
    assert abs(cross - 2 / 3) < 0.02, cross

    A0 = np.stack([grid.cw(0.7), grid.cw(0.3)])
    pmd = GMMNLSE(modes, grid, ['LP01', 'LP01'], polarisation=['x', 'y'], raman=None,
                  n2=0.0, alpha_dB_km=0.0, dispersion_order=1, D_PMD=0.5, seed=0)
    out = pmd.propagate(A0, 500.0, dz=10.0).fields[-1]
    drift = abs(np.sum(np.abs(out) ** 2) / np.sum(np.abs(A0) ** 2) - 1)
    assert drift < 1e-12, drift
    print(f'polarisation OK: one populated polarisation reproduces the scalar solver to '
          f'machine precision; cross-polarised XPM factor {cross:.4f} (exact 2/3); PMD '
          f'unitary to {drift:.1e}')


CHECKS = {
    'single_mode': check_single_mode,
    'self_steepening': check_self_steepening,
    'degenerate': check_degenerate_coherent_term,
    'raman_gain': check_raman_gain,
    'mean_noise': check_mean_noise,
    'stochastic_noise': check_stochastic_noise,
    'photon_number': check_photon_number,
    'mode_coupling': check_mode_coupling,
    'adaptive': check_adaptive_stepping,
    'mode_loss': check_mode_loss,
    'dispersive_overlaps': check_dispersive_overlaps,
    'vacuum_seed': check_vacuum_seed,
    'backward': check_backward_noise,
    'polarisation': check_polarisation,
}

if __name__ == '__main__':
    for name in (sys.argv[1:] or CHECKS):
        CHECKS[name]()
    if not sys.argv[1:]:
        print('\nAll fiber.gmmnlse checks passed.')
