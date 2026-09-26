"""Validation of the Hollenbeck-Cantrell Raman model (fiber.raman_models) and the
wavelength-dependent attenuation model (fiber.attenuation) in every propagator that takes it.

    python tests/test_spectral_models.py                  # all checks
    python tests/test_spectral_models.py hc_transform ... # selected checks
"""
import sys

import numpy as np

from fiber.attenuation import SMF28_ULTRA_MAX_DB_KM, SpectralLoss, smf28_ultra
from fiber.constants import c
from fiber.fiber_params import make_fiber
from fiber.gmmnlse import GMMNLSE, ModeLoss, TimeGrid
from fiber.grin_modes import DESIGNS, FibreModes
from fiber.polarization import PolarizationPropagator
from fiber.propagator import FiberPropagator
from fiber.raman_models import (_HC_AMPLITUDE, _HC_GAUSS_FWHM, _HC_LORENTZ_FWHM, _HC_POSITION,
                                HollenbeckCantrell, LinAgrawal, gain_shape,
                                peak_gain_coefficient)
from fiber.wdm_propagator import WDMPropagator

LAM = 1550e-9
N2 = 2.6e-20


def check_hc_transform():
    """The closed-form (Faddeeva) transform must equal a direct numerical Fourier integral of
    the time-domain sum of damped, Gaussian-broadened oscillators, with H(0) = 1 and the gain
    peak at the measured 13.2 THz."""
    model = HollenbeckCantrell()
    t = np.arange(0, 6e-12, 0.05e-15)
    x = 100 * c
    h = sum(a * np.exp(-np.pi * x * lw * t - (np.pi * x * gw) ** 2 * t ** 2 / 4)
            * np.sin(2 * np.pi * x * p * t)
            for p, a, gw, lw in zip(_HC_POSITION, _HC_AMPLITUDE, _HC_GAUSS_FWHM, _HC_LORENTZ_FWHM))
    h /= np.trapezoid(h, t)
    Omega = 2 * np.pi * np.array([0.1, 0.39, 1, 3, 8, 13.2, 15, 20, 30, 40]) * 1e12
    numeric = np.array([np.trapezoid(h * np.exp(-1j * W * t), t) for W in Omega])
    err = np.max(np.abs(model.H(Omega) - numeric)) / np.max(np.abs(numeric))
    assert err < 1e-5, err
    assert abs(model.H(0.0) - 1) < 1e-12
    nu = np.linspace(0.05e12, 40e12, 8000)
    peak = nu[np.argmax(gain_shape(model, 2 * np.pi * nu))]
    assert abs(peak - 13.2e12) < 0.1e12, peak
    g_peak = peak_gain_coefficient(model, N2, LAM)
    matched = peak_gain_coefficient(HollenbeckCantrell(f_R=0.22), N2, LAM)
    assert abs(matched / peak_gain_coefficient(LinAgrawal(), N2, LAM) - 1) < 0.01
    print(f'Hollenbeck-Cantrell transform OK: closed form vs numerical integral {err:.1e}, '
          f'gain peak {peak / 1e12:.2f} THz, peak g_R {g_peak:.2e} m/W at f_R = 0.18')


def check_hc_in_gmmnlse():
    """CW pump in LP01, weak probe in LP11a at the Stokes shift, Hollenbeck-Cantrell Raman:
    probe gain must match exp[(n2 omega_s/c) S gain_shape P0 L], as for LinAgrawal."""
    modes = FibreModes(DESIGNS['om3'], LAM, span_Hz=26e12, max_modes=3)
    grid = TimeGrid(2 ** 10, 20e-15, LAM)
    shift = round(13.2e12 / grid.df) * grid.df
    P0, L = 1.0, 1000.0
    model = HollenbeckCantrell()
    A0 = np.stack([grid.cw(P0), grid.cw(1e-6, -shift)])
    out = GMMNLSE(modes, grid, ['LP01', 'LP11a'], raman=model,
                  alpha_dB_km=0.0).propagate(A0, L, dz=10.0).fields[-1, 1]
    k = int(np.argmax(np.abs(np.fft.fft(A0[1]))))
    gain_num = np.log(np.abs(np.fft.fft(out)[k]) ** 2 / np.abs(np.fft.fft(A0[1])[k]) ** 2)
    S = modes.overlap_tensor([0, modes.index_of('LP11a')])[1, 0, 1, 0]
    omega_s = grid.omega0 - 2 * np.pi * shift
    gain_ana = N2 * omega_s / c * S * gain_shape(model, 2 * np.pi * shift) * P0 * L
    assert abs(gain_num / gain_ana - 1) < 2e-3, (gain_num, gain_ana)
    print(f'Hollenbeck-Cantrell in GMMNLSE OK: ln G = {gain_num:.5f} vs analytic {gain_ana:.5f}')


def check_loss_fit():
    """smf28_ultra reproduces the Corning maxima it was fitted to, with a physical Rayleigh
    coefficient; anchored() pins one wavelength exactly and keeps the shape."""
    model = smf28_ultra()
    lam = np.array(sorted(SMF28_ULTRA_MAX_DB_KM))
    spec = np.array([SMF28_ULTRA_MAX_DB_KM[x] for x in lam])
    resid = np.max(np.abs(model.dB_km(lam) - spec))
    assert resid < 0.01, resid
    assert 0.7 < model.rayleigh < 1.0, model.rayleigh     # silica SMF: ~0.8-0.9 dB/km um^4
    assert model.uv == 0.0
    pinned = model.anchored(1550e-9, 0.17)
    assert abs(pinned.dB_km(1550e-9) - 0.17) < 1e-12
    ratio = pinned.dB_km(lam) / model.dB_km(lam)
    assert np.ptp(ratio) < 1e-12
    parts = sum(model.components(1383e-9).values())
    assert abs(parts - model.dB_km(1383e-9)) < 1e-12
    print(f'loss fit OK: max residual {resid:.4f} dB/km against the datasheet, '
          f'A_R = {model.rayleigh:.3f} dB/km um^4, '
          f'{model.dB_km(1310e-9):.3f} / {model.dB_km(1550e-9):.3f} dB/km at 1310 / 1550 nm')


def check_flat_limit():
    """A wavelength-flat SpectralLoss must reproduce the scalar alpha exactly in every
    propagator that accepts one."""
    flat = SpectralLoss(flat=0.2)
    scalar, spectral = make_fiber('smf28'), make_fiber('smf28', loss_model=flat)
    assert spectral.alpha == scalar.alpha
    n, dt = 2 ** 10, 1e-12
    t = (np.arange(n) - n / 2) * dt
    A0 = np.sqrt(0.1) / np.cosh(t / 20e-12) + 0j
    worst = 0.0
    for run in (lambda f: FiberPropagator(f).propagate(A0, dt, 20e3, n_steps=40),
                lambda f: WDMPropagator(f, [0.0, 100e9]).propagate(np.stack([A0, A0]), dt, 20e3,
                                                                    n_steps=40),
                lambda f: PolarizationPropagator(f, seed=1).propagate(A0, dt, 20e3, n_steps=40)):
        a, b = run(scalar), run(spectral)
        worst = max(worst, np.max(np.abs(a - b)) / np.max(np.abs(a)))

    modes = FibreModes(DESIGNS['smf28'], LAM, span_Hz=26e12, max_modes=1)
    grid = TimeGrid(2 ** 10, 20e-15, LAM)
    A = grid.cw(1.0)[None, :]
    for spec, ref in ((flat, 0.2), (ModeLoss(base_dB_km=flat), ModeLoss(base_dB_km=0.2))):
        kw = dict(raman=LinAgrawal(), noise='mean')
        a = GMMNLSE(modes, grid, ['LP01'], alpha_dB_km=ref, **kw).propagate(A, 5e3, dz=500.0)
        b = GMMNLSE(modes, grid, ['LP01'], alpha_dB_km=spec, **kw).propagate(A, 5e3, dz=500.0)
        worst = max(worst, np.max(np.abs(a.fields - b.fields)) / np.max(np.abs(a.fields)),
                    np.max(np.abs(a.noise_psd - b.noise_psd)) / np.max(a.noise_psd))
    assert worst < 1e-13, worst
    print(f'flat-loss limit OK: FiberPropagator, WDM, polarisation and GMMNLSE (field and '
          f'mean noise) match the scalar alpha to {worst:.1e}')


def check_spectral_propagation():
    """Linear propagation: each frequency bin must decay as exp(-alpha(lambda) L), with the
    physical frequency at omega0 - Omega, and each WDM channel at its own carrier."""
    model = smf28_ultra()
    L = 50e3
    fiber = make_fiber('smf28', loss_model=model, material_overrides=dict(n2=0.0))
    n, dt = 2 ** 12, 20e-15
    t = (np.arange(n) - n / 2) * dt
    A0 = np.exp(-t ** 2 / (2 * (40e-15) ** 2)) + 0j
    out = FiberPropagator(fiber, include_raman=False).propagate(A0, dt, L, n_steps=10)
    Omega = 2 * np.pi * np.fft.fftfreq(n, d=dt)
    lam = 2 * np.pi * c / (fiber.omega0 - Omega)
    Sin, Sout = np.abs(np.fft.fft(A0)) ** 2, np.abs(np.fft.fft(out)) ** 2
    ok = Sin > 1e-6 * Sin.max()
    err = np.max(np.abs(np.log(Sout[ok] / Sin[ok]) + model.per_m(lam[ok]) * L))
    assert err < 1e-9, err
    loss_span = model.dB_km(lam[ok]) * L / 1e3
    assert np.ptp(loss_span) > 1.0      # the test band sees a real spectral tilt

    offsets = np.array([0.0, -8e12, 10e12])   # 1550, ~1616, ~1473 nm
    A = np.ones((3, 64), dtype=complex)
    out = WDMPropagator(fiber, offsets, include_raman=False).propagate(A, 1e-12, L, n_steps=10)
    lam_ch = c / (c / LAM + offsets)
    err_ch = np.max(np.abs(np.log(np.mean(np.abs(out) ** 2, axis=1)) + model.per_m(lam_ch) * L))
    assert err_ch < 1e-9, err_ch

    modes = FibreModes(DESIGNS['smf28'], LAM, span_Hz=30e12, max_modes=1)
    grid = TimeGrid(2 ** 11, 20e-15, LAM)
    A0 = np.exp(-grid.t ** 2 / (2 * (40e-15) ** 2)) + 0j
    res = GMMNLSE(modes, grid, ['LP01'], n2=0.0, raman=None, alpha_dB_km=model,
                  self_steepening=False).propagate(A0[None, :], L, dz=L / 10)
    Sin, Sout = np.abs(np.fft.fft(A0)) ** 2, np.abs(np.fft.fft(res.fields[-1, 0])) ** 2
    ok = (Sin > 1e-6 * Sin.max()) & modes.is_guided(0, grid.omega)
    lam = 2 * np.pi * c / grid.omega
    err_g = np.max(np.abs(np.log(Sout[ok] / Sin[ok]) + model.per_m(lam[ok]) * L))
    assert err_g < 1e-9, err_g
    print(f'spectral propagation OK: per-bin decay matches exp(-alpha(lambda) L) to '
          f'{max(err, err_ch, err_g):.1e} nepers across a {np.ptp(loss_span):.1f} dB tilt '
          f'(FiberPropagator, WDM channels, GMMNLSE)')


CHECKS = {
    'hc_transform': check_hc_transform,
    'hc_gmmnlse': check_hc_in_gmmnlse,
    'loss_fit': check_loss_fit,
    'flat_limit': check_flat_limit,
    'spectral_propagation': check_spectral_propagation,
}

if __name__ == '__main__':
    for name in (sys.argv[1:] or CHECKS):
        CHECKS[name]()
    if not sys.argv[1:]:
        print('\nAll spectral-model checks passed.')
