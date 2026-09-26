"""
Wavelength-dependent fibre attenuation, in the form Woodward's thesis uses (Imperial, 2015,
eqs. 2.3.3-2.3.4, after Walker, JLT 4, 1125, 1986):

    alpha(lam) = A_R / lam^4                            Rayleigh scattering
               + A_0                                    wavelength-independent excess
               + A_UV exp(B_UV / lam)                   Urbach tail of the UV edge
               + A_IR exp(-B_IR / lam)                  multiphonon tail of the IR edge
               + sum_i A_i exp(-(lam - lam_i)^2 / (2 sigma_i^2))   OH overtones
                                                        [dB/km, lam in um]

A_0 is not in the thesis form. It is the intercept of the classic alpha-versus-lam^-4 plot:
waveguide imperfection and microbend loss, which Walker's form lumps into the impurity
term. It is needed because the fit below is otherwise forced to explain the 1310/1550 ratio
(1.78 at the datasheet, 1.96 for pure lam^-4) with the UV tail.

The amplitudes are linear in alpha, so SpectralLoss.fit recovers A_R, A_0, A_IR and the OH
amplitudes from a handful of datasheet points by non-negative least squares. A_UV is NOT
fitted: across 1.3-1.6 um exp(B_UV/lam) and lam^-4 are nearly collinear, and a free fit
splits the loss between them about evenly (UV ~0.09 dB/km at 1550 nm, when the UV tail is
known to be a small fraction of the Rayleigh loss there). It is held at the value passed
(default 0); set it from a composition-based estimate if you have one. B_UV = 4.63 um and B_IR = 48.48 um are the textbook edge constants
for germanosilicate SMF (Keiser, Optical Fiber Communications) -- quoted from memory of that
text and not checked against Walker's paper, so treat them as the least certain numbers
here. They set how fast each tail falls; the fitted amplitudes absorb any error at the
datasheet wavelengths, so it shows up only between and beyond those points. The OH line
width (sigma = 10 nm, ~24 nm FWHM) is likewise an assumption, not a fitted value.

smf28_ultra() is fitted to Corning's SMF-28 Ultra maximum attenuation (PI-1424, G.652.D):
0.32 / 0.32 / 0.21 / 0.18 / 0.20 dB/km at 1310 / 1383 / 1490 / 1550 / 1625 nm. These are
specification MAXIMA, so the curve is an upper envelope; typical spans run 0.01-0.02 dB/km
lower. Rescale with anchored() to a measured value at one wavelength, or refit to an OTDR
or cut-back spectrum. The form is only meant for ~1.2-1.7 um; outside that it is an
extrapolation of two exponential tails.

Which propagators use it
------------------------
FiberParams(loss_model=...) makes alpha frequency dependent in FiberPropagator (and its
quantum subclass), WDMPropagator (each channel at its own carrier) and
PolarizationPropagator, and fixes the scalar fiber.alpha at lambda0 for the closed-form
modules (brillouin, four_wave_mixing, rayleigh_backscatter). GMMNLSE takes a SpectralLoss as
alpha_dB_km, or as ModeLoss.base_dB_km beneath differential mode attenuation. The
multimode_fiber family (multimode_propagator, mode_coupling, hybrid_crosstalk) keeps its
per-mode scalar alpha: its DMA power law is calibrated per OM grade at one wavelength.

    from fiber.attenuation import SpectralLoss, smf28_ultra
"""
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import nnls

_DB_PER_NEPER = 10 * np.log10(np.e)

# Corning SMF-28 Ultra, maximum attenuation (dB/km) by wavelength (m); 1383 nm is the
# post-hydrogen-ageing figure.
SMF28_ULTRA_MAX_DB_KM = {1310e-9: 0.32, 1383e-9: 0.32, 1490e-9: 0.21, 1550e-9: 0.18,
                         1625e-9: 0.20}


@dataclass(frozen=True)
class SpectralLoss:
    """alpha(lam) in dB/km from the Rayleigh, UV, IR and OH terms above.

    rayleigh : A_R (dB/km um^4)
    flat : A_0 (dB/km)
    uv, ir : A_UV, A_IR (dB/km)
    uv_edge_um, ir_edge_um : B_UV, B_IR (um)
    oh_peaks : tuple of (centre_nm, amplitude_dB_km, sigma_nm)
    """
    rayleigh: float = 0.0
    flat: float = 0.0
    uv: float = 0.0
    ir: float = 0.0
    uv_edge_um: float = 4.63
    ir_edge_um: float = 48.48
    oh_peaks: tuple = ()

    def _columns(self, wavelength):
        lam = np.asarray(wavelength, dtype=float) * 1e6
        cols = [lam ** -4, np.ones_like(lam), np.exp(self.uv_edge_um / lam), np.exp(-self.ir_edge_um / lam)]
        cols += [np.exp(-(lam * 1e3 - c0) ** 2 / (2 * s ** 2)) for c0, _, s in self.oh_peaks]
        return cols

    def _amplitudes(self):
        return [self.rayleigh, self.flat, self.uv, self.ir] + [a for _, a, _ in self.oh_peaks]

    def dB_km(self, wavelength):
        """Attenuation (dB/km) at wavelength (m); scalar or array."""
        return sum(a * col for a, col in zip(self._amplitudes(), self._columns(wavelength)))

    def per_m(self, wavelength):
        """Power attenuation coefficient (1/m) at wavelength (m)."""
        return self.dB_km(wavelength) / _DB_PER_NEPER / 1e3

    def components(self, wavelength):
        """Each term's contribution (dB/km), keyed 'rayleigh', 'flat', 'uv', 'ir', 'oh_<nm>'."""
        keys = ['rayleigh', 'flat', 'uv', 'ir'] + [f'oh_{c0:g}' for c0, _, _ in self.oh_peaks]
        return {k: a * col for k, a, col in zip(keys, self._amplitudes(),
                                                self._columns(wavelength))}

    def anchored(self, wavelength, dB_km):
        """Copy scaled as a whole so that alpha(wavelength) = dB_km, keeping the shape."""
        s = dB_km / float(self.dB_km(wavelength))
        return replace(self, rayleigh=self.rayleigh * s, flat=self.flat * s, uv=self.uv * s, ir=self.ir * s,
                       oh_peaks=tuple((c0, a * s, w) for c0, a, w in self.oh_peaks))

    @classmethod
    def fit(cls, wavelengths, dB_km, oh_peaks=((1383.0, 10.0),), uv=0.0, **edges):
        """Non-negative least-squares fit of A_R, A_0, A_IR and the OH amplitudes.

        wavelengths : (m), dB_km : attenuation there
        oh_peaks : (centre_nm, sigma_nm) per OH line to include; its amplitude is fitted
        uv : A_UV, held fixed (see the module docstring for why it is not fitted)
        edges : uv_edge_um / ir_edge_um overrides

        With as many points as terms the fit interpolates them; leave out an OH line whose
        wavelength has no data point, or its amplitude is unconstrained.
        """
        shape = cls(uv=uv, oh_peaks=tuple((c0, 0.0, s) for c0, s in oh_peaks), **edges)
        cols = shape._columns(np.asarray(wavelengths, dtype=float))
        target = np.asarray(dB_km, dtype=float) - uv * cols[2]
        amp, _ = nnls(np.column_stack(cols[:2] + cols[3:]), target)
        return replace(shape, rayleigh=amp[0], flat=amp[1], ir=amp[2],
                       oh_peaks=tuple((c0, a, s) for (c0, s), a in zip(oh_peaks, amp[3:])))


def smf28_ultra():
    """SpectralLoss fitted to the Corning SMF-28 Ultra maxima (an upper envelope)."""
    lam, dB = zip(*sorted(SMF28_ULTRA_MAX_DB_KM.items()))
    return SpectralLoss.fit(lam, dB)
