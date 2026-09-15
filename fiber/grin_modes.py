"""
Scalar LP modes of graded-index (alpha-profile) and step-index fibres: propagation
constants beta_p(omega), normalised mode fields, and the four-index nonlinear overlap
tensor S_plmn used by fiber.gmmnlse.

Index profile, with Delta n held fixed across wavelength and the cladding given by
Malitson's fused-silica Sellmeier equation:

    n(r, lam) = n_clad(lam) + dn0 * (1 - (r/a)**alpha)    r < a
              = n_clad(lam)                               r >= a

dn0 follows from the numerical aperture at lambda_NA; alpha = inf is a step index.

Radial equation for LP_lm with psi = R(r) * {1, cos(l phi), sin(l phi)}:

    (1/r) d/dr (r dR/dr) - (l^2/r^2) R + k^2 n(r)^2 R = beta^2 R

discretised with a finite-volume stencil on r_j = (j + 1/2) dr (zero flux through the
axis, R = 0 at the cladding edge) and symmetrised with y = sqrt(r) R, which makes it a
tridiagonal eigenproblem.

Modes are normalised to integral psi^2 dA = 1, so S_plmn = integral psi_p psi_l psi_m
psi_n dA and A_eff,p = 1/S_pppp. Fields and overlaps are taken at the design
wavelength; only beta is frequency dependent (the usual GMMNLSE approximation).

    from fiber.grin_modes import FibreModes, DESIGNS
    modes = FibreModes(DESIGNS['om3'], wavelength=1551.72e-9)
"""
import itertools
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh_tridiagonal

from fiber.constants import c

_SELLMEIER_B = (0.6961663, 0.4079426, 0.8974794)
_SELLMEIER_C_UM = (0.0684043, 0.1162414, 9.896161)


def silica_index(wavelength):
    """Refractive index of fused silica (Malitson, JOSA 55, 1205, 1965)."""
    x2 = (np.asarray(wavelength, dtype=float) * 1e6) ** 2
    n2 = 1.0
    for B, C in zip(_SELLMEIER_B, _SELLMEIER_C_UM):
        n2 = n2 + B * x2 / (x2 - C * C)
    return np.sqrt(n2)


@dataclass(frozen=True)
class FibreDesign:
    name: str
    core_radius: float
    NA: float
    alpha_profile: float
    clad_radius: float = 62.5e-6
    lambda_NA: float = 850e-9

    @property
    def dn0(self):
        n_cl = silica_index(self.lambda_NA)
        return float(np.sqrt(n_cl ** 2 + self.NA ** 2) - n_cl)

    def index(self, r, wavelength):
        r = np.asarray(r, dtype=float)
        if np.isinf(self.alpha_profile):
            shape = (r < self.core_radius).astype(float)
        else:
            shape = np.where(r < self.core_radius,
                             1.0 - (r / self.core_radius) ** self.alpha_profile, 0.0)
        return silica_index(wavelength) + self.dn0 * shape


# OM2-OM5 share the 50/125 NA 0.2 geometry; the grades differ in how precisely the
# profile hits its optimum alpha, which is the parameter to vary for DMD studies.
# SMF-28 uses an effective index-step NA of 0.115 rather than the 0.14 datasheet value
# (a far-field measurement that overstates the step): with a 4.1 um core it reproduces
# Corning's A_eff = 85 um^2 and D = 16.6 ps/nm/km at 1550 nm, zero dispersion at
# 1304 nm and LP11 cutoff at 1232 nm. Solve SMF-28 with dr <= 0.02 um.
DESIGNS = {
    'om1': FibreDesign('om1', core_radius=31.25e-6, NA=0.275, alpha_profile=2.05),
    'om3': FibreDesign('om3', core_radius=25e-6, NA=0.200, alpha_profile=2.05),
    'smf28': FibreDesign('smf28', core_radius=4.1e-6, NA=0.115, alpha_profile=np.inf,
                         lambda_NA=1310e-9),
}


@dataclass(frozen=True)
class LPMode:
    l: int
    m: int
    orientation: str  # '' for l = 0, 'a' = cos(l phi), 'b' = sin(l phi)

    @property
    def label(self):
        return f'LP{self.l}{self.m}{self.orientation}'


def _radial_solve(design, wavelength, l, dr):
    """Guided LP_l* solutions: radii, beta (descending), R(r) columns normalised to
    integral R^2 r dr = 1 with the first lobe positive."""
    r = (np.arange(int(round(design.clad_radius / dr))) + 0.5) * dr
    k = 2 * np.pi / wavelength
    n = design.index(r, wavelength)
    rp, rm = r + 0.5 * dr, r - 0.5 * dr
    diag = (rp + rm) / (dr ** 2 * r) + l ** 2 / r ** 2 - (k * n) ** 2
    off = -rp[:-1] / (dr ** 2 * np.sqrt(r[:-1] * r[1:]))
    upper = -(k * silica_index(wavelength)) ** 2
    lower = diag.min() - 2 * np.abs(off).max() - 1.0
    if lower >= upper:
        return r, np.empty(0), np.empty((len(r), 0))
    mu, y = eigh_tridiagonal(diag, off, select='v', select_range=(lower, upper))
    order = np.argsort(mu)
    beta = np.sqrt(-mu[order])
    R = y[:, order] / np.sqrt(r)[:, None]
    R /= np.sqrt(np.sum(R ** 2 * r[:, None], axis=0) * dr)
    first_lobe = np.argmax(np.abs(R) > 0.3 * np.abs(R).max(axis=0), axis=0)
    R *= np.sign(R[first_lobe, np.arange(R.shape[1])])
    return r, beta, R


class FibreModes:
    """Guided scalar LP modes of `design` at `wavelength`.

    Parameters
    ----------
    design : FibreDesign
    wavelength : float -- design wavelength (m); fields and overlaps are taken here
    span_Hz : float -- beta(omega) is fitted over wavelength +- this optical frequency
    dr : float -- radial grid step (m)
    max_modes : int or None -- keep only the highest-beta modes (degenerate a/b pairs
        are kept together)
    fit_degree : int -- polynomial degree of each beta(omega) fit
    """

    def __init__(self, design, wavelength, span_Hz=40e12, dr=0.05e-6, max_modes=None,
                 fit_degree=6, n_scan=17):
        self.design = design
        self.wavelength = wavelength
        self.omega0 = 2 * np.pi * c / wavelength
        self.span = 2 * np.pi * span_Hz

        modes, betas, radial = [], [], []
        l = 0
        while True:
            r, beta, R = _radial_solve(design, wavelength, l, dr)
            if beta.size == 0:
                break
            for m in range(beta.size):
                for orient in ([''] if l == 0 else ['a', 'b']):
                    modes.append(LPMode(l, m + 1, orient))
                    betas.append(beta[m])
                    radial.append(R[:, m])
            l += 1
        order = np.argsort(-np.asarray(betas), kind='stable')
        modes = [modes[i] for i in order]
        betas = np.asarray(betas)[order]
        radial = np.asarray(radial)[order]
        if max_modes is not None and max_modes < len(modes):
            keep = max_modes
            if modes[keep - 1].orientation == 'a':
                keep += 1
            modes, betas, radial = modes[:keep], betas[:keep], radial[:keep]

        self.r = r
        self.dr = dr
        self.modes = modes
        self.labels = [p.label for p in modes]
        self.beta0 = betas
        self._R = radial  # (M, Nr)
        self._fit_dispersion(dr, fit_degree, n_scan)

    def __len__(self):
        return len(self.modes)

    def index_of(self, label):
        return self.labels.index(label)

    # ------------------------------------------------------------------ dispersion
    def _fit_dispersion(self, dr, degree, n_scan):
        d_omega = np.linspace(-self.span, self.span, n_scan)
        wavelengths = 2 * np.pi * c / (self.omega0 + d_omega)
        lm = sorted({(p.l, p.m) for p in self.modes})
        table = {key: np.full(n_scan, np.nan) for key in lm}
        for i, lam in enumerate(wavelengths):
            for l in sorted({key[0] for key in lm}):
                _, beta, _ = _radial_solve(self.design, lam, l, dr)
                for (ll, m) in lm:
                    if ll == l and m <= beta.size:
                        table[(ll, m)][i] = beta[m - 1]
        self._x_scale = self.span
        self._fits, self._guided_from, self._cutoff_found = {}, {}, {}
        for key, beta in table.items():
            ok = np.isfinite(beta)
            deg = min(degree, ok.sum() - 2)
            if deg < 2:
                raise RuntimeError(f'LP{key[0]}{key[1]} is guided at too few scan '
                                   f'frequencies to fit its dispersion; reduce span_Hz')
            self._fits[key] = np.polynomial.Polynomial.fit(
                d_omega[ok] / self._x_scale, beta[ok], deg, domain=[-1, 1])
            first = int(np.flatnonzero(ok).min())
            self._cutoff_found[key] = first > 0
            if first == 0:
                self._guided_from[key] = d_omega[0]
                continue
            # bisect between the last unguided and first guided scan points, so the guided
            # mask switches at the true cutoff rather than at a scan frequency
            lo, hi = d_omega[first - 1], d_omega[first]
            for _ in range(20):
                mid = 0.5 * (lo + hi)
                _, b_mid, _ = _radial_solve(self.design, 2 * np.pi * c / (self.omega0 + mid), key[0], dr)
                lo, hi = (lo, mid) if b_mid.size >= key[1] else (mid, hi)
            self._guided_from[key] = hi

    def beta(self, p, omega):
        """beta_p at physical angular frequency omega (rad/m)."""
        mode = self.modes[p]
        return self._fits[(mode.l, mode.m)]((np.asarray(omega) - self.omega0) / self._x_scale)

    def beta_derivative(self, p, order):
        """d^order beta_p / d omega^order at the design frequency (s^order/m)."""
        mode = self.modes[p]
        return float(self._fits[(mode.l, mode.m)].deriv(order)(0.0) / self._x_scale ** order)

    def is_guided(self, p, omega):
        mode = self.modes[p]
        return (np.asarray(omega) - self.omega0) >= self._guided_from[(mode.l, mode.m)]

    def cutoff_wavelength(self, p):
        """Long-wavelength cutoff of mode p (m), or nan if it stays guided across the span."""
        key = (self.modes[p].l, self.modes[p].m)
        if not self._cutoff_found[key]:
            return float('nan')
        return 2 * np.pi * c / (self.omega0 + self._guided_from[key])

    def dispersion_parameter(self, p):
        """D = -2 pi c beta2 / lambda^2 in ps/(nm km)."""
        return -2 * np.pi * c * self.beta_derivative(p, 2) / self.wavelength ** 2 * 1e6

    # ------------------------------------------------------------------ overlaps
    def _azimuth(self, idx):
        l_max = max(self.modes[i].l for i in idx)
        phi = np.linspace(0, 2 * np.pi, 64 + 16 * l_max, endpoint=False)
        rows = []
        for i in idx:
            mode = self.modes[i]
            if mode.l == 0:
                rows.append(np.full_like(phi, 1 / np.sqrt(2 * np.pi)))
            elif mode.orientation == 'a':
                rows.append(np.cos(mode.l * phi) / np.sqrt(np.pi))
            else:
                rows.append(np.sin(mode.l * phi) / np.sqrt(np.pi))
        return np.asarray(rows), phi[1] - phi[0]

    def overlap_tensor(self, idx=None):
        """Dense S_plmn (1/m^2) over the modes in `idx` (default: all)."""
        idx = list(range(len(self))) if idx is None else list(idx)
        R = self._R[idx]
        Phi, dphi = self._azimuth(idx)
        wr = self.r * self.dr
        K = len(idx)
        S = np.zeros((K, K, K, K))
        for p, l, m, n in itertools.combinations_with_replacement(range(K), 4):
            ang = np.sum(Phi[p] * Phi[l] * Phi[m] * Phi[n]) * dphi
            if abs(ang) < 1e-12:
                continue
            val = np.sum(R[p] * R[l] * R[m] * R[n] * wr) * ang
            for perm in set(itertools.permutations((p, l, m, n))):
                S[perm] = val
        return S

    def intensity_overlaps(self, receivers=None, sources=None):
        """S_qlql between receiver modes q and source modes l, shape (Q, L)."""
        receivers = list(range(len(self))) if receivers is None else list(receivers)
        sources = list(range(len(self))) if sources is None else list(sources)
        allidx = receivers + sources
        Phi, dphi = self._azimuth(allidx)
        Q = len(receivers)
        rad = (self._R[receivers] ** 2 * self.r * self.dr) @ (self._R[sources] ** 2).T
        ang = (Phi[:Q] ** 2) @ (Phi[Q:] ** 2).T * dphi
        return rad * ang

    def effective_area(self, p):
        return 1.0 / self.overlap_tensor([p])[0, 0, 0, 0]

    def degenerate_groups(self, tol=1e-6):
        """Lists of mode indices whose beta0 agree to `tol` rad/m."""
        groups = []
        for i in np.argsort(-self.beta0):
            if groups and abs(self.beta0[groups[-1][0]] - self.beta0[i]) <= tol:
                groups[-1].append(int(i))
            else:
                groups.append([int(i)])
        return groups
