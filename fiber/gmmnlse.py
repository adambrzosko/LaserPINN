"""
Generalised multimode nonlinear Schroedinger equation (GMMNLSE) solver, scalar and
co-polarised (Poletti & Horak, JOSA B 25, 1645, 2008):

    dA_p/dz = L_p{A_p}
            + i (n2/c) (omega0 + i d/dt) sum_lmn S_plmn A_l [(1-f_R) A_m A_n* + f_R h*(A_m A_n*)]
            + spontaneous Raman noise

L_p is the full per-mode dispersion beta_p(omega) from fiber.grin_modes plus loss,
written in the frame of the reference mode (its beta0 and beta1 removed). The
(omega0 + i d/dt) factor is self-steepening. Integration is fourth-order Runge-Kutta in
the interaction picture (RK4IP; Hult, JLT 25, 3770, 2007). Envelope convention as in
the rest of fiber/: physical angular frequency = omega0 - Omega.

Which tensor terms are kept
---------------------------
A term (p, l, m, n) is kept only if

    |beta0_l + beta0_m - beta0_n - beta0_p| <= coherence_tol    (rad/m)   and
    at most two of p, l, m, n are spectators (propagated modes that are not pumps).

Both conditions are invariant under the index pairing that makes the Kerr and Raman
terms conserve photon number, so the truncated equation still conserves it exactly.

Terms failing the first test rotate faster than the step can follow. Over a step longer
than 2*pi/|dbeta0| they average to ~zero (rotating-wave approximation), whereas
integrating them coarsely creates spurious coherent power transfer. The default,
coherence_tol = 0.1/dz, keeps exactly the terms the step resolves: SPM, XPM,
intermodal Raman gain (always phase matched), and coherent coupling within exactly
degenerate mode pairs. Shrinking dz admits the near-matched intergroup four-wave-mixing
terms; coherence_tol = inf with small dz is the complete equation. Spectators hold
noise-level fields, so terms quadratic in them are second order and dropped.

Spontaneous Raman noise
-----------------------
From the fluctuation-dissipation theorem, receiver mode q gains noise from pump mode l
with power spectral density growing as

    dS_q/dz = hbar*omega * g_ql(shift) * P_l * (n_th + [Stokes side])
    g_ql     = 2 (n2 omega / c) S_qlql f_R (-Im H(shift))

noise='mean'        integrates the ensemble-mean PSD exactly for every guided mode
                    (deterministic and fast; what an optical spectrum analyser reads).
                    The pump spectrum is convolved with the Raman kernel without
                    wrap-around.
noise='stochastic'  adds Langevin fields to the propagated modes, driven by the local
                    instantaneous pump power, so pulsed pumps and gated detection are
                    resolved. Assumes each pump is narrowband compared with the Raman
                    spectrum.

Not included: the orthogonal polarisation (cross-polarised Raman and XPM), spontaneous
four-wave mixing (needs vacuum-noise seeding), frequency dependence of the mode fields
and overlaps, and random linear mode coupling.

    from fiber.grin_modes import FibreModes, DESIGNS
    from fiber.gmmnlse import GMMNLSE, TimeGrid
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import fftconvolve

from fiber.constants import c, hbar
from fiber.raman_models import LinAgrawal, gain_shape, spontaneous_shape

_UNGUIDED_LOSS = 1e3  # 1/m amplitude decay applied where a mode is not guided
_DB_PER_NEPER = 10 * np.log10(np.e)


@dataclass(frozen=True)
class TimeGrid:
    n_points: int
    dt: float
    wavelength: float

    @property
    def omega0(self):
        return 2 * np.pi * c / self.wavelength

    @property
    def Omega(self):
        return 2 * np.pi * np.fft.fftfreq(self.n_points, d=self.dt)

    @property
    def omega(self):
        return self.omega0 - self.Omega

    @property
    def t(self):
        return (np.arange(self.n_points) - self.n_points // 2) * self.dt

    @property
    def df(self):
        return 1.0 / (self.n_points * self.dt)

    def cw(self, power, offset_Hz=0.0):
        """CW field of `power` (W) at physical frequency offset `offset_Hz`. The offset must
        sit on a frequency bin so the tone does not leak across the spectrum."""
        bins = offset_Hz / self.df
        if abs(bins - round(bins)) > 1e-6:
            raise ValueError(f'offset {offset_Hz:.6g} Hz is not a multiple of the bin '
                             f'spacing {self.df:.6g} Hz; adjust n_points or dt')
        return np.sqrt(power) * np.exp(-2j * np.pi * round(bins) * self.df * np.arange(self.n_points) * self.dt)


@dataclass
class PropagationResult:
    grid: TimeGrid
    z: np.ndarray                       # saved positions (m)
    labels: list                        # propagated mode labels
    fields: np.ndarray                  # (Nz, K, N) time-domain fields, sqrt(W)
    noise_labels: list = field(default_factory=list)
    noise_psd: np.ndarray = None        # (Nz, Q, N) mean spontaneous-Raman PSD, W/Hz

    def field_psd(self):
        """(Nz, K, N) power spectral density of the coherent fields, W/Hz."""
        N = self.grid.n_points
        return np.abs(np.fft.fft(self.fields, axis=-1)) ** 2 * self.grid.dt / N

    def mean_power(self):
        """(Nz, K) window-averaged power of each propagated mode, W."""
        return np.mean(np.abs(self.fields) ** 2, axis=-1)


def to_dbm_in_rbw(psd, wavelength, rbw_nm):
    """Convert a PSD (W/Hz) at `wavelength` (m) into dBm within an RBW (nm), as an OSA
    reading of a spectrally smooth signal."""
    bandwidth_Hz = c * rbw_nm * 1e-9 / np.asarray(wavelength) ** 2
    return 10 * np.log10(np.maximum(np.asarray(psd) * bandwidth_Hz, 1e-300) * 1e3)


class GMMNLSE:
    """Scalar GMMNLSE propagator over a set of LP modes.

    Parameters
    ----------
    modes : fiber.grin_modes.FibreModes
    grid : TimeGrid -- its wavelength must equal modes.wavelength
    propagate : sequence of mode labels or indices carried as fields
    pumps : subset of `propagate` holding significant power (default: all). The rest
        are spectators, which matters for term selection and noise sources.
    n2 : nonlinear index (m^2/W)
    raman : model from fiber.raman_models, or None for Kerr only
    alpha_dB_km : float, or dict of label -> dB/km with an optional 'default' key
    self_steepening : bool
    noise : 'none', 'mean' or 'stochastic'
    temperature : K, for the phonon occupation
    noise_modes : labels/indices accumulating the mean noise PSD (default: all guided modes)
    coherence_tol : rad/m; None uses 0.1/dz
    dispersion_order : None for the full beta(omega) fit, or an int to truncate the Taylor
        series at omega0 (e.g. 3 keeps beta0..beta3)
    reference : label/index of the mode defining the co-moving frame
    seed : RNG seed for noise='stochastic'
    """

    def __init__(self, modes, grid, propagate, pumps=None, n2=2.6e-20, raman=LinAgrawal(),
                 alpha_dB_km=0.3, self_steepening=True, noise='none', temperature=300.0,
                 noise_modes=None, coherence_tol=None, dispersion_order=None,
                 reference=0, seed=None):
        if abs(grid.wavelength - modes.wavelength) > 1e-15:
            raise ValueError('grid.wavelength must equal modes.wavelength (overlaps are '
                             'evaluated at the mode-solver wavelength)')
        band = np.max(np.abs(grid.Omega))
        if band > modes.span * (1 + 1e-9):
            raise ValueError(f'grid spans +-{band / 2 / np.pi / 1e12:.1f} THz but the dispersion '
                             f'fit covers +-{modes.span / 2 / np.pi / 1e12:.1f} THz; build '
                             f'FibreModes with a larger span_Hz')
        if noise not in ('none', 'mean', 'stochastic'):
            raise ValueError("noise must be 'none', 'mean' or 'stochastic'")

        self.modes, self.grid = modes, grid
        self.prop = [self._index(p) for p in propagate]
        pump_set = set(self.prop if pumps is None else [self._index(p) for p in pumps])
        if not pump_set <= set(self.prop):
            raise ValueError('pumps must be a subset of propagate')
        self.pump_mask = np.array([p in pump_set for p in self.prop])
        self.n2, self.raman = n2, raman
        self.self_steepening = self_steepening
        self.noise, self.temperature = noise, temperature
        self.coherence_tol = coherence_tol
        self.dispersion_order = dispersion_order
        self.ref = self._index(reference)
        self.rng = np.random.default_rng(seed)
        self.noise_modes = (list(range(len(modes))) if noise_modes is None
                            else [self._index(q) for q in noise_modes])
        self._alpha_spec = alpha_dB_km

    # ---------------------------------------------------------------- setup
    def _index(self, p):
        return self.modes.index_of(p) if isinstance(p, str) else int(p)

    def _alpha(self, p):
        spec = self._alpha_spec
        if isinstance(spec, dict):
            dB = spec.get(self.modes.labels[p], spec.get('default', 0.0))
        else:
            dB = spec
        return dB / _DB_PER_NEPER / 1e3  # power attenuation, 1/m

    def _beta(self, p):
        omega = self.grid.omega
        if self.dispersion_order is None:
            return self.modes.beta(p, omega)
        dw = omega - self.modes.omega0
        beta = np.full_like(omega, self.modes.beta(p, self.modes.omega0))
        fact = 1.0
        for k in range(1, self.dispersion_order + 1):
            fact *= k
            beta = beta + self.modes.beta_derivative(p, k) / fact * dw ** k
        return beta

    def _linear_operator(self, idx):
        omega, omega0 = self.grid.omega, self.modes.omega0
        b0 = self.modes.beta(self.ref, omega0)
        b1 = self.modes.beta_derivative(self.ref, 1)
        rows = []
        for p in idx:
            Lp = 1j * (self._beta(p) - b0 - b1 * (omega - omega0)) - self._alpha(p) / 2
            rows.append(np.where(self.modes.is_guided(p, omega), Lp, -_UNGUIDED_LOSS))
        return np.asarray(rows)

    def _build_terms(self, tol):
        K = len(self.prop)
        S = self.modes.overlap_tensor(self.prop)
        b0 = self.modes.beta0[self.prop]
        spec = (~self.pump_mask).astype(int)
        P, L, M, N = (a.ravel() for a in np.meshgrid(*[np.arange(K)] * 4, indexing='ij'))
        s = S[P, L, M, N]
        keep = ((np.abs(s) > 1e-12 * np.abs(S).max())
                & (np.abs(b0[L] + b0[M] - b0[N] - b0[P]) <= tol)
                & (spec[P] + spec[L] + spec[M] + spec[N] <= 2))
        P, L, M, N, s = P[keep], L[keep], M[keep], N[keep], s[keep]
        pairs, w = np.unique(np.stack([M, N], axis=1), axis=0, return_inverse=True)
        self._pair_m, self._pair_n = pairs[:, 0], pairs[:, 1]
        self._terms = {p: (L[P == p], w.ravel()[P == p], s[P == p]) for p in np.unique(P)}
        self.n_terms = int(keep.sum())

    # ---------------------------------------------------------------- operators
    def _nonlinear(self, Af):
        At = np.fft.ifft(Af, axis=1)
        B = At[self._pair_m] * np.conj(At[self._pair_n])
        if self.raman is not None:
            W = ((1 - self.raman.f_R) * B
                 + self.raman.f_R * np.fft.ifft(np.fft.fft(B, axis=1) * self._H[None, :], axis=1))
        else:
            W = B
        NL = np.zeros_like(At)
        for p, (l_idx, w_idx, coef) in self._terms.items():
            NL[p] = np.einsum('t,tn,tn->n', coef, At[l_idx], W[w_idx])
        return self._nl_prefactor[None, :] * np.fft.fft(NL, axis=1)

    def _rk4ip(self, Af, h):
        E = np.exp(self._L * h / 2)
        AI = E * Af
        k1 = E * (h * self._nonlinear(Af))
        k2 = h * self._nonlinear(AI + k1 / 2)
        k3 = h * self._nonlinear(AI + k2 / 2)
        k4 = h * self._nonlinear(E * (AI + k3))
        return E * (AI + k1 / 6 + k2 / 3 + k3 / 3) + k4 / 6

    # ---------------------------------------------------------------- noise
    def _pump_spectra(self, Af):
        N = self.grid.n_points
        return np.abs(Af[self.pump_mask]) ** 2 / N ** 2  # W per bin, sums to mean power

    def _prepare_mean_noise(self):
        g = self.grid
        N = g.n_points
        pump_modes = [p for p, is_pump in zip(self.prop, self.pump_mask) if is_pump]
        self._S_qp = self.modes.intensity_overlaps(self.noise_modes, pump_modes)
        offsets = (np.arange(2 * N - 1) - (N - 1)) * 2 * np.pi * g.df
        self._k_src = spontaneous_shape(self.raman, offsets, self.temperature)
        self._k_gain = gain_shape(self.raman, offsets)
        omega = g.omega
        self._guided_q = np.asarray([self.modes.is_guided(q, omega) for q in self.noise_modes])
        self._alpha_q = np.where(self._guided_q,
                                 np.asarray([self._alpha(q) for q in self.noise_modes])[:, None],
                                 2 * _UNGUIDED_LOSS)
        self._psd = np.zeros((len(self.noise_modes), N))

    def _convolve(self, kernel, pw):
        N = self.grid.n_points
        full = fftconvolve(np.fft.fftshift(pw), kernel)
        return np.fft.ifftshift(full[N - 1:2 * N - 1])

    def _mean_noise_step(self, pw_start, pw_end, h):
        omega = self.grid.omega
        coupling = self.n2 * omega / c
        pw = 0.5 * (pw_start + pw_end)
        src = np.zeros_like(self._psd)
        gain = np.zeros_like(self._psd)
        for j in range(pw.shape[0]):
            src += np.outer(self._S_qp[:, j], self._convolve(self._k_src, pw[j]))
            gain += np.outer(self._S_qp[:, j], self._convolve(self._k_gain, pw[j]))
        src *= hbar * omega * coupling
        kappa = gain * coupling - self._alpha_q
        x = kappa * h
        growth = np.where(np.abs(x) < 1e-12, h, np.expm1(x) / np.where(kappa == 0, 1.0, kappa))
        self._psd = self._psd * np.exp(x) + src * growth

    def _stochastic_noise_step(self, Af, h):
        g = self.grid
        omega, Omega, N = g.omega, g.Omega, g.n_points
        At = np.fft.ifft(Af, axis=1)
        pump_rows = np.flatnonzero(self.pump_mask)
        S = self.modes.intensity_overlaps(self.prop, [self.prop[j] for j in pump_rows])
        for col, j in enumerate(pump_rows):
            pw = np.abs(Af[j]) ** 2
            if pw.sum() == 0:
                continue
            centre = np.sum(Omega * pw) / pw.sum()
            shape = (hbar * omega * (self.n2 * omega / c)
                     * spontaneous_shape(self.raman, Omega - centre, self.temperature))
            drive = np.sqrt(np.abs(At[j]) ** 2 * h / g.dt)
            for q in range(len(self.prop)):
                xi = (self.rng.standard_normal(N) + 1j * self.rng.standard_normal(N)) / np.sqrt(2)
                Af[q] += np.fft.fft(xi * drive) * np.sqrt(shape * S[q, col])
        return Af

    # ---------------------------------------------------------------- driver
    def propagate(self, A0, length, dz, z_save=None):
        """Propagate launch fields A0 (K, N) (time domain, sqrt(W)) over `length` (m) with
        steps of at most `dz` (m). Fields (and the mean noise PSD) are stored at each
        position in `z_save` (default: the output only)."""
        A0 = np.asarray(A0, dtype=complex)
        if A0.shape != (len(self.prop), self.grid.n_points):
            raise ValueError(f'A0 must have shape ({len(self.prop)}, {self.grid.n_points})')
        z_save = np.unique(np.append(np.asarray([] if z_save is None else z_save, float), length))
        if z_save.min() < 0 or z_save.max() > length:
            raise ValueError('z_save must lie within [0, length]')

        self._build_terms(0.1 / dz if self.coherence_tol is None else self.coherence_tol)
        self._L = self._linear_operator(self.prop)
        self._H = self.raman.H(self.grid.Omega) if self.raman is not None else None
        self._nl_prefactor = 1j * self.n2 / c * (self.grid.omega if self.self_steepening
                                                  else np.full(self.grid.n_points, self.modes.omega0))
        if self.noise == 'mean':
            if self.raman is None:
                raise ValueError("noise='mean' requires a Raman model")
            self._prepare_mean_noise()

        Af = np.fft.fft(A0, axis=1)
        fields, psds, z = [], [], 0.0
        for target in z_save:
            n_steps = int(np.ceil((target - z) / dz - 1e-9))
            h = (target - z) / n_steps if n_steps else 0.0
            for _ in range(n_steps):
                pw_start = self._pump_spectra(Af) if self.noise == 'mean' else None
                Af = self._rk4ip(Af, h)
                if self.noise == 'mean':
                    self._mean_noise_step(pw_start, self._pump_spectra(Af), h)
                elif self.noise == 'stochastic':
                    Af = self._stochastic_noise_step(Af, h)
            z = target
            fields.append(np.fft.ifft(Af, axis=1))
            if self.noise == 'mean':
                psds.append(self._psd.copy())

        return PropagationResult(
            grid=self.grid, z=z_save, labels=[self.modes.labels[p] for p in self.prop],
            fields=np.asarray(fields),
            noise_labels=[self.modes.labels[q] for q in self.noise_modes] if self.noise == 'mean' else [],
            noise_psd=np.asarray(psds) if self.noise == 'mean' else None)
