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
                    spectrum. With raman=None the Langevin drive is absent and the run is
                    Kerr-only, which is the configuration for isolating spontaneous
                    four-wave mixing from the vacuum seed.

Random linear mode coupling
---------------------------
Real spans also exchange power between modes LINEARLY, through bends, splices and
connectors, with no intensity dependence. Passing `mode_coupling=ModeCoupling(...)`
applies one random unitary per step to the propagated modes,

    U(z) = expm(i K),  K Hermitian,  E|K_pq|^2 = kappa^2 w_pq dz
    w_pq = 1 / (1 + (dbeta0_pq * L_c)^2)

so the coupling is strongest between quasi-degenerate modes and is suppressed as a
Lorentzian in their propagation-constant mismatch, with correlation length L_c
(Marcuse's coupled-power picture). U is exactly unitary, so total power over the
propagated modes is conserved to machine precision, and because K is redrawn every step
the coupled amplitude accumulates as a random walk: coupled POWER grows as kappa^2 w z.

Coupling acts on the propagated modes only, which is the closed subspace that matters
when a classical channel and a QKD channel share a fibre: classical power that lands in
the QKD channel's mode then scatters there with the full same-mode coefficient. Two
consequences to keep in mind. The accumulated mean-noise PSD is not itself redistributed
by U (a second-order effect, since that power is minute), and a mode that starts as a
spectator can acquire real power, so with strong coupling list both modes in `pumps`.
kappa has no datasheet value; calibrate it against a measured coupling length.

Frequency dependence of the mode fields is available through dispersive_overlaps, which
scales every overlap by A_eff(omega0)/A_eff(omega): the modes breathe with wavelength
(+21% in A_eff across 1450-1750 nm in OM3) while their shapes do not, so one scalar
carries the effect (see FibreModes.area_scale). It is off by default.

Spontaneous four-wave mixing follows from vacuum_seed=True with noise='stochastic', which
launches half a photon per mode per bin so the Kerr terms have zero-point field to
amplify; ensemble-average over seeds to read the spontaneous photon flux.

Polarisation is resolved by listing a spatial mode twice with polarisation=['x', 'y', ...]:
cross-polarised Kerr terms then take the exact isotropic-chi(3) factor 2/3, cross-polarised
Raman takes raman_copol_ratio (a placeholder, see the parameter), and D_PMD adds coarse-step
polarisation mode dispersion per spatial mode. A run with all power in one polarisation
reduces to the scalar solver exactly.

    from fiber.grin_modes import FibreModes, DESIGNS
    from fiber.gmmnlse import GMMNLSE, TimeGrid
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm
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


@dataclass(frozen=True)
class ModeCoupling:
    """Random linear mode coupling (bends, splices, connectors); see the module docstring.

    kappa : coupling strength (rad/sqrt(m)); 0 disables coupling
    correlation_length : L_c (m) of the perturbation, setting how fast coupling falls off
        with the propagation-constant mismatch between two modes
    """
    kappa: float = 0.0
    correlation_length: float = 1.0

    def weights(self, beta0):
        """Lorentzian weight w_pq = 1/(1 + (dbeta0 L_c)^2) for propagation constants beta0."""
        d = np.asarray(beta0)[:, None] - np.asarray(beta0)[None, :]
        return 1.0 / (1.0 + (d * self.correlation_length) ** 2)


@dataclass(frozen=True)
class ModeLoss:
    """Per-mode attenuation derived from the mode profiles rather than assumed.

    Two mechanisms on top of a base loss, both keyed to quantities fiber.grin_modes
    computes from the solved fields:

    Differential mode attenuation -- excess loss proportional to the fraction of the
    mode's power beyond the core radius (cladding_power_fraction), since that is the part
    of the field which samples the interface, the cladding and the coating. Higher-order
    modes therefore attenuate faster, as they do in a real MMF.

    Macrobend loss -- Marcuse's exponential in the guidance margin,
    alpha_bend ~ C exp(-2 gamma^3 R / (3 beta^2)) with gamma = sqrt(beta^2 - (n_cl k0)^2),
    so a bend of radius R strips the weakly guided high-order modes first and leaves the
    fundamental essentially untouched. C (bend_prefactor_dB_km) sets the scale; like
    ModeCoupling's kappa it has no datasheet value and should be calibrated against a
    measured bend-loss curve. bend_radius=None is a straight fibre.

    base_dB_km : fundamental-mode attenuation (dB/km)
    dma_dB_km : excess dB/km at unit cladding-power fraction
    bend_radius : bend radius (m), or None
    bend_prefactor_dB_km : macrobend prefactor C (dB/km)
    """
    base_dB_km: float = 0.3
    dma_dB_km: float = 0.0
    bend_radius: float = None
    bend_prefactor_dB_km: float = 1e4

    def dB_km(self, modes, p):
        loss = self.base_dB_km + self.dma_dB_km * modes.cladding_power_fraction(p)
        if self.bend_radius:
            gamma = modes.guidance_margin(p)
            if gamma <= 0:
                return loss + self.bend_prefactor_dB_km
            exponent = -2 * gamma ** 3 * self.bend_radius / (3 * modes.beta0[p] ** 2)
            loss += self.bend_prefactor_dB_km * np.exp(exponent)
        return loss


@dataclass
class PropagationResult:
    grid: TimeGrid
    z: np.ndarray                       # saved positions (m)
    labels: list                        # propagated mode labels
    fields: np.ndarray                  # (Nz, K, N) time-domain fields, sqrt(W)
    noise_labels: list = field(default_factory=list)
    noise_psd: np.ndarray = None        # (Nz, Q, N) mean spontaneous-Raman PSD, W/Hz
    noise_psd_backward: np.ndarray = None   # (Nz, Q, N) counter-propagating PSD at z = 0

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
    alpha_dB_km : float, a dict of label -> dB/km with an optional 'default' key, or a
        ModeLoss deriving each mode's loss from its profile (differential mode attenuation
        and macrobend loss). The default 0.3 is the OM3 C-band figure; pass 0.2 for
        SMF-28 at 1550 nm
    self_steepening : bool
    noise : 'none', 'mean' or 'stochastic'
    polarisation : None for a scalar (co-polarised) run, or one label per propagated field,
        each 'x' or 'y'. Listing the same spatial mode twice, once per label, resolves its
        two polarisations: cross-polarised Kerr terms then carry the exact isotropic factor
        2/3 and cross-polarised Raman carries raman_copol_ratio. Terms that would flip the
        net polarisation are dropped
    raman_copol_ratio : cross- to co-polarised Raman gain (0-1). A representative
        placeholder, not a settled constant -- it depends on detuning; calibrate against a
        measurement if it matters. Default 0.5, as in fiber.polarization
    D_PMD : PMD parameter (ps/sqrt(km)) applied per spatial mode by coarse-step rotation,
        so each mode acquires its own birefringence walk. Default 0
    backward : also accumulate the counter-propagating spontaneous-Raman PSD emerging at
        z = 0, reported as PropagationResult.noise_psd_backward. Requires noise='mean';
        spontaneous scattering does not deplete the pump, so this needs no boundary-value
        iteration (stimulated backward coupling belongs to fiber.brillouin). Default False
    vacuum_seed : seed each propagated mode with half a photon per frequency bin at launch,
        so the Kerr terms can generate spontaneous four-wave mixing and the run carries a
        shot-noise floor. Requires noise='stochastic' (it is a single realisation, to be
        ensemble-averaged); default False
    temperature : K, for the phonon occupation (default 295, the laboratory value used
        by the coexistence and Raman-spectrum studies)
    noise_modes : labels/indices accumulating the mean noise PSD (default: all guided modes)
    mode_coupling : ModeCoupling for random linear coupling among the propagated modes,
        or None (default) for an ideal fibre with no linear coupling
    coherence_tol : rad/m; None uses 0.1/dz
    dispersion_order : None for the full beta(omega) fit, or an int to truncate the Taylor
        series at omega0 (e.g. 3 keeps beta0..beta3)
    dispersive_overlaps : scale every overlap by A_eff(omega0)/A_eff(omega) across the
        grid (see FibreModes.area_scale), so modes breathe with wavelength instead of
        being frozen at the design frequency. Worth ~6% on the Raman PSD at a 13 THz
        Stokes shift in OM3 and more over a supercontinuum-width grid; negligible over a
        few hundred GHz. Default False, which keeps the frozen-mode behaviour
    reference : label/index of the mode defining the co-moving frame
    seed : RNG seed for noise='stochastic'
    """

    def __init__(self, modes, grid, propagate, pumps=None, n2=2.6e-20, raman=LinAgrawal(),
                 alpha_dB_km=0.3, self_steepening=True, noise='none', temperature=295.0,
                 noise_modes=None, mode_coupling=None, coherence_tol=None,
                 dispersion_order=None, dispersive_overlaps=False, vacuum_seed=False,
                 backward=False, polarisation=None, raman_copol_ratio=0.5, D_PMD=0.0,
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
        self.mode_coupling = mode_coupling
        self.dispersive_overlaps = dispersive_overlaps
        if vacuum_seed and noise != 'stochastic':
            raise ValueError("vacuum_seed requires noise='stochastic'")
        if noise == 'stochastic' and raman is None and not vacuum_seed:
            raise ValueError("noise='stochastic' with raman=None adds nothing unless "
                             "vacuum_seed=True; pass a Raman model or the seed")
        self.vacuum_seed = vacuum_seed
        if backward and noise != 'mean':
            raise ValueError("backward=True requires noise='mean'")
        self.backward = backward
        self.pol = None if polarisation is None else list(polarisation)
        if self.pol is not None:
            if len(self.pol) != len(self.prop):
                raise ValueError('polarisation must give one label per propagated field')
            index = {}
            for row, (p, s) in enumerate(zip(self.prop, self.pol)):
                index.setdefault(p, {})[s] = row
            self._pol_pairs = [(v['x'], v['y']) for v in index.values() if {'x', 'y'} <= set(v)]
        self.raman_copol_ratio = raman_copol_ratio
        self.D_PMD_SI = D_PMD * 1e-12 / np.sqrt(1e3)   # ps/sqrt(km) -> s/sqrt(m)
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
        if isinstance(spec, ModeLoss):
            dB = spec.dB_km(self.modes, p)
        elif isinstance(spec, dict):
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
        kerr, raman = self._polarisation_weights(P, L, M, N)
        pairs, w = np.unique(np.stack([M, N], axis=1), axis=0, return_inverse=True)
        self._pair_m, self._pair_n = pairs[:, 0], pairs[:, 1]
        w = w.ravel()
        self._terms = {p: (L[P == p], w[P == p], (s * kerr)[P == p], (s * raman)[P == p])
                       for p in np.unique(P)}
        self.n_terms = int(keep.sum())

    def _polarisation_weights(self, P, L, M, N):
        """Kerr and Raman weights for each tensor term, from the polarisation labels.

        Scalar runs (no polarisation axis) weight everything 1, reproducing the previous
        behaviour exactly. With polarisation resolved, an index set that is entirely
        co-polarised keeps its full coefficient; a term mixing the two polarisations takes
        the isotropic-chi(3) factor 2/3 on the Kerr part -- exact, from the symmetry of the
        silica tensor, the same factor fiber.polarization uses and its test verifies -- and
        raman_copol_ratio on the Raman part, which is NOT a settled constant but a
        representative placeholder (it depends on detuning), inherited from that module."""
        ones = np.ones(len(P))
        if self.pol is None:
            return ones, ones
        pol = np.asarray(self.pol)
        same = (pol[P] == pol[L]) & (pol[M] == pol[N])
        cross = (pol[P] == pol[L]) & (pol[M] != pol[N])
        allowed = same | cross          # terms that do not flip net polarisation
        kerr = np.where(same & (pol[P] == pol[M]), 1.0,
                        np.where(same, 2.0 / 3.0, 0.0))
        raman = np.where(same & (pol[P] == pol[M]), 1.0,
                         np.where(same, self.raman_copol_ratio, 0.0))
        return kerr * allowed, raman * allowed

    # ---------------------------------------------------------------- operators
    def _nonlinear(self, Af):
        At = np.fft.ifft(Af, axis=1)
        B = At[self._pair_m] * np.conj(At[self._pair_n])
        if self.raman is not None:
            # Kerr and Raman parts are weighted separately, since co- and cross-polarised
            # terms scale differently (2/3 vs raman_copol_ratio).
            Bk = (1 - self.raman.f_R) * B
            Br = self.raman.f_R * np.fft.ifft(np.fft.fft(B, axis=1) * self._H[None, :], axis=1)
        else:
            Bk, Br = B, None
        NL = np.zeros_like(At)
        for p, (l_idx, w_idx, ck, cr) in self._terms.items():
            NL[p] = np.einsum('t,tn,tn->n', ck, At[l_idx], Bk[w_idx])
            if Br is not None:
                NL[p] += np.einsum('t,tn,tn->n', cr, At[l_idx], Br[w_idx])
        return self._nl_prefactor[None, :] * np.fft.fft(NL, axis=1)

    def _rk4ip(self, Af, h):
        E = np.exp(self._L * h / 2)
        AI = E * Af
        k1 = E * (h * self._nonlinear(Af))
        k2 = h * self._nonlinear(AI + k1 / 2)
        k3 = h * self._nonlinear(AI + k2 / 2)
        k4 = h * self._nonlinear(E * (AI + k3))
        return E * (AI + k1 / 6 + k2 / 3 + k3 / 3) + k4 / 6

    # ---------------------------------------------------------------- polarisation
    def _pmd_step(self, Af, h):
        """Coarse-step PMD, per spatial mode: a Haar-random SU(2) rotation of each mode's
        Jones vector followed by a fixed local DGD of D_PMD*sqrt(h) split symmetrically
        between its eigen-axes. Independent draws per step make the accumulated DGD a
        random walk, giving RMS(DGD) ~ D_PMD*sqrt(L) -- the same construction, and the
        same calibration, as fiber.polarization.PolarizationPropagator."""
        if self.D_PMD_SI <= 0:
            return Af
        Omega = self.grid.Omega
        dgd = self.D_PMD_SI * np.sqrt(h)
        for rows in self._pol_pairs:
            a, b = rows
            r = self.rng.standard_normal(4)
            alpha_c, beta_c = r[0] + 1j * r[1], r[2] + 1j * r[3]
            norm = np.sqrt(abs(alpha_c) ** 2 + abs(beta_c) ** 2)
            alpha_c, beta_c = alpha_c / norm, beta_c / norm
            Ax, Ay = Af[a].copy(), Af[b].copy()
            Af[a] = (alpha_c * Ax - np.conj(beta_c) * Ay) * np.exp(-1j * (dgd / 2) * Omega)
            Af[b] = (beta_c * Ax + np.conj(alpha_c) * Ay) * np.exp(1j * (dgd / 2) * Omega)
        return Af

    # ---------------------------------------------------------------- linear coupling
    def _coupling_step(self, Af, h):
        """One random unitary rotation among the propagated modes. K is built from
        independent upper-triangle draws so E|K_pq|^2 = kappa^2 w_pq h exactly."""
        K = len(self.prop)
        sigma = self._cpl_sigma * np.sqrt(h)
        re = self.rng.standard_normal((K, K))
        im = self.rng.standard_normal((K, K))
        upper = (re + 1j * im) / np.sqrt(2) * sigma
        iu = np.triu_indices(K, 1)
        M = np.zeros((K, K), complex)
        M[iu] = upper[iu]
        M = M + M.conj().T
        np.fill_diagonal(M, np.diag(re) * np.diag(sigma))   # real diagonal: random phase
        return expm(1j * M) @ Af

    # ---------------------------------------------------------------- noise
    def _pump_spectra(self, Af):
        N = self.grid.n_points
        return np.abs(Af[self.pump_mask]) ** 2 / N ** 2  # W per bin, sums to mean power

    def _prepare_backward_noise(self):
        """State for the counter-propagating spontaneous-Raman PSD.

        Backward noise generated at z travels back to the input, so it is attenuated over
        z rather than over the remaining L - z. Because spontaneous scattering does not
        deplete the pump, no boundary-value iteration is needed: accumulating
        exp(-alpha z) * source(z) dz during the same forward march gives the PSD emerging
        at z = 0 exactly. (Depletive backward coupling -- stimulated Brillouin above
        threshold -- is a genuine two-point problem and stays with
        fiber.brillouin.BrillouinPropagator, which shoots on the boundary value.)

        In the lossless-pump limit this reproduces the textbook forms: forward noise grows
        as L exp(-alpha L) and peaks at L = 1/alpha, while backward noise saturates as
        (1 - exp(-2 alpha L)) / (2 alpha)."""
        self._psd_bwd = np.zeros((len(self.noise_modes), self.grid.n_points))
        self._z_marched = 0.0

    def _backward_noise_step(self, pw_start, pw_end, h):
        """One step of the backward PSD, using the same source term as the forward update
        but weighted by the round-trip attenuation back to the input."""
        omega = self.grid.omega
        coupling = self.n2 * omega / c * self._area_scale
        pw = 0.5 * (pw_start + pw_end)
        src = np.zeros_like(self._psd_bwd)
        for j in range(pw.shape[0]):
            src += np.outer(self._S_qp[:, j], self._convolve(self._k_src, pw[j]))
        src *= hbar * omega * coupling
        z_mid = self._z_marched + 0.5 * h
        self._psd_bwd += src * np.exp(-self._alpha_q * z_mid) * h
        self._z_marched += h

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
        coupling = self.n2 * omega / c * self._area_scale
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

    def _vacuum_seed(self):
        """One half photon per mode per frequency bin, with random phase: the vacuum
        fluctuations that seed spontaneous four-wave mixing (and set the shot-noise floor).

        A classical envelope carries no zero-point field, so without this the Kerr terms
        have nothing to amplify and the solver produces FWM only where a real idler was
        launched. Seeding half a photon per bin lets the same deterministic terms generate
        spontaneous photon pairs, which is what a populated DWDM grid does around a QKD
        channel. Added once at launch, in the frequency domain. Normalisation: a bin's PSD
        is |Af|^2 dt/N (see PropagationResult.field_psd), so its photon number is
        PSD*df/(hbar*omega) = |Af|^2/(N^2 hbar omega) using dt*df = 1/N -- half a photon is
        therefore |Af|^2 = hbar*omega*N^2/2. The factor N is the same one whose omission
        previously made the frequency-domain Langevin noise N times too weak."""
        g = self.grid
        amp = np.sqrt(hbar * np.abs(g.omega) / 2) * g.n_points
        phase = np.exp(2j * np.pi * self.rng.random((len(self.prop), g.n_points)))
        guided = np.asarray([self.modes.is_guided(p, g.omega) for p in self.prop])
        return np.where(guided, amp[None, :] * phase, 0.0)

    def _stochastic_noise_step(self, Af, h):
        if self.raman is None:
            return Af          # Kerr-only run: the vacuum seed is the whole noise source
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
            shape = (hbar * omega * (self.n2 * omega / c) * self._area_scale
                     * spontaneous_shape(self.raman, Omega - centre, self.temperature))
            drive = np.sqrt(np.abs(At[j]) ** 2 * h / g.dt)
            for q in range(len(self.prop)):
                xi = (self.rng.standard_normal(N) + 1j * self.rng.standard_normal(N)) / np.sqrt(2)
                Af[q] += np.fft.fft(xi * drive) * np.sqrt(shape * S[q, col])
        return Af

    # ---------------------------------------------------------------- stepping
    def _advance(self, Af, h, stepped=None):
        """One accepted step of length h: deterministic RK4IP, then linear mode coupling
        and the spontaneous-noise update, each applied exactly once per step. `stepped`
        passes in an already-computed deterministic step -- the finer solution from the
        error controller -- so the adaptive path keeps the better field instead of
        recomputing the coarse one."""
        pw_start = self._pump_spectra(Af) if self.noise == 'mean' else None
        Af = self._rk4ip(Af, h) if stepped is None else stepped
        if self._coupled:
            Af = self._coupling_step(Af, h)
        if self.pol is not None:
            Af = self._pmd_step(Af, h)
        if self.noise == 'mean':
            pw_end = self._pump_spectra(Af)
            self._mean_noise_step(pw_start, pw_end, h)
            if self.backward:
                self._backward_noise_step(pw_start, pw_end, h)
        elif self.noise == 'stochastic':
            Af = self._stochastic_noise_step(Af, h)
        return Af

    def _rk4ip_with_error(self, Af, h):
        """Step-doubling: one step of h against two of h/2. Returns the finer solution and
        a relative local-error estimate. RK4IP is fourth order, so the error scales as
        h^5 and the controller below uses the exponent 1/5."""
        coarse = self._rk4ip(Af, h)
        fine = self._rk4ip(self._rk4ip(Af, h / 2), h / 2)
        return fine, float(np.linalg.norm(fine - coarse) / (np.linalg.norm(fine) + 1e-300))

    def _march(self, Af, z, target, dz, adaptive):
        """Advance from z to target. Fixed steps when `adaptive` is None; otherwise
        step-doubling error control with `dz` as the maximum step."""
        if adaptive is None:
            n_steps = int(np.ceil((target - z) / dz - 1e-9))
            h = (target - z) / n_steps if n_steps else 0.0
            for _ in range(n_steps):
                Af = self._advance(Af, h)
                self.n_steps += 1
            return Af

        h_min = dz * 1e-6
        while z < target - 1e-9 * max(1.0, abs(target)):
            h = min(self._h_next, target - z)
            while True:
                fine, err = self._rk4ip_with_error(Af, h)
                if err <= adaptive or h <= h_min:
                    break
                h = max(h_min, min(0.9 * h * (adaptive / err) ** 0.2, target - z))
                self.n_rejected += 1
            Af = self._advance(Af, h, stepped=fine)
            z += h
            self.n_steps += 1
            grow = 5.0 if err <= 0 else min(5.0, max(0.2, 0.9 * (adaptive / err) ** 0.2))
            self._h_next = min(dz, h * grow)
        return Af

    # ---------------------------------------------------------------- driver
    def propagate(self, A0, length, dz, z_save=None, adaptive=None):
        """Propagate launch fields A0 (K, N) (time domain, sqrt(W)) over `length` (m) with
        steps of at most `dz` (m). Fields (and the mean noise PSD) are stored at each
        position in `z_save` (default: the output only).

        adaptive : None for fixed steps of dz (the default), or a relative local-error
            target (e.g. 1e-8) for step-doubling error control. `dz` is then the MAXIMUM
            step, never exceeded, so the tensor-term selection built from it stays valid;
            steps only shrink where the field demands it. Mode coupling and the noise
            update are applied once per accepted step, so their statistics are unchanged.
            After the call, `n_steps` and `n_rejected` report the work done."""
        A0 = np.asarray(A0, dtype=complex)
        if A0.shape != (len(self.prop), self.grid.n_points):
            raise ValueError(f'A0 must have shape ({len(self.prop)}, {self.grid.n_points})')
        z_save = np.unique(np.append(np.asarray([] if z_save is None else z_save, float), length))
        if z_save.min() < 0 or z_save.max() > length:
            raise ValueError('z_save must lie within [0, length]')

        self._build_terms(0.1 / dz if self.coherence_tol is None else self.coherence_tol)
        self._L = self._linear_operator(self.prop)
        self._H = self.raman.H(self.grid.Omega) if self.raman is not None else None
        self._area_scale = (self.modes.area_scale(self.grid.omega) if self.dispersive_overlaps
                            else np.ones(self.grid.n_points))
        self._nl_prefactor = (1j * self.n2 / c * self._area_scale
                              * (self.grid.omega if self.self_steepening
                                 else np.full(self.grid.n_points, self.modes.omega0)))
        if self.noise == 'mean':
            if self.raman is None:
                raise ValueError("noise='mean' requires a Raman model")
            self._prepare_mean_noise()
            if self.backward:
                self._prepare_backward_noise()
        self._coupled = self.mode_coupling is not None and self.mode_coupling.kappa > 0
        if self._coupled:
            self._cpl_sigma = (self.mode_coupling.kappa
                               * np.sqrt(self.mode_coupling.weights(self.modes.beta0[self.prop])))

        Af = np.fft.fft(A0, axis=1)
        if self.vacuum_seed:
            Af = Af + self._vacuum_seed()
        fields, psds, psds_bwd, z = [], [], [], 0.0
        self.n_steps, self.n_rejected, self._h_next = 0, 0, dz
        for target in z_save:
            Af = self._march(Af, z, target, dz, adaptive)
            z = target
            fields.append(np.fft.ifft(Af, axis=1))
            if self.noise == 'mean':
                psds.append(self._psd.copy())
                if self.backward:
                    psds_bwd.append(self._psd_bwd.copy())

        return PropagationResult(
            grid=self.grid, z=z_save, labels=[self.modes.labels[p] for p in self.prop],
            fields=np.asarray(fields),
            noise_labels=[self.modes.labels[q] for q in self.noise_modes] if self.noise == 'mean' else [],
            noise_psd=np.asarray(psds) if self.noise == 'mean' else None,
            noise_psd_backward=np.asarray(psds_bwd) if self.backward else None)
