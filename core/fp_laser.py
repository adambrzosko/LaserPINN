"""
Multi-longitudinal-mode Fabry-Perot laser rate equations.

Extends the single-mode DFB solver to M competing modes sharing one
carrier reservoir.  Each mode has its own photon density S_j, phase φ_j,
and spectral gain position λ_j set by the FP cavity FSR.

Designed for studying phase randomisation under gain switching, with
and without broadband (SLD) injection.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable

# ── Physical constants ─────────────────────────────────────────────────────────
q = 1.602e-19       # electron charge (C)
h = 6.626e-34       # Planck's constant (J·s)
c = 3e8              # speed of light (m/s)
kB = 1.381e-23       # Boltzmann constant (J/K)


@dataclass
class FPLaserParams:
    """Parameters for a multi-mode 1550 nm InGaAsP/InP Fabry-Perot laser."""

    # Wavelength
    lambda0: float = 1550e-9          # centre wavelength (m)

    # Cavity geometry
    L: float = 300e-6                 # cavity length (m)
    w: float = 2e-6                   # active region width (m)
    d: float = 0.2e-6                 # active layer thickness (m)
    Gamma: float = 0.3                # optical confinement factor

    # Material / gain
    n_g: float = 3.7                  # group refractive index
    a: float = 2.5e-20               # differential gain (m²)
    N_tr: float = 1.5e24             # transparency carrier density (m⁻³)
    epsilon: float = 3e-23           # gain compression factor (m³)

    # Gain bandwidth (parabolic/Gaussian profile)
    gain_bw: float = 40e-9           # gain FWHM (m), typ. 30-50 nm for InGaAsP

    # Carrier recombination
    A_nr: float = 1e8                 # non-radiative recombination (s⁻¹)
    B: float = 1e-16                  # radiative recombination (m³/s)
    C_aug: float = 3e-41              # Auger recombination (m⁶/s)

    # Loss
    alpha_i: float = 2000            # internal loss (m⁻¹)
    R1: float = 0.32                  # facet reflectivity (front, cleaved)
    R2: float = 0.32                  # facet reflectivity (rear, cleaved)

    # Spontaneous emission
    beta_sp: float = 1e-4            # spontaneous emission coupling factor

    # Linewidth enhancement
    alpha_H: float = 3.0             # Henry alpha parameter

    # Multi-mode settings
    n_modes: int = 21                # number of longitudinal modes to track

    # Derived quantities
    V: float = field(init=False)
    v_g: float = field(init=False)
    alpha_m: float = field(init=False)
    tau_p: float = field(init=False)
    nu0: float = field(init=False)
    FSR_hz: float = field(init=False)
    FSR_lam: float = field(init=False)   # FSR in metres (SI)

    def __post_init__(self):
        self.V = self.L * self.w * self.d
        self.v_g = c / self.n_g
        self.alpha_m = (1 / (2 * self.L)) * np.log(1 / (self.R1 * self.R2))
        self.tau_p = 1 / (self.v_g * (self.alpha_i + self.alpha_m))
        self.nu0 = c / self.lambda0
        self.FSR_hz = c / (2 * self.n_g * self.L)
        self.FSR_lam = self.lambda0**2 / (2 * self.n_g * self.L)  # metres

    def gain_profile(self, lam_offsets):
        """Spectral gain envelope G(Δλ), normalised to 1 at centre.

        Gaussian with FWHM = gain_bw.
        """
        sigma = self.gain_bw / (2 * np.sqrt(2 * np.log(2)))
        return np.exp(-lam_offsets**2 / (2 * sigma**2))

    def mode_wavelengths(self):
        """Wavelength offsets Δλ_j of each mode from λ₀ (metres)."""
        M = self.n_modes
        j = np.arange(M) - M // 2  # centred on j=0
        return j * self.FSR_lam

    def mode_frequencies(self):
        """Frequency offsets δν_j of each mode from ν₀ (Hz)."""
        M = self.n_modes
        j = np.arange(M) - M // 2
        return j * self.FSR_hz

    def threshold_current(self):
        """Threshold current for the peak-gain mode."""
        g_th = (self.alpha_i + self.alpha_m) / self.Gamma
        N_th = self.N_tr + g_th / self.a
        R_sp = self.A_nr * N_th + self.B * N_th**2 + self.C_aug * N_th**3
        return q * self.V * R_sp

    def output_power_per_mode(self, S_j):
        """Output power from front facet for each mode (W)."""
        eta_i = 0.8
        frac_front = (1 - self.R1) / ((1 - self.R1) + (1 - self.R2))
        return eta_i * frac_front * h * self.nu0 * self.V * S_j / self.tau_p

    def summary(self):
        """Print key parameters."""
        I_th = self.threshold_current()
        print(f"FP Laser Parameters:")
        print(f"  Wavelength:        {self.lambda0*1e9:.0f} nm")
        print(f"  Cavity length:     {self.L*1e6:.0f} µm")
        print(f"  Photon lifetime:   {self.tau_p*1e12:.2f} ps")
        print(f"  FSR:               {self.FSR_hz*1e-9:.1f} GHz  ({self.FSR_lam*1e9:.2f} nm)")
        print(f"  Gain bandwidth:    {self.gain_bw*1e9:.0f} nm")
        print(f"  Modes tracked:     {self.n_modes} ({self.n_modes * self.FSR_lam*1e9:.1f} nm span)")
        print(f"  Threshold current: {I_th*1e3:.1f} mA")
        print(f"  Mirror loss:       {self.alpha_m:.0f} m⁻¹")


# ── Multi-mode stochastic solver ───────────────────────────────────────────────

def solve_fp_stochastic(
    params: FPLaserParams,
    I_func: Callable,
    t_span: tuple,
    dt: float = 0.5e-12,
    y0: Optional[dict] = None,
    S_inj_density: float = 0.0,
    seed: Optional[int] = None,
    carrier_noise_scale: float = 1.0,
):
    """Euler-Maruyama solver for M-mode FP laser, complex-field formulation.

    Tracks the complex field E_j = sqrt(S_j) exp(i phi_j) of each longitudinal
    mode.  Spontaneous emission and broadband (SLD) injection both enter as
    random-phase phasor additions — the mechanism by which injected light
    overwrites the intracavity phase.  A real-valued (S, phi) formulation
    cannot represent this, since adding photon density to S leaves phi
    untouched.

    Modes are coupled through the shared carrier reservoir and through gain
    cross-saturation (the 1 + epsilon * S_total denominator).

    Parameters
    ----------
    params : FPLaserParams
    I_func : callable(t) -> current (A)
    t_span : (t_start, t_end)
    dt : float
        Time step (s). Must resolve the fastest dynamics (~tau_p).
    y0 : dict, optional
        Initial conditions {'N': float, 'E': complex array(M)}.
    S_inj_density : float
        Injected photon DENSITY per mode (m^-3), matching the convention of
        core.sld_injection.  Converted internally to a rate via /tau_p.
        Broadband SLD light is incoherent across modes, so each mode
        receives an independent random phasor.
    seed : int, optional

    Returns
    -------
    result : dict with keys 't', 'N', 'E', 'S', 'phi', 'params'
    """
    rng = np.random.default_rng(seed)
    M = params.n_modes

    # Mode-dependent quantities
    dlam = params.mode_wavelengths()     # Δλ_j (metres, SI)
    G_j = params.gain_profile(dlam)      # gain envelope for each mode
    dnu_j = params.mode_frequencies()    # frequency offset (Hz)
    domega_j = 2 * np.pi * dnu_j         # angular frequency offset (rad/s)

    # Time array
    t_eval = np.arange(t_span[0], t_span[1], dt)
    n_steps = len(t_eval)

    # Storage
    N_arr = np.zeros(n_steps)
    E_arr = np.zeros((n_steps, M), dtype=complex)

    # Initial conditions
    if y0 is None:
        N_arr[0] = params.N_tr
        # small seed with random phase in each mode
        E_arr[0, :] = np.sqrt(1e10) * np.exp(1j * rng.uniform(0, 2 * np.pi, M))
    else:
        N_arr[0] = y0['N']
        E_arr[0, :] = y0['E']

    sqrt_half_dt = np.sqrt(dt / 2)
    S_floor = 1e-10

    # SLD injection rate (density / photon lifetime), per mode
    R_SLD = max(S_inj_density, 0.0) / params.tau_p

    E = E_arr[0, :].copy()

    for k in range(n_steps - 1):
        Nk = N_arr[k]
        S_j = np.abs(E)**2
        S_total = np.sum(S_j)

        I = I_func(t_eval[k])

        # Gain per mode: spectral profile × material gain with cross-saturation
        g_mat = params.a * (Nk - params.N_tr) / (1 + params.epsilon * S_total)
        g_j = g_mat * G_j

        # Recombination
        R_sp = params.A_nr * Nk + params.B * Nk**2 + params.C_aug * Nk**3

        # beta_sp is the fraction coupling into ONE mode (same convention as
        # core/sld_injection.py), so every mode gets the full rate - no /M.
        R_sp_mode = params.beta_sp * params.B * Nk**2

        # ── Field update: complex gain carries the alpha_H phase-amplitude coupling ──
        net_gain_j = 0.5 * (1 + 1j * params.alpha_H) * (
            params.Gamma * params.v_g * g_j - 1 / params.tau_p
        )
        # exact for the linear part; Euler lets alpha_H leak into |E|
        E = E * np.exp(net_gain_j * dt)

        # Mode frequency offsets: each mode rotates at its own rate
        E = E * np.exp(1j * domega_j * dt)

        # ── Random-phase phasor additions ──
        # Spontaneous emission (independent per mode)
        E += np.sqrt(max(R_sp_mode, 0.0)) * sqrt_half_dt * (
            rng.standard_normal(M) + 1j * rng.standard_normal(M)
        )
        # Broadband SLD injection (incoherent across modes)
        if R_SLD > 0:
            E += np.sqrt(R_SLD) * sqrt_half_dt * (
                rng.standard_normal(M) + 1j * rng.standard_normal(M)
            )

        # ── Carrier equation ──
        stim_total = params.Gamma * params.v_g * np.sum(g_j * S_j)
        dN = (I / (q * params.V) - R_sp - stim_total) * dt
        # carrier_noise_scale is a diagnostic knob: pass 1/sqrt(V) to test the
        # density-vs-number normalisation of the carrier Langevin force.
        F_N = carrier_noise_scale * np.sqrt(2 * max(R_sp, 0) * dt) * rng.standard_normal()

        N_arr[k + 1] = Nk + dN + F_N
        E_arr[k + 1, :] = E

    S_arr = np.abs(E_arr)**2
    phi_arr = np.angle(E_arr)

    return {
        't': t_eval,
        'N': N_arr,
        'E': E_arr,        # shape (n_steps, M), complex
        'S': S_arr,        # shape (n_steps, M)
        'phi': phi_arr,    # shape (n_steps, M)
        'params': params,
        'G_j': G_j,
        'dlam': dlam,      # mode wavelength offsets (m)
        'dnu': dnu_j,      # mode frequency offsets (Hz)
    }


# ── Multi-pulse gain-switching driver ──────────────────────────────────────────

def gain_switch_fp(
    params: FPLaserParams,
    f_rep: float,
    n_pulses: int,
    I_off: float,
    I_on: float,
    duty: float = 0.30,
    dt: float = 0.5e-12,
    S_inj_density: float = 0.0,
    seed: Optional[int] = None,
    warmup_pulses: int = 20,
    carrier_noise_scale: float = 1.0,
):
    """Gain-switch the FP laser and return per-pulse, per-mode peak fields.

    Parameters
    ----------
    params : FPLaserParams
    f_rep : float  — repetition rate (Hz)
    n_pulses : int — number of pulses to record (after warmup)
    I_off, I_on : float — bias and peak current (A)
    duty : float — duty cycle
    dt : float — solver time step (s)
    S_inj_density : float
        Injected photon density (m^-3) per mode, matching the convention of
        core.sld_injection.  Broadband SLD light is incoherent across modes,
        so each mode receives an independent random phasor.
    seed : int, optional
    warmup_pulses : int — pulses to discard for initial transient

    Returns
    -------
    result : dict
        'E_peak' : complex array (n_pulses, M) — peak E-field per mode per pulse
        'S_peak' : array (n_pulses, M) — peak photon density per mode
        'phi_peak' : array (n_pulses, M) — phase at peak per mode
        'S_peak_total' : array (n_pulses,) — total peak photon density
        't_peak' : array (n_pulses,) — time of total peak
        'params' : FPLaserParams
    """
    M = params.n_modes
    T_rep = 1 / f_rep
    t_on = duty * T_rep

    total_pulses = warmup_pulses + n_pulses
    t_total = total_pulses * T_rep

    # Raised-cosine current waveform
    t_rise = min(20e-12, t_on / 4)

    def I_func(t):
        t_in_period = t % T_rep
        if t_in_period < t_rise:
            blend = 0.5 * (1 - np.cos(np.pi * t_in_period / t_rise))
            return I_off + (I_on - I_off) * blend
        elif t_in_period < t_on - t_rise:
            return I_on
        elif t_in_period < t_on:
            blend = 0.5 * (1 + np.cos(np.pi * (t_in_period - t_on + t_rise) / t_rise))
            return I_off + (I_on - I_off) * blend
        else:
            return I_off

    # Run full simulation
    sol = solve_fp_stochastic(
        params, I_func, (0, t_total), dt=dt,
        S_inj_density=S_inj_density, seed=seed,
        carrier_noise_scale=carrier_noise_scale,
    )

    t = sol['t']
    S = sol['S']         # (n_steps, M)
    phi = sol['phi']     # (n_steps, M)
    S_total = np.sum(S, axis=1)

    # Extract per-pulse peak fields
    E_peak = np.zeros((n_pulses, M), dtype=complex)
    S_peak = np.zeros((n_pulses, M))
    phi_peak = np.zeros((n_pulses, M))
    S_peak_total = np.zeros(n_pulses)
    t_peak = np.zeros(n_pulses)

    steps_per_period = int(T_rep / dt)

    for p in range(n_pulses):
        pulse_idx = warmup_pulses + p
        i_start = int(pulse_idx * steps_per_period)
        i_end = min(i_start + steps_per_period, len(t))
        if i_end <= i_start:
            break

        # Find peak of total photon density in this period
        S_tot_window = S_total[i_start:i_end]
        i_pk = np.argmax(S_tot_window) + i_start

        S_peak[p, :] = S[i_pk, :]
        phi_peak[p, :] = phi[i_pk, :]
        S_peak_total[p] = S_total[i_pk]
        t_peak[p] = t[i_pk]

        E_peak[p, :] = sol['E'][i_pk, :]

    return {
        'E_peak': E_peak,
        'S_peak': S_peak,
        'phi_peak': phi_peak,
        'S_peak_total': S_peak_total,
        't_peak': t_peak,
        'params': params,
        'sol': sol,
    }


# ── AMZI analysis for multi-mode field ─────────────────────────────────────────

def amzi_splitting_ratio(E_peak, dnu_j, tau_amzi):
    """Compute AMZI splitting ratio η for multi-mode pulses.

    For a field E_n = Σ_j √S_j exp(i(ω_j t + φ_j)) at pulse n, the AMZI
    with delay τ gives:
        E_A = (E_n + E_{n-1} × exp(i ω_j τ)) / √2
        E_B = (E_n - E_{n-1} × exp(i ω_j τ)) / √2

    and η = |E_A|² / (|E_A|² + |E_B|²).

    Parameters
    ----------
    E_peak : complex array (n_pulses, M)
    dnu_j : array (M,) — frequency offsets (Hz)
    tau_amzi : float — AMZI delay (s), should match 1/f_rep for
               consecutive-pulse interference

    Returns
    -------
    eta : array (n_pulses - 1,) — splitting ratio for pulse pairs (n, n-1)
    """
    n_pulses, M = E_peak.shape
    domega_j = 2 * np.pi * dnu_j
    phase_shift = np.exp(1j * domega_j * tau_amzi)  # (M,)

    eta = np.zeros(n_pulses - 1)

    for n in range(1, n_pulses):
        E_n = E_peak[n, :]        # (M,)
        E_prev = E_peak[n-1, :]   # (M,)

        # Each mode acquires a phase from propagating through the delay arm
        E_prev_delayed = E_prev * phase_shift

        # Interfere: sum over modes then combine arms
        E_A_modes = (E_n + E_prev_delayed) / np.sqrt(2)
        E_B_modes = (E_n - E_prev_delayed) / np.sqrt(2)

        I_A = np.abs(np.sum(E_A_modes))**2
        I_B = np.abs(np.sum(E_B_modes))**2

        if I_A + I_B > 0:
            eta[n-1] = I_A / (I_A + I_B)
        else:
            eta[n-1] = 0.5

    return eta


def compute_r1(E_peak):
    """Order parameter r₁ = |⟨exp(iΔφ)⟩| from total field phase.

    Uses the phase of the total (summed) field at each pulse.
    """
    E_total = np.sum(E_peak, axis=1)  # sum over modes → (n_pulses,)
    phases = np.angle(E_total)
    dphases = np.diff(phases)
    return np.abs(np.mean(np.exp(1j * dphases)))


def mode_power_stats(S_peak):
    """Mode partition noise statistics.

    Returns
    -------
    dict with:
        'mean_spectrum' : mean power per mode
        'std_spectrum' : std of power per mode
        'mpn_k' : mode partition noise parameter k_j for each mode
        'dominant_mode' : which mode has peak power for each pulse
    """
    S_total = np.sum(S_peak, axis=1, keepdims=True)
    fractions = S_peak / np.maximum(S_total, 1e-30)

    mean_frac = np.mean(fractions, axis=0)
    std_frac = np.std(fractions, axis=0)

    # Mode partition noise parameter: k_j² = Var(f_j) / (f_j (1 - f_j))
    mpn_k = np.sqrt(std_frac**2 / np.maximum(mean_frac * (1 - mean_frac), 1e-30))

    dominant = np.argmax(S_peak, axis=1)

    return {
        'mean_spectrum': np.mean(S_peak, axis=0),
        'std_spectrum': np.std(S_peak, axis=0),
        'mean_fraction': mean_frac,
        'std_fraction': std_frac,
        'mpn_k': mpn_k,
        'dominant_mode': dominant,
    }


# ── Convenience ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    fp = FPLaserParams()
    fp.summary()

    I_th = fp.threshold_current()
    print(f"\nRunning 1000-pulse gain-switching test at 2 GHz...")
    print(f"  I_off = 0.9 × I_th = {0.9*I_th*1e3:.1f} mA")
    print(f"  I_on  = 3.0 × I_th = {3.0*I_th*1e3:.1f} mA")

    res = gain_switch_fp(
        fp,
        f_rep=2e9,
        n_pulses=1000,
        I_off=0.9 * I_th,
        I_on=3.0 * I_th,
        duty=0.30,
        seed=42,
    )

    r1 = compute_r1(res['E_peak'])
    stats = mode_power_stats(res['S_peak'])

    print(f"\n  r₁ (total field) = {r1:.4f}")
    print(f"  Dominant mode switches: "
          f"{np.sum(np.diff(stats['dominant_mode']) != 0)} / {len(stats['dominant_mode'])-1} pulses")
    print(f"  Peak mode partition noise k = {np.max(stats['mpn_k']):.3f}")
    sorted_spec = np.sort(stats['mean_spectrum'])
    if sorted_spec[-2] > 0:
        print(f"  Mean SMSR = {10*np.log10(sorted_spec[-1]/sorted_spec[-2]):.1f} dB")
    else:
        print(f"  Mean SMSR = inf dB")
