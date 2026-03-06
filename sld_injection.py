"""
SLD (Superluminescent Diode) traveling-wave model with coherent optical
injection into a DFB laser.

Models:
  1. SLD — sectioned traveling-wave amplifier with forward/backward ASE
  2. Injection — Lang-Kobayashi coherent injection terms added to the
     DFB laser rate equations (dS/dt, dphi/dt)

Requires: numpy, scipy, matplotlib
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field

from dfb_laser import DFBLaserParams, solve_transient, q, h, c


# ── SLD parameters ────────────────────────────────────────────────────────────

@dataclass
class SLDParams:
    """Parameters for a broadband InGaAsP/InP superluminescent diode."""

    # Wavelength (configurable — set to 1550 nm to match DFB, or 1300 nm)
    lambda0: float = 1550e-9          # center emission wavelength (m)

    # Geometry
    L: float = 600e-6                 # waveguide length (m)
    w: float = 3e-6                   # active region width (m)
    d: float = 0.15e-6                # active layer thickness (m)
    Gamma: float = 0.35               # optical confinement factor

    # Material / gain
    n_g: float = 3.5                  # group refractive index
    a: float = 3.0e-20                # differential gain (m^2)
    N_tr: float = 1.2e24              # transparency carrier density (m^-3)

    # Carrier recombination
    A_nr: float = 1e8                 # non-radiative (SRH) recombination (s^-1)
    B: float = 1e-16                  # radiative recombination (m^3/s)
    C_aug: float = 3e-41              # Auger recombination (m^6/s)

    # Loss
    alpha_i: float = 1500             # internal waveguide loss (m^-1)

    # Facet reflectivities (AR coated for SLD — must be very low)
    R_front: float = 1e-4             # front facet reflectivity
    R_rear: float = 1e-4              # rear facet reflectivity

    # Spontaneous emission
    beta_sp: float = 1e-4             # fraction coupled into guided mode
    delta_nu: float = 8e12            # spontaneous emission bandwidth (Hz)

    # Sectioned model
    K: int = 50                       # number of spatial sections

    # Derived
    V: float = field(init=False)
    v_g: float = field(init=False)
    nu0: float = field(init=False)
    A_cross: float = field(init=False)
    dz: float = field(init=False)
    P_sat: float = field(init=False)   # saturation power (W)

    def __post_init__(self):
        self.V = self.L * self.w * self.d
        self.v_g = c / self.n_g
        self.nu0 = c / self.lambda0
        self.A_cross = self.w * self.d
        self.dz = self.L / self.K
        # Saturation power: P_sat = h*nu*A_cross / (Gamma * a * tau_c)
        # Use tau_c at ~2x transparency as representative carrier lifetime
        tau_c = self.carrier_lifetime(self.N_tr * 2)
        self.P_sat = h * self.nu0 * self.A_cross / (self.Gamma * self.a * tau_c)

    def gain(self, N, P_local=0.0):
        """Material gain g(N) in m^-1 with power-based saturation.

        Parameters
        ----------
        N : carrier density (m^-3)
        P_local : local optical power (W) for gain saturation
        """
        g0 = self.a * (N - self.N_tr)
        return g0 / (1 + P_local / self.P_sat)

    def R_sp(self, N):
        """Total spontaneous recombination rate (m^-3 s^-1)."""
        return self.A_nr * N + self.B * N**2 + self.C_aug * N**3

    def carrier_lifetime(self, N):
        return 1 / (self.A_nr + self.B * N + self.C_aug * N**2)


# ── Injection parameters ──────────────────────────────────────────────────────

@dataclass
class InjectionParams:
    """Parameters for coherent optical injection from SLD into DFB laser."""

    eta_coupling: float = 0.1         # overall coupling efficiency (0 to 1)
    delta_nu: float = 0.0             # frequency detuning nu_laser - nu_SLD (Hz)

    # Derived (call compute_derived after creating laser params)
    kappa: float = field(init=False, default=0.0)
    delta_omega: float = field(init=False, default=0.0)

    def compute_derived(self, laser: DFBLaserParams):
        """Compute injection rate and angular frequency detuning.

        kappa = (v_g / (2*L)) * sqrt(1 - R1) * eta_coupling
        """
        self.kappa = (laser.v_g / (2 * laser.L)) * np.sqrt(1 - laser.R1) * self.eta_coupling
        self.delta_omega = 2 * np.pi * self.delta_nu


# ── SLD traveling-wave solver ─────────────────────────────────────────────────

def solve_sld_steady_state(sld, I_sld, relax=0.3, max_iter=500, tol=1e-6):
    """
    Solve the SLD sectioned traveling-wave model for steady-state ASE output.

    The waveguide is divided into K sections.  Forward and backward ASE
    powers are propagated through each section accounting for stimulated
    emission, absorption, spontaneous emission, and gain compression.
    Carrier density is updated iteratively until self-consistent.

    Parameters
    ----------
    sld : SLDParams
    I_sld : float — total injection current (A)
    relax : float — under-relaxation factor for carrier update
    max_iter : int
    tol : float — convergence tolerance (relative to N_tr)

    Returns
    -------
    dict with keys: N, P_f, P_b, P_out, z, converged, n_iter
    """
    K = sld.K
    dz = sld.dz

    # Uniform current density
    J = I_sld / (sld.L * sld.w)  # A/m^2
    pump = J / (q * sld.d)       # carrier injection rate (m^-3 s^-1)

    # Initialize carrier density just above transparency
    N = np.full(K, sld.N_tr * 1.2)

    # ASE power arrays: K+1 boundaries (0, dz, 2*dz, ..., L)
    P_f = np.zeros(K + 1)
    P_b = np.zeros(K + 1)

    converged = False
    for iteration in range(max_iter):
        N_prev = N.copy()

        # ── Forward pass (z = 0 → L) ──
        P_f[0] = sld.R_rear * P_b[0]  # rear facet reflection
        for k in range(K):
            # Local optical power for gain saturation
            P_local = max(P_f[k] + 0.5 * (P_b[k] + P_b[k + 1]), 0.0)

            g_net = sld.Gamma * sld.gain(N[k], P_local) - sld.alpha_i
            gdz = g_net * dz
            gdz_clamped = np.clip(gdz, -40, 40)

            # Spontaneous emission power coupled into guided mode per section
            P_sp = sld.Gamma * sld.beta_sp * sld.B * N[k]**2 * h * sld.nu0 * sld.A_cross * dz

            # Amplification factor
            exp_gdz = np.exp(gdz_clamped)
            if abs(gdz) > 1e-8:
                amp_factor = (exp_gdz - 1) / gdz
            else:
                amp_factor = 1.0 + 0.5 * gdz  # Taylor expansion

            P_f[k + 1] = P_f[k] * exp_gdz + P_sp * amp_factor
            P_f[k + 1] = max(P_f[k + 1], 0.0)

        # ── Backward pass (z = L → 0) ──
        P_b[K] = sld.R_front * P_f[K]  # front facet reflection
        for k in range(K - 1, -1, -1):
            # Local optical power for gain saturation
            P_local = max(P_b[k + 1] + 0.5 * (P_f[k] + P_f[k + 1]), 0.0)

            g_net = sld.Gamma * sld.gain(N[k], P_local) - sld.alpha_i
            gdz = g_net * dz
            gdz_clamped = np.clip(gdz, -40, 40)

            P_sp = sld.Gamma * sld.beta_sp * sld.B * N[k]**2 * h * sld.nu0 * sld.A_cross * dz

            exp_gdz = np.exp(gdz_clamped)
            if abs(gdz) > 1e-8:
                amp_factor = (exp_gdz - 1) / gdz
            else:
                amp_factor = 1.0 + 0.5 * gdz

            P_b[k] = P_b[k + 1] * exp_gdz + P_sp * amp_factor
            P_b[k] = max(P_b[k], 0.0)

        # ── Update carrier density (steady-state rate equation) ──
        for k in range(K):
            # Total optical power at section midpoint
            P_opt = 0.5 * (P_f[k] + P_f[k + 1]) + 0.5 * (P_b[k] + P_b[k + 1])
            P_opt = max(P_opt, 0.0)

            # Stimulated emission rate (power absorbed/emitted per unit volume)
            g_mat = sld.gain(N[k], P_opt)
            R_stim = sld.Gamma * g_mat * P_opt / (h * sld.nu0 * sld.A_cross)

            # Steady-state: pump = R_sp(N) + R_stim → solve for N
            R_rec = sld.R_sp(N[k])
            tau_eff = sld.carrier_lifetime(N[k])
            N_new = N[k] + (pump - R_rec - R_stim) * tau_eff
            N_new = max(N_new, sld.N_tr * 0.1)

            N[k] = N[k] + relax * (N_new - N[k])

        # ── Check convergence ──
        delta = np.max(np.abs(N - N_prev)) / sld.N_tr
        if delta < tol:
            converged = True
            break

    P_out = P_f[K] * (1 - sld.R_front)
    z = np.linspace(0, sld.L, K + 1)

    return {
        'N': N,
        'P_f': P_f,
        'P_b': P_b,
        'P_out': P_out,
        'z': z,
        'converged': converged,
        'n_iter': iteration + 1,
    }


# ── SLD output → injection field ─────────────────────────────────────────────

def sld_to_injection_field(sld, P_sld_out, laser, inj):
    """
    Convert total broadband SLD output power to injected photon density
    inside the DFB laser cavity.

    Only the spectral slice of the SLD within the DFB cavity resonance
    bandwidth contributes to coherent injection.

    Returns (S_inj, phi_inj).
    """
    # DFB cavity resonance bandwidth
    delta_nu_laser = laser.v_g * laser.alpha_m / (2 * np.pi)

    # Fraction of SLD spectrum within DFB resonance
    spectral_fraction = delta_nu_laser / sld.delta_nu

    # Coupled coherent power
    P_coherent = P_sld_out * spectral_fraction * inj.eta_coupling

    # Convert to intracavity photon density: P = h*nu * V * S / tau_p
    S_inj = P_coherent * laser.tau_p / (h * laser.nu0 * laser.V)

    return S_inj, 0.0  # phi_inj = 0 (CW phase reference)


# ── Modified laser rate equations with injection ──────────────────────────────

def rate_equations_injection(t, y, params, I_func, inj, S_inj, phi_inj_func):
    """
    DFB laser rate equations with coherent injection (Lang-Kobayashi).

    dN/dt  = I/(qV) - R_sp - Gamma*v_g*g*S
    dS/dt  = (Gamma*v_g*g - 1/tau_p)*S + beta_sp*B*N^2
             + 2*kappa*sqrt(S*S_inj)*cos(phi - phi_inj)
    dphi/dt = 0.5*alpha_H*(Gamma*v_g*a*(N-N_tr) - 1/tau_p)
              - delta_omega + kappa*sqrt(S_inj/S)*sin(phi - phi_inj)
    """
    N, S, phi = y
    S = max(S, 1e-10)

    I = I_func(t)
    g = params.gain(N, S)
    R_sp = params.A * N + params.B * N**2 + params.C * N**3

    # Injection field
    S_i = S_inj(t) if callable(S_inj) else S_inj
    S_i = max(S_i, 0.0)
    phi_i = phi_inj_func(t) if phi_inj_func is not None else 0.0
    dphi = phi - phi_i

    kappa = inj.kappa
    d_omega = inj.delta_omega

    dNdt = I / (q * params.V) - R_sp - params.Gamma * params.v_g * g * S

    dSdt = (params.Gamma * params.v_g * g - 1 / params.tau_p) * S \
           + params.beta_sp * params.B * N**2 \
           + 2 * kappa * np.sqrt(S * S_i) * np.cos(dphi)

    dphidt = 0.5 * params.alpha_H * (
        params.Gamma * params.v_g * params.a * (N - params.N_tr) - 1 / params.tau_p
    ) - d_omega + kappa * np.sqrt(S_i / S) * np.sin(dphi)

    return [dNdt, dSdt, dphidt]


def solve_transient_injection(params, I_func, inj, S_inj,
                              phi_inj_func=None, t_span=None,
                              t_eval=None, y0=None):
    """Solve modified rate equations with coherent injection."""
    if t_span is None:
        t_span = [0, 20e-9]
    if y0 is None:
        y0 = [params.N_tr, 1e10, 0.0]
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 10000)

    sol = solve_ivp(
        rate_equations_injection, t_span, y0,
        args=(params, I_func, inj, S_inj, phi_inj_func),
        t_eval=t_eval,
        method='RK45',
        rtol=1e-9, atol=1e-12,
        max_step=(t_span[1] - t_span[0]) / 2000,
    )
    return sol


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_sld_output(sld, I_sld):
    """SLD spatial profiles: carrier density, ASE powers, net gain."""
    result = solve_sld_steady_state(sld, I_sld)

    z_mm = result['z'] * 1e3
    z_mid = 0.5 * (z_mm[:-1] + z_mm[1:])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    status = 'converged' if result['converged'] else f'NOT converged ({result["n_iter"]} iters)'
    fig.suptitle(f'SLD Steady State  (I = {I_sld*1e3:.0f} mA, '
                 f'P$_{{out}}$ = {result["P_out"]*1e3:.2f} mW, {status})')

    axes[0, 0].plot(z_mid, result['N'] * 1e-24, 'b-', lw=1.5)
    axes[0, 0].set_ylabel('Carrier density (10$^{24}$ m$^{-3}$)')
    axes[0, 0].set_xlabel('Position (mm)')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(z_mm, result['P_f'] * 1e3, 'r-', lw=1.5, label='Forward')
    axes[0, 1].plot(z_mm, result['P_b'] * 1e3, 'b--', lw=1.5, label='Backward')
    axes[0, 1].set_ylabel('ASE power (mW)')
    axes[0, 1].set_xlabel('Position (mm)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    g_net = sld.Gamma * sld.gain(result['N'], 0.0) - sld.alpha_i
    axes[1, 0].plot(z_mid, g_net * 1e-2, 'g-', lw=1.5)
    axes[1, 0].axhline(0, color='k', ls='--', alpha=0.4)
    axes[1, 0].set_ylabel('Net modal gain (cm$^{-1}$)')
    axes[1, 0].set_xlabel('Position (mm)')
    axes[1, 0].grid(True, alpha=0.3)

    # Total ASE at boundaries
    P_total = result['P_f'] + result['P_b']
    axes[1, 1].plot(z_mm, P_total * 1e3, 'k-', lw=1.5)
    axes[1, 1].set_ylabel('Total ASE power (mW)')
    axes[1, 1].set_xlabel('Position (mm)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, result


def plot_sld_LI_curve(sld, I_max=200e-3, n_points=50):
    """SLD L-I characteristic."""
    currents = np.linspace(0, I_max, n_points)
    powers = np.zeros(n_points)
    for i, I in enumerate(currents):
        r = solve_sld_steady_state(sld, I)
        powers[i] = r['P_out']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(currents * 1e3, powers * 1e3, 'b-', lw=2)
    ax.set_xlabel('Injection current (mA)')
    ax.set_ylabel('Output power (mW)')
    ax.set_title(f'SLD L-I Characteristic ({sld.lambda0*1e9:.0f} nm)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, I_max * 1e3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    return fig


def plot_injection_comparison(laser, I_bias, inj, S_inj, t_end=20e-9):
    """Compare DFB laser free-running vs with optical injection."""
    I_func = lambda t: I_bias
    t_eval = np.linspace(0, t_end, 10000)

    # Free-running
    sol_free = solve_transient(laser, I_func, [0, t_end], t_eval=t_eval)

    # With injection
    sol_inj = solve_transient_injection(
        laser, I_func, inj, S_inj,
        t_span=[0, t_end], t_eval=t_eval,
    )

    t_ns = t_eval * 1e9

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f'DFB Laser: Free-running vs Injected  '
                 f'($\\kappa$ = {inj.kappa:.2e} s$^{{-1}}$, '
                 f'$\\Delta\\nu$ = {inj.delta_nu*1e-9:.1f} GHz, '
                 f'S$_{{inj}}$ = {S_inj:.1e} m$^{{-3}}$)', fontsize=12)

    # Carrier density
    axes[0, 0].plot(t_ns, sol_free.y[0] * 1e-24, 'b-', lw=1.5, label='Free', alpha=0.8)
    axes[0, 0].plot(t_ns, sol_inj.y[0] * 1e-24, 'r--', lw=1.5, label='Injected', alpha=0.8)
    axes[0, 0].set_ylabel('Carrier density (10$^{24}$ m$^{-3}$)')
    axes[0, 0].set_xlabel('Time (ns)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Photon density
    axes[0, 1].semilogy(t_ns, sol_free.y[1], 'b-', lw=1.5, label='Free', alpha=0.8)
    axes[0, 1].semilogy(t_ns, sol_inj.y[1], 'r--', lw=1.5, label='Injected', alpha=0.8)
    axes[0, 1].set_ylabel('Photon density (m$^{-3}$)')
    axes[0, 1].set_xlabel('Time (ns)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Output power
    P_free = laser.output_power(sol_free.y[1])
    P_inj = laser.output_power(sol_inj.y[1])
    axes[1, 0].plot(t_ns, P_free * 1e3, 'b-', lw=1.5, label='Free', alpha=0.8)
    axes[1, 0].plot(t_ns, P_inj * 1e3, 'r--', lw=1.5, label='Injected', alpha=0.8)
    axes[1, 0].set_ylabel('Output power (mW)')
    axes[1, 0].set_xlabel('Time (ns)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Chirp
    chirp_free = np.gradient(sol_free.y[2], sol_free.t) / (2 * np.pi) * 1e-9
    chirp_inj = np.gradient(sol_inj.y[2], sol_inj.t) / (2 * np.pi) * 1e-9
    axes[1, 1].plot(t_ns, chirp_free, 'b-', lw=1.5, label='Free', alpha=0.8)
    axes[1, 1].plot(t_ns, chirp_inj, 'r--', lw=1.5, label='Injected', alpha=0.8)
    axes[1, 1].set_ylabel('Chirp (GHz)')
    axes[1, 1].set_xlabel('Time (ns)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_injection_locking_map(laser, I_bias, S_inj_values, delta_nu_values,
                               t_end=30e-9):
    """
    Injection locking map: sweep detuning and injection strength.
    Color = coefficient of variation of photon density in steady state.
    Low CV = locked, high CV = oscillating/chaotic.
    """
    I_func = lambda t: I_bias
    n_det = len(delta_nu_values)
    n_sinj = len(S_inj_values)
    stability = np.zeros((n_det, n_sinj))

    # Get free-running steady state for initial conditions
    sol_ss = solve_transient(laser, I_func, [0, 10e-9])
    y0_ss = [sol_ss.y[0, -1], sol_ss.y[1, -1], sol_ss.y[2, -1]]

    for i, d_nu in enumerate(delta_nu_values):
        for j, S_i in enumerate(S_inj_values):
            inj_tmp = InjectionParams(eta_coupling=1.0, delta_nu=d_nu)
            inj_tmp.compute_derived(laser)

            sol = solve_transient_injection(
                laser, I_func, inj_tmp, S_i,
                t_span=[0, t_end],
                t_eval=np.linspace(0, t_end, 5000),
                y0=y0_ss,
            )

            # Classify by CV of S in the last third
            S_tail = sol.y[1, len(sol.t) * 2 // 3:]
            S_mean = np.mean(S_tail)
            S_std = np.std(S_tail)
            stability[i, j] = S_std / max(S_mean, 1e-10)

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.pcolormesh(
        S_inj_values, delta_nu_values * 1e-9, stability,
        shading='auto', cmap='viridis_r',
    )
    ax.set_xlabel('Injected photon density S$_{inj}$ (m$^{-3}$)')
    ax.set_ylabel('Frequency detuning (GHz)')
    ax.set_xscale('log')
    ax.set_title('Injection Locking Map  (CV of photon density)')
    plt.colorbar(im, ax=ax, label='Coefficient of variation')
    plt.tight_layout()
    return fig, stability


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    print("=" * 65)
    print("  SLD + Coherent Injection into DFB Laser")
    print("=" * 65)

    # ── 1. SLD simulation ─────────────────────────────────────────────────
    sld = SLDParams()          # 1550 nm default
    I_sld = 150e-3             # 150 mA drive current

    print(f"\n  SLD:  {sld.lambda0*1e9:.0f} nm, L = {sld.L*1e6:.0f} um, "
          f"K = {sld.K} sections")
    print(f"  Drive current: {I_sld*1e3:.0f} mA")

    print("\n  1/5  SLD spatial profiles...")
    fig1, sld_result = plot_sld_output(sld, I_sld)
    fig1.savefig('sld_profiles.png', dpi=150)
    print(f"       P_out = {sld_result['P_out']*1e3:.2f} mW  "
          f"({'converged' if sld_result['converged'] else 'NOT converged'} "
          f"in {sld_result['n_iter']} iters)")
    print("       Saved: sld_profiles.png")

    print("\n  2/5  SLD L-I curve...")
    fig2 = plot_sld_LI_curve(sld)
    fig2.savefig('sld_li_curve.png', dpi=150)
    print("       Saved: sld_li_curve.png")

    # ── 2. Set up DFB laser + injection ───────────────────────────────────
    laser = DFBLaserParams()
    I_th = laser.threshold_current()
    I_bias = 3 * I_th

    inj = InjectionParams(eta_coupling=0.05, delta_nu=5e9)
    inj.compute_derived(laser)

    S_inj, phi_inj = sld_to_injection_field(sld, sld_result['P_out'], laser, inj)

    print(f"\n  DFB:  I_bias = {I_bias*1e3:.1f} mA  ({I_bias/I_th:.0f}x threshold)")
    print(f"  Injection: kappa = {inj.kappa:.2e} s^-1, "
          f"detuning = {inj.delta_nu*1e-9:.1f} GHz")
    print(f"  S_inj = {S_inj:.2e} m^-3  "
          f"(spectral fraction = {laser.v_g * laser.alpha_m / (2*np.pi) / sld.delta_nu:.2e})")

    # ── 3. Injection comparison ───────────────────────────────────────────
    print("\n  3/5  Injection comparison (free-running vs injected)...")
    fig3 = plot_injection_comparison(laser, I_bias, inj, S_inj)
    fig3.savefig('injection_comparison.png', dpi=150)
    print("       Saved: injection_comparison.png")

    # ── 4. Injection locking map ──────────────────────────────────────────
    print("\n  4/5  Injection locking map (this may take a moment)...")
    delta_nu_vals = np.linspace(-20e9, 20e9, 21)
    S_inj_vals = np.logspace(
        np.log10(max(S_inj * 0.01, 1e10)),
        np.log10(S_inj * 100),
        15,
    )
    fig4, stab = plot_injection_locking_map(laser, I_bias, S_inj_vals, delta_nu_vals)
    fig4.savefig('injection_locking_map.png', dpi=150)
    print("       Saved: injection_locking_map.png")

    # ── 5. Strong injection locking ───────────────────────────────────────
    print("\n  5/5  Strong injection locking demo (zero detuning)...")
    inj_strong = InjectionParams(eta_coupling=0.3, delta_nu=0.0)
    inj_strong.compute_derived(laser)
    S_inj_strong = S_inj * 50
    fig5 = plot_injection_comparison(laser, I_bias, inj_strong, S_inj_strong, t_end=30e-9)
    fig5.savefig('injection_locking_strong.png', dpi=150)
    print("       Saved: injection_locking_strong.png")

    plt.close('all')
    print("\n" + "=" * 65)
    print("  Done. All figures saved to working directory.")
    print("=" * 65)
