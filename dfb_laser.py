import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import welch
from dataclasses import dataclass, field


# ── Physical constants ──────────────────────────────────────────────────────────
q = 1.602e-19       # electron charge (C)
h = 6.626e-34       # Planck's constant (J·s)
c = 3e8              # speed of light (m/s)


@dataclass
class DFBLaserParams:
    """Parameters for a 1550 nm InGaAsP/InP DFB laser."""

    # Wavelength
    lambda0: float = 1550e-9          # emission wavelength (m)

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

    # Carrier recombination
    A: float = 1e8                    # non-radiative recombination (s⁻¹)
    B: float = 1e-16                  # radiative recombination (m³/s)
    C: float = 3e-41                  # Auger recombination (m⁶/s)

    # Loss
    alpha_i: float = 2000            # internal loss (m⁻¹)
    R1: float = 0.32                  # facet reflectivity (front)
    R2: float = 0.95                  # facet reflectivity (rear, HR coated)

    # Spontaneous emission
    beta_sp: float = 1e-4            # spontaneous emission coupling factor

    # Linewidth enhancement
    alpha_H: float = 3.0             # Henry alpha parameter

    # Derived quantities (computed in __post_init__)
    V: float = field(init=False)
    v_g: float = field(init=False)
    alpha_m: float = field(init=False)
    tau_p: float = field(init=False)
    nu0: float = field(init=False)

    def __post_init__(self):
        self.V = self.L * self.w * self.d                        # active volume
        self.v_g = c / self.n_g                                  # group velocity
        self.alpha_m = (1 / (2 * self.L)) * np.log(1 / (self.R1 * self.R2))  # mirror loss
        self.tau_p = 1 / (self.v_g * (self.alpha_i + self.alpha_m))           # photon lifetime
        self.nu0 = c / self.lambda0                              # optical frequency

    def carrier_lifetime(self, N):
        """Carrier lifetime including all recombination terms."""
        return 1 / (self.A + self.B * N + self.C * N**2)

    def gain(self, N, S):
        """Material gain with compression: g(N, S)."""
        return self.a * (N - self.N_tr) / (1 + self.epsilon * S)

    def threshold_current(self):
        """Estimate threshold current (A)."""
        g_th = (self.alpha_i + self.alpha_m) / self.Gamma
        N_th = self.N_tr + g_th / self.a
        tau_n = self.carrier_lifetime(N_th)
        return q * self.V * N_th / tau_n

    def output_power(self, S):
        """Output power from front facet (W)."""
        eta_i = 0.8  # internal quantum efficiency
        frac_front = (1 - self.R1) / ((1 - self.R1) + (1 - self.R2))
        return eta_i * frac_front * h * self.nu0 * self.V * S / self.tau_p


# ── Rate equations ──────────────────────────────────────────────────────────────

def rate_equations(t, y, params, I_func):
    """3-equation DFB laser rate equations: dN/dt, dS/dt, dphi/dt."""
    N, S, phi = y
    S = max(S, 0)  # prevent negative photon density

    I = I_func(t)
    g = params.gain(N, S)
    R_sp = params.A * N + params.B * N**2 + params.C * N**3

    dNdt = I / (q * params.V) - R_sp - params.Gamma * params.v_g * g * S
    dSdt = (params.Gamma * params.v_g * g - 1 / params.tau_p) * S \
           + params.beta_sp * params.B * N**2
    dphidt = 0.5 * params.alpha_H * (params.Gamma * params.v_g * params.a * (N - params.N_tr)
                                       - 1 / params.tau_p)

    return [dNdt, dSdt, dphidt]


def solve_transient(params, I_func, t_span, t_eval=None, y0=None):
    """Solve the rate equations for an arbitrary current waveform."""
    if y0 is None:
        y0 = [params.N_tr, 1e10, 0.0]  # small photon seed
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 5000)

    sol = solve_ivp(
        rate_equations, t_span, y0,
        args=(params, I_func),
        t_eval=t_eval,
        method='RK45',
        rtol=1e-9, atol=1e-12,
        max_step=(t_span[1] - t_span[0]) / 1000,
    )
    return sol


# ── Analysis functions ──────────────────────────────────────────────────────────

def plot_transient(params, I_bias, t_end=5e-9):
    """Step response: turn on laser from 0 to I_bias."""
    I_func = lambda t: I_bias

    sol = solve_transient(params, I_func, [0, t_end])
    t_ns = sol.t * 1e9
    N, S, phi = sol.y

    P = params.output_power(S)
    chirp = np.gradient(phi, sol.t) / (2 * np.pi) * 1e-9  # GHz

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'DFB Laser Transient Response (I = {I_bias*1e3:.1f} mA)')

    axes[0, 0].plot(t_ns, N * 1e-24)
    axes[0, 0].set_ylabel('Carrier density (×10²⁴ m⁻³)')
    axes[0, 0].set_xlabel('Time (ns)')
    axes[0, 0].grid(True)

    axes[0, 1].plot(t_ns, S * 1e-21)
    axes[0, 1].set_ylabel('Photon density (×10²¹ m⁻³)')
    axes[0, 1].set_xlabel('Time (ns)')
    axes[0, 1].grid(True)

    axes[1, 0].plot(t_ns, P * 1e3)
    axes[1, 0].set_ylabel('Output power (mW)')
    axes[1, 0].set_xlabel('Time (ns)')
    axes[1, 0].grid(True)

    axes[1, 1].plot(t_ns, chirp)
    axes[1, 1].set_ylabel('Chirp (GHz)')
    axes[1, 1].set_xlabel('Time (ns)')
    axes[1, 1].grid(True)

    plt.tight_layout()
    return fig


def plot_LI_curve(params, I_max=100e-3, n_points=200):
    """Steady-state L-I characteristic."""
    currents = np.linspace(0, I_max, n_points)
    powers = np.zeros(n_points)

    t_end = 10e-9  # long enough to reach steady state

    for i, I in enumerate(currents):
        sol = solve_transient(params, lambda t, Ic=I: Ic, [0, t_end])
        S_ss = np.mean(sol.y[1, -100:])  # average last samples
        powers[i] = params.output_power(max(S_ss, 0))

    I_th = params.threshold_current()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(currents * 1e3, powers * 1e3, 'b-', linewidth=2)
    ax.axvline(I_th * 1e3, color='r', linestyle='--', label=f'$I_{{th}}$ = {I_th*1e3:.1f} mA')
    ax.set_xlabel('Injection current (mA)')
    ax.set_ylabel('Output power (mW)')
    ax.set_title('L-I Characteristic')
    ax.legend()
    ax.grid(True)
    ax.set_xlim(0, I_max * 1e3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    return fig


def plot_modulation_response(params, I_bias, f_min=1e8, f_max=30e9, n_freqs=80):
    """Small-signal modulation response |H(f)|²."""
    freqs = np.logspace(np.log10(f_min), np.log10(f_max), n_freqs)
    response_dB = np.zeros(n_freqs)

    I_mod = 0.05 * I_bias  # 5% modulation depth
    n_cycles = 10
    t_settle = 5e-9  # time for bias point to settle

    # First, get steady-state at bias
    sol_ss = solve_transient(params, lambda t: I_bias, [0, t_settle])
    y0_ss = [sol_ss.y[0, -1], sol_ss.y[1, -1], sol_ss.y[2, -1]]
    S_dc = y0_ss[1]

    for i, f in enumerate(freqs):
        T = 1 / f
        t_sim = n_cycles * T
        n_pts = max(2000, int(t_sim * f * 50))
        t_eval = np.linspace(0, t_sim, n_pts)

        I_func = lambda t, freq=f: I_bias + I_mod * np.sin(2 * np.pi * freq * t)
        sol = solve_transient(params, I_func, [0, t_sim], t_eval=t_eval, y0=y0_ss)

        # Use last few cycles for response
        S = sol.y[1]
        n_last = len(S) // 2  # second half (settled)
        S_ac = S[n_last:] - np.mean(S[n_last:])

        # Extract amplitude at modulation frequency via FFT
        dt = sol.t[1] - sol.t[0]
        fft_S = np.abs(np.fft.rfft(S_ac))
        fft_freqs = np.fft.rfftfreq(len(S_ac), dt)

        idx_f = np.argmin(np.abs(fft_freqs - f))
        response_dB[i] = 20 * np.log10(fft_S[idx_f] / max(fft_S[1], 1e-30))

    # Normalize to DC response
    response_dB -= response_dB[0]

    # Find 3-dB bandwidth
    idx_3dB = np.where(response_dB <= -3)[0]
    f_3dB = freqs[idx_3dB[0]] if len(idx_3dB) > 0 else freqs[-1]

    # Find resonance peak
    idx_peak = np.argmax(response_dB)
    f_r = freqs[idx_peak]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(freqs * 1e-9, response_dB, 'b-', linewidth=2)
    ax.axhline(-3, color='r', linestyle='--', alpha=0.7, label=f'−3 dB → f₃dB = {f_3dB*1e-9:.1f} GHz')
    if idx_peak > 0:
        ax.axvline(f_r * 1e-9, color='g', linestyle=':', alpha=0.7, label=f'fᵣ = {f_r*1e-9:.1f} GHz')
    ax.set_xlabel('Modulation frequency (GHz)')
    ax.set_ylabel('Response (dB)')
    ax.set_title(f'Small-Signal Modulation Response (I = {I_bias*1e3:.1f} mA)')
    ax.legend()
    ax.grid(True, which='both', alpha=0.5)
    ax.set_ylim(bottom=-20)

    plt.tight_layout()
    return fig


def plot_RIN(params, I_bias, t_sim=50e-9, dt=1e-12):
    """Relative intensity noise spectrum with Langevin noise sources."""
    t_settle = 5e-9
    sol_ss = solve_transient(params, lambda t: I_bias, [0, t_settle])
    y0 = [sol_ss.y[0, -1], sol_ss.y[1, -1], sol_ss.y[2, -1]]

    N_ss, S_ss = y0[0], y0[1]

    # Manual Euler-Maruyama integration with Langevin noise
    n_steps = int(t_sim / dt)
    t = np.linspace(0, t_sim, n_steps)
    N_arr = np.zeros(n_steps)
    S_arr = np.zeros(n_steps)
    phi_arr = np.zeros(n_steps)
    N_arr[0], S_arr[0], phi_arr[0] = y0

    rng = np.random.default_rng(42)

    for k in range(n_steps - 1):
        Nk, Sk = N_arr[k], max(S_arr[k], 0)
        phik = phi_arr[k]

        g = params.gain(Nk, Sk)
        R_sp = params.A * Nk + params.B * Nk**2 + params.C * Nk**3
        R_st = params.Gamma * params.v_g * g * Sk

        # Deterministic part
        dN = (I_bias / (q * params.V) - R_sp - R_st) * dt
        dS = (R_st - Sk / params.tau_p + params.beta_sp * params.B * Nk**2) * dt
        dphi = 0.5 * params.alpha_H * (params.Gamma * params.v_g * params.a * (Nk - params.N_tr)
                                         - 1 / params.tau_p) * dt

        # Langevin noise strengths
        F_N = np.sqrt(2 * R_sp * dt) * rng.standard_normal()
        F_S = np.sqrt(2 * params.beta_sp * params.B * Nk**2 * dt) * rng.standard_normal()

        N_arr[k + 1] = Nk + dN + F_N
        S_arr[k + 1] = max(Sk + dS + F_S, 0)
        phi_arr[k + 1] = phik + dphi

    # Compute RIN using Welch's method
    S_mean = np.mean(S_arr[len(S_arr)//4:])
    dS_rel = (S_arr[len(S_arr)//4:] - S_mean) / S_mean

    fs = 1 / dt
    nperseg = min(len(dS_rel) // 4, 4096)
    f_rin, psd = welch(dS_rel, fs=fs, nperseg=nperseg, noverlap=nperseg//2)

    RIN_dB = 10 * np.log10(psd + 1e-30)

    fig, ax = plt.subplots(figsize=(8, 5))
    mask = f_rin > 0
    ax.semilogx(f_rin[mask] * 1e-9, RIN_dB[mask], 'b-', linewidth=1)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('RIN (dB/Hz)')
    ax.set_title(f'Relative Intensity Noise (I = {I_bias*1e3:.1f} mA)')
    ax.grid(True, which='both', alpha=0.5)

    plt.tight_layout()
    return fig


def plot_eye_diagram(params, I_bias, I_mod, bitrate=10e9, n_bits=127):
    """Eye diagram for NRZ modulation."""
    # Generate PRBS-7 sequence
    reg = np.array([1, 0, 0, 0, 0, 0, 1], dtype=int)
    bits = np.zeros(n_bits, dtype=int)
    for i in range(n_bits):
        bits[i] = reg[-1]
        feedback = reg[5] ^ reg[6]
        reg = np.roll(reg, 1)
        reg[0] = feedback

    T_bit = 1 / bitrate
    t_rise = T_bit * 0.1  # 10% rise/fall time
    t_total = n_bits * T_bit

    # Build current waveform with finite rise time
    def I_func(t):
        bit_idx = int(t / T_bit) % n_bits
        target = I_bias + I_mod * bits[bit_idx]
        if bit_idx > 0:
            prev = I_bias + I_mod * bits[bit_idx - 1]
        else:
            prev = I_bias + I_mod * bits[-1]
        t_in_bit = t - bit_idx * T_bit
        if t_in_bit < t_rise:
            return prev + (target - prev) * t_in_bit / t_rise
        return target

    # Let it settle at first bit level
    I_settle = I_bias + I_mod * bits[0]
    sol_ss = solve_transient(params, lambda t: I_settle, [0, 5e-9])
    y0 = [sol_ss.y[0, -1], sol_ss.y[1, -1], sol_ss.y[2, -1]]

    # Simulate
    pts_per_bit = 100
    n_pts = n_bits * pts_per_bit
    t_eval = np.linspace(0, t_total, n_pts)
    sol = solve_transient(params, I_func, [0, t_total], t_eval=t_eval, y0=y0)

    P = params.output_power(sol.y[1])
    chirp = np.gradient(sol.y[2], sol.t) / (2 * np.pi) * 1e-9  # GHz

    # Build eye diagram: overlay 2-bit windows
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Eye Diagram ({bitrate*1e-9:.0f} Gb/s NRZ)')

    skip_bits = 5  # skip initial transient bits
    for b in range(skip_bits, n_bits - 2):
        idx_start = b * pts_per_bit
        idx_end = (b + 2) * pts_per_bit
        if idx_end > len(P):
            break
        t_window = np.linspace(0, 2, idx_end - idx_start)  # normalized to bit period
        axes[0].plot(t_window, P[idx_start:idx_end] * 1e3, 'b-', alpha=0.1, linewidth=0.5)
        axes[1].plot(t_window, chirp[idx_start:idx_end], 'r-', alpha=0.1, linewidth=0.5)

    axes[0].set_xlabel('Time (bit periods)')
    axes[0].set_ylabel('Output power (mW)')
    axes[0].set_title('Intensity Eye')
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Time (bit periods)')
    axes[1].set_ylabel('Chirp (GHz)')
    axes[1].set_title('Chirp Eye')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ── Main ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    params = DFBLaserParams()

    I_th = params.threshold_current()
    print(f"DFB Laser Parameters:")
    print(f"  Wavelength:        {params.lambda0*1e9:.0f} nm")
    print(f"  Cavity length:     {params.L*1e6:.0f} µm")
    print(f"  Photon lifetime:   {params.tau_p*1e12:.2f} ps")
    print(f"  Threshold current: {I_th*1e3:.1f} mA")
    print()

    I_bias = 3 * I_th  # operate at 3× threshold

    print(f"Running analyses at I_bias = {I_bias*1e3:.1f} mA ({I_bias/I_th:.1f}× threshold)...")
    print()

    print("1/5  Transient response...")
    plot_transient(params, I_bias)

    print("2/5  L-I curve...")
    plot_LI_curve(params)

    print("3/5  Modulation response...")
    plot_modulation_response(params, I_bias)

    print("4/5  RIN spectrum...")
    plot_RIN(params, I_bias)

    print("5/5  Eye diagram...")
    I_mod = 0.5 * I_th  # modulation amplitude
    plot_eye_diagram(params, I_bias, I_mod, bitrate=10e9)

    print("Done. Showing plots...")
    plt.show()
