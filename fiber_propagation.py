"""
Fiber propagation of gain-switched pulses: dispersion penalty analysis.

Gain-switched DFB pulses carry strong frequency chirp (alpha_H ~ 3).
In standard SMF-28 fiber (D ~ 17 ps/nm/km at 1550 nm), this chirp
causes temporal broadening.  SLD injection randomises the phase but
also changes the instantaneous spectral content, modifying the
dispersion penalty.

Physics:
  Split-step Fourier method (SSFM) solving the nonlinear Schrodinger
  equation in the slowly-varying envelope approximation:

    dA/dz = -alpha/2*A - j*(beta2/2)*d2A/dt2 + (beta3/6)*d3A/dt3
            + j*gamma*|A|^2*A

  where A(z,t) is the pulse envelope in a co-moving frame.

Strategy:
  1. Generate a gain-switched pulse train (few pulses) at selected
     frequencies using the existing numba solver, but this time store
     the full complex field E(t) = Er(t) + j*Ei(t).
  2. Convert intracavity field to output field (front facet).
  3. Propagate through L km of SMF-28 via SSFM.
  4. Quantify: pulse broadening, peak power reduction, chirp evolution,
     spectral width change.
  5. Compare free-running vs SLD-injected at 2, 5, 10 GHz.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time as _time
from numba import njit

from dfb_laser import DFBLaserParams, q, h, c


# ── Fiber parameters (SMF-28 at 1550 nm) ─────────────────────────────

ALPHA_DB = 0.2          # attenuation (dB/km)
ALPHA = ALPHA_DB / (10 * np.log10(np.e)) / 1e3   # (1/m)  ~ 4.6e-5
D_PS = 17.0             # dispersion (ps/nm/km)
LAMBDA0 = 1550e-9       # center wavelength (m)
# beta2 from D:  beta2 = -D * lambda^2 / (2*pi*c)
BETA2 = -D_PS * 1e-6 * LAMBDA0**2 / (2 * np.pi * c)  # s^2/m
BETA3 = 0.07e-39        # third-order dispersion (s^3/m)  ~0.07 ps^3/km
GAMMA_NL = 1.3e-3       # nonlinear coefficient (1/W/m)
N_EFF = 1.468           # effective index (for Aeff ~ 80 um^2)
A_EFF = 80e-12          # effective mode area (m^2)


# ── Numba solver: full waveform output ────────────────────────────────

@njit
def simulate_pulse_waveform(
        n_pulses, pts_period, dt,
        I_off, I_on, t_on, t_rise,
        V, Gamma, v_g, a_gain, N_tr, eps,
        A, B, C_aug, tau_p, beta_sp, alpha_H, q_e,
        S_inj, seed):
    """Generate full complex field waveform E(t) for n_pulses.

    Returns Er, Ei arrays of shape (n_pulses * pts_period,).
    Also returns N(t) for chirp analysis.
    """
    np.random.seed(seed)
    sqrt_half_dt = np.sqrt(dt / 2.0)

    total_pts = n_pulses * pts_period
    Er_out = np.empty(total_pts)
    Ei_out = np.empty(total_pts)
    N_out = np.empty(total_pts)

    N = N_tr * 1.2
    Er = np.sqrt(1e10)
    Ei = 0.0

    R_SLD = S_inj / tau_p
    amp_sld = np.sqrt(R_SLD) * sqrt_half_dt if R_SLD > 0.0 else 0.0

    idx = 0
    for p in range(n_pulses):
        for k in range(pts_period):
            t = k * dt

            # Current waveform (raised cosine edges)
            if t < t_rise:
                f = 0.5 * (1.0 - np.cos(np.pi * t / t_rise))
                I_cur = I_off + (I_on - I_off) * f
            elif t < t_on - t_rise:
                I_cur = I_on
            elif t < t_on:
                f = 0.5 * (1.0 - np.cos(np.pi * (t_on - t) / t_rise))
                I_cur = I_off + (I_on - I_off) * f
            else:
                I_cur = I_off

            S = Er * Er + Ei * Ei
            if S < 1e-10:
                S = 1e-10

            g = a_gain * (N - N_tr) / (1.0 + eps * S)
            R_sp = A * N + B * N * N + C_aug * N * N * N
            R_sp_mode = beta_sp * B * N * N

            gamma = 0.5 * (Gamma * v_g * g - 1.0 / tau_p)

            Er_new = (1.0 + gamma * dt) * Er - gamma * alpha_H * dt * Ei
            Ei_new = (1.0 + gamma * dt) * Ei + gamma * alpha_H * dt * Er
            Er = Er_new
            Ei = Ei_new

            # Spontaneous emission noise
            amp_sp = np.sqrt(R_sp_mode) * sqrt_half_dt
            Er += amp_sp * np.random.randn()
            Ei += amp_sp * np.random.randn()

            # SLD injection noise
            if amp_sld > 0.0:
                Er += amp_sld * np.random.randn()
                Ei += amp_sld * np.random.randn()

            # Carrier dynamics
            dN = (I_cur / (q_e * V) - R_sp - Gamma * v_g * g * S) * dt
            FN = np.sqrt(2.0 * R_sp * dt) * np.random.randn()
            N += dN + FN

            Er_out[idx] = Er
            Ei_out[idx] = Ei
            N_out[idx] = N
            idx += 1

    return Er_out, Ei_out, N_out


# ── Intracavity → output field conversion ─────────────────────────────

def intracavity_to_output(Er, Ei, laser):
    """Convert intracavity photon density field to output power envelope.

    The intracavity field has units ~ sqrt(photon density).
    Output power:  P = eta_i * frac_front * h*nu * V * S / tau_p

    We want A(t) with |A|^2 in Watts, so:
        A = sqrt(P) * exp(j*phi)
        P = eta_i * frac_front * h*nu * V * (Er^2+Ei^2) / tau_p

    Preserve the phase structure: just scale the complex field.
    """
    eta_i = 0.8
    frac_front = (1 - laser.R1) / ((1 - laser.R1) + (1 - laser.R2))
    scale = np.sqrt(eta_i * frac_front * h * laser.nu0 * laser.V / laser.tau_p)

    A_r = scale * Er
    A_i = scale * Ei
    return A_r + 1j * A_i


# ── Split-step Fourier method ─────────────────────────────────────────

def ssfm_propagate(A, dt, L_fiber, beta2, beta3=0.0, gamma=0.0,
                   alpha=0.0, n_steps=None, step_size=100.0):
    """Propagate envelope A(t) through fiber using symmetric SSFM.

    Parameters
    ----------
    A : complex array — input field envelope, |A|^2 in Watts
    dt : float — time step (s)
    L_fiber : float — fiber length (m)
    beta2 : float — GVD (s^2/m)
    beta3 : float — TOD (s^3/m)
    gamma : float — nonlinear coefficient (1/W/m)
    alpha : float — loss coefficient (1/m)
    n_steps : int — number of spatial steps (overrides step_size)
    step_size : float — spatial step size (m), default 100 m

    Returns
    -------
    A_out : complex array — output field after propagation
    """
    N = len(A)
    if n_steps is None:
        n_steps = max(int(np.ceil(L_fiber / step_size)), 1)
    dz = L_fiber / n_steps

    # Frequency grid
    omega = 2 * np.pi * np.fft.fftfreq(N, d=dt)

    # Linear operator in frequency domain (half-step)
    D_half = np.exp((-alpha / 2 + 1j * beta2 / 2 * omega**2
                     - 1j * beta3 / 6 * omega**3) * dz / 2)

    A_f = np.fft.fft(A)

    for step in range(n_steps):
        # Half linear step
        A_f *= D_half
        A_t = np.fft.ifft(A_f)

        # Full nonlinear step
        if gamma > 0:
            A_t *= np.exp(1j * gamma * np.abs(A_t)**2 * dz)

        # Half linear step
        A_f = np.fft.fft(A_t) * D_half

    A_out = np.fft.ifft(A_f)
    return A_out


# ── Pulse metrics ─────────────────────────────────────────────────────

def pulse_metrics(A, dt, t_center=None, window_fraction=0.8):
    """Extract metrics from a single pulse envelope.

    Returns dict with: peak_power, fwhm, rms_width, energy,
    chirp_bandwidth, spectral_width_3dB, time_bandwidth_product.
    """
    P = np.abs(A)**2
    peak_power = float(np.max(P))
    peak_idx = int(np.argmax(P))

    # Energy
    energy = float(np.sum(P) * dt)

    # FWHM (temporal)
    half_max = peak_power / 2.0
    above = np.where(P >= half_max)[0]
    if len(above) > 1:
        fwhm = float((above[-1] - above[0]) * dt)
    else:
        fwhm = dt

    # RMS width
    t_arr = np.arange(len(A)) * dt
    t_mean = np.sum(t_arr * P) / np.sum(P) if np.sum(P) > 0 else 0
    rms_width = float(np.sqrt(np.sum((t_arr - t_mean)**2 * P) / np.sum(P)))

    # Spectral width
    A_f = np.fft.fftshift(np.fft.fft(A))
    S_f = np.abs(A_f)**2
    f_arr = np.fft.fftshift(np.fft.fftfreq(len(A), d=dt))

    S_f_max = np.max(S_f)
    above_f = np.where(S_f >= S_f_max / 2.0)[0]
    if len(above_f) > 1:
        spec_width_3dB = float(abs(f_arr[above_f[-1]] - f_arr[above_f[0]]))
    else:
        spec_width_3dB = 1.0 / dt

    # RMS spectral width
    f_mean = np.sum(f_arr * S_f) / np.sum(S_f) if np.sum(S_f) > 0 else 0
    rms_spec = float(np.sqrt(np.sum((f_arr - f_mean)**2 * S_f) / np.sum(S_f)))

    # Time-bandwidth product
    tbp = rms_width * rms_spec * 2 * np.pi  # dimensionless

    # Instantaneous frequency (chirp) at peak
    phase = np.unwrap(np.angle(A))
    inst_freq = np.gradient(phase, dt) / (2 * np.pi)
    chirp_at_peak = float(inst_freq[peak_idx])

    return dict(
        peak_power=peak_power,
        fwhm=fwhm,
        rms_width=rms_width,
        energy=energy,
        spec_width_3dB=spec_width_3dB,
        rms_spec=rms_spec,
        tbp=tbp,
        chirp_at_peak=chirp_at_peak,
    )


def extract_single_pulse(A, dt, pts_period, pulse_idx):
    """Extract a single pulse from a multi-pulse waveform."""
    start = pulse_idx * pts_period
    end = (pulse_idx + 1) * pts_period
    return A[start:end]


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 70)
    print("  Fiber Propagation of Gain-Switched Pulses")
    print("=" * 70)

    laser = DFBLaserParams()
    I_th = laser.threshold_current()

    DT = 0.5e-12          # 0.5 ps time step (need fine resolution for chirp)
    DUTY = 0.30
    I_BIAS_FACTOR = 0.9
    I_PEAK_FACTOR = 5.0
    I_off = I_BIAS_FACTOR * I_th
    I_on = I_PEAK_FACTOR * I_th

    # Generate enough pulses to discard transients and have statistics
    N_PULSES = 32
    N_DISCARD = 4          # first 4 settle the carrier reservoir
    PULSE_IDX = 16         # which pulse to use for single-pulse analysis

    freqs = [2e9, 5e9, 10e9]
    fiber_lengths_km = [0, 5, 10, 20, 50]

    # Fiber params
    beta2 = BETA2
    beta3 = BETA3
    gamma_nl = GAMMA_NL
    alpha_fiber = ALPHA

    print(f"\n  Laser: lambda = {laser.lambda0*1e9:.0f} nm, "
          f"alpha_H = {laser.alpha_H}, tau_p = {laser.tau_p*1e12:.2f} ps")
    print(f"  Fiber: D = {D_PS:.1f} ps/nm/km, "
          f"beta2 = {beta2*1e27:.3f} ps^2/km, "
          f"gamma = {gamma_nl*1e3:.1f} /W/km, "
          f"alpha = {ALPHA_DB:.1f} dB/km")
    print(f"  Bias = {I_BIAS_FACTOR}x I_th, Peak = {I_PEAK_FACTOR}x I_th, "
          f"Duty = {DUTY*100:.0f}%")

    # Compile JIT
    print("\n  Compiling JIT solver...")
    t0 = _time.time()
    _w = simulate_pulse_waveform(
        4, 50, DT, I_off, I_on, 15e-12, 5e-12,
        laser.V, laser.Gamma, laser.v_g, laser.a, laser.N_tr, laser.epsilon,
        laser.A, laser.B, laser.C, laser.tau_p, laser.beta_sp, laser.alpha_H, q,
        0.0, 0)
    del _w
    print(f"  Compiled in {_time.time()-t0:.1f}s")

    # ── Storage ──────────────────────────────────────────────────────

    # For each (freq, mode, fiber_length): store pulse metrics
    results = {}

    total_t0 = _time.time()

    for f_rep in freqs:
        T_rep = 1.0 / f_rep
        t_on = DUTY * T_rep
        pts = max(int(round(T_rep / DT)), 100)
        dt = T_rep / pts
        t_rise = min(20e-12, t_on / 4.0)
        fg = f_rep * 1e-9

        for mode, S_inj_val, label in [
            ('freerun', 0.0, 'Free-running'),
            ('sld', 3.0e19, 'SLD-injected'),
        ]:
            seed = 42 + int(fg) * 100 + (1000 if mode == 'sld' else 0)

            print(f"\n  {fg:.0f} GHz, {label} ({pts} pts/period, dt={dt*1e12:.2f} ps)")
            t0 = _time.time()

            Er, Ei, N_carr = simulate_pulse_waveform(
                N_PULSES, pts, dt,
                I_off, I_on, t_on, t_rise,
                laser.V, laser.Gamma, laser.v_g, laser.a,
                laser.N_tr, laser.epsilon,
                laser.A, laser.B, laser.C,
                laser.tau_p, laser.beta_sp, laser.alpha_H, q,
                S_inj_val, seed)

            elapsed = _time.time() - t0
            print(f"    Generated in {elapsed:.2f}s")

            # Convert to output field
            A_full = intracavity_to_output(Er, Ei, laser)

            # Extract representative pulse
            A_pulse = extract_single_pulse(A_full, dt, pts, PULSE_IDX)

            # Zero-pad for spectral resolution
            pad_factor = 4
            n_padded = len(A_pulse) * pad_factor
            A_padded = np.zeros(n_padded, dtype=complex)
            # Center the pulse in the padded window
            offset = (n_padded - len(A_pulse)) // 2
            A_padded[offset:offset + len(A_pulse)] = A_pulse

            # Propagate through different fiber lengths
            key_base = f'{fg:.0f}GHz_{mode}'
            results[key_base] = {}

            for L_km in fiber_lengths_km:
                L_m = L_km * 1e3
                if L_m == 0:
                    A_out = A_padded.copy()
                else:
                    # Step size: max 50 m or L/100
                    step = min(50.0, L_m / 100)
                    A_out = ssfm_propagate(
                        A_padded, dt, L_m, beta2, beta3,
                        gamma_nl, alpha_fiber, step_size=step)

                m = pulse_metrics(A_out, dt)
                results[key_base][L_km] = dict(
                    metrics=m,
                    A=A_out,
                    dt=dt,
                )
                print(f"    L={L_km:3d} km: FWHM={m['fwhm']*1e12:.1f} ps, "
                      f"P_pk={m['peak_power']*1e3:.2f} mW, "
                      f"BW={m['spec_width_3dB']*1e-9:.1f} GHz, "
                      f"TBP={m['tbp']:.2f}")

    total_elapsed = _time.time() - total_t0
    print(f"\n  Total: {total_elapsed:.1f}s")

    # ── Figure 1: Pulse temporal evolution through fiber ──────────────

    fig1, axes1 = plt.subplots(2, 3, figsize=(18, 10))
    fig1.suptitle(
        'Gain-Switched Pulse Propagation in SMF-28\n'
        f'$\\alpha_H$ = {laser.alpha_H}, D = {D_PS} ps/nm/km, '
        f'I = {I_PEAK_FACTOR}$\\times I_{{th}}$',
        fontsize=13)

    colors_L = plt.cm.viridis(np.linspace(0.1, 0.9, len(fiber_lengths_km)))

    for col, f_rep in enumerate(freqs):
        fg = f_rep * 1e-9
        for row, (mode, label) in enumerate([
            ('freerun', 'Free-running'), ('sld', 'SLD-injected')
        ]):
            ax = axes1[row, col]
            key = f'{fg:.0f}GHz_{mode}'

            for j, L_km in enumerate(fiber_lengths_km):
                d = results[key][L_km]
                A_out = d['A']
                dt_val = d['dt']
                P = np.abs(A_out)**2 * 1e3  # mW

                t_ps = np.arange(len(P)) * dt_val * 1e12
                # Center on peak
                pk_idx = np.argmax(P)
                t_centered = t_ps - t_ps[pk_idx]

                ax.plot(t_centered, P, color=colors_L[j], lw=1.5,
                        label=f'{L_km} km')

            ax.set_title(f'{fg:.0f} GHz — {label}')
            ax.set_xlabel('Time (ps)')
            ax.set_ylabel('Power (mW)')
            ax.legend(fontsize=7, loc='upper right')
            ax.grid(True, alpha=0.3)

            # Zoom to pulse region
            key0 = f'{fg:.0f}GHz_{mode}'
            fwhm0 = results[key0][0]['metrics']['fwhm']
            ax.set_xlim(-8 * fwhm0 * 1e12, 8 * fwhm0 * 1e12)

    plt.tight_layout()
    fig1.savefig('images/fiber_propagation/pulse_evolution.png', dpi=150)
    print("\n  Saved: images/fiber_propagation/pulse_evolution.png")

    # ── Figure 2: Spectral evolution through fiber ────────────────────

    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
    fig2.suptitle(
        'Spectral Evolution Through Fiber\n'
        f'$\\alpha_H$ = {laser.alpha_H}, D = {D_PS} ps/nm/km',
        fontsize=13)

    for col, f_rep in enumerate(freqs):
        fg = f_rep * 1e-9
        for row, (mode, label) in enumerate([
            ('freerun', 'Free-running'), ('sld', 'SLD-injected')
        ]):
            ax = axes2[row, col]
            key = f'{fg:.0f}GHz_{mode}'

            for j, L_km in enumerate(fiber_lengths_km):
                d = results[key][L_km]
                A_out = d['A']
                dt_val = d['dt']

                # Spectrum
                A_f = np.fft.fftshift(np.fft.fft(A_out))
                S_f = np.abs(A_f)**2
                f_arr = np.fft.fftshift(np.fft.fftfreq(len(A_out), d=dt_val))

                # Normalize
                S_f_norm = S_f / np.max(S_f) if np.max(S_f) > 0 else S_f

                ax.plot(f_arr * 1e-9, 10 * np.log10(S_f_norm + 1e-30),
                        color=colors_L[j], lw=1.2, label=f'{L_km} km')

            ax.set_title(f'{fg:.0f} GHz — {label}')
            ax.set_xlabel('Frequency offset (GHz)')
            ax.set_ylabel('Power spectral density (dB)')
            ax.set_ylim(-40, 3)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

            # Spectral zoom
            bw0 = results[key][0]['metrics']['spec_width_3dB']
            ax.set_xlim(-4 * bw0 * 1e-9, 4 * bw0 * 1e-9)

    plt.tight_layout()
    fig2.savefig('images/fiber_propagation/spectral_evolution.png', dpi=150)
    print("  Saved: images/fiber_propagation/spectral_evolution.png")

    # ── Figure 3: Chirp evolution ─────────────────────────────────────

    fig3, axes3 = plt.subplots(2, 3, figsize=(18, 10))
    fig3.suptitle(
        'Instantaneous Frequency (Chirp) Through Fiber\n'
        f'$\\alpha_H$ = {laser.alpha_H}',
        fontsize=13)

    for col, f_rep in enumerate(freqs):
        fg = f_rep * 1e-9
        for row, (mode, label) in enumerate([
            ('freerun', 'Free-running'), ('sld', 'SLD-injected')
        ]):
            ax = axes3[row, col]
            key = f'{fg:.0f}GHz_{mode}'

            for j, L_km in enumerate([0, 10, 50]):
                if L_km not in results[key]:
                    continue
                d = results[key][L_km]
                A_out = d['A']
                dt_val = d['dt']
                P = np.abs(A_out)**2

                # Chirp only where there's significant power
                phase = np.unwrap(np.angle(A_out))
                inst_freq = np.gradient(phase, dt_val) / (2 * np.pi)

                t_ps = np.arange(len(A_out)) * dt_val * 1e12
                pk_idx = np.argmax(P)
                t_centered = t_ps - t_ps[pk_idx]

                # Mask low-power regions
                P_thresh = np.max(P) * 0.05
                mask = P > P_thresh

                cidx = [0, 2, 4][j] if j < 3 else j
                ax.plot(t_centered[mask], inst_freq[mask] * 1e-9,
                        color=colors_L[[0, 2, 4][j]], lw=1.5,
                        label=f'{L_km} km')

            ax.set_title(f'{fg:.0f} GHz — {label}')
            ax.set_xlabel('Time (ps)')
            ax.set_ylabel('Inst. frequency (GHz)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            fwhm0 = results[key][0]['metrics']['fwhm']
            ax.set_xlim(-6 * fwhm0 * 1e12, 6 * fwhm0 * 1e12)

    plt.tight_layout()
    fig3.savefig('images/fiber_propagation/chirp_evolution.png', dpi=150)
    print("  Saved: images/fiber_propagation/chirp_evolution.png")

    # ── Figure 4: Dispersion penalty summary ──────────────────────────

    fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))
    fig4.suptitle(
        'Dispersion Penalty: Free-Running vs SLD-Injected\n'
        f'SMF-28, $\\alpha_H$ = {laser.alpha_H}',
        fontsize=13)

    markers = {'2GHz': 'o', '5GHz': 's', '10GHz': 'D'}
    colors_freq = {'2GHz': 'C0', '5GHz': 'C1', '10GHz': 'C3'}

    for f_rep in freqs:
        fg = f_rep * 1e-9
        fk = f'{fg:.0f}GHz'

        for mode, ls in [('freerun', '-'), ('sld', '--')]:
            key = f'{fk}_{mode}'
            lbl = f'{fg:.0f} GHz {"free" if mode == "freerun" else "SLD"}'

            fwhm_vals = [results[key][L]['metrics']['fwhm'] * 1e12
                         for L in fiber_lengths_km]
            pk_vals = [results[key][L]['metrics']['peak_power'] * 1e3
                       for L in fiber_lengths_km]
            bw_vals = [results[key][L]['metrics']['spec_width_3dB'] * 1e-9
                       for L in fiber_lengths_km]
            tbp_vals = [results[key][L]['metrics']['tbp']
                        for L in fiber_lengths_km]

            # Normalize to 0 km
            fwhm0 = fwhm_vals[0]
            pk0 = pk_vals[0]

            axes4[0, 0].plot(fiber_lengths_km,
                             [f / fwhm0 for f in fwhm_vals],
                             f'{markers[fk]}{ls}', color=colors_freq[fk],
                             lw=2, ms=6, label=lbl)
            axes4[0, 1].plot(fiber_lengths_km, pk_vals,
                             f'{markers[fk]}{ls}', color=colors_freq[fk],
                             lw=2, ms=6, label=lbl)
            axes4[1, 0].plot(fiber_lengths_km, bw_vals,
                             f'{markers[fk]}{ls}', color=colors_freq[fk],
                             lw=2, ms=6, label=lbl)
            axes4[1, 1].plot(fiber_lengths_km, tbp_vals,
                             f'{markers[fk]}{ls}', color=colors_freq[fk],
                             lw=2, ms=6, label=lbl)

    axes4[0, 0].set_ylabel('FWHM / FWHM$_0$')
    axes4[0, 0].set_title('Pulse broadening factor')
    axes4[0, 0].axhline(1.0, color='gray', ls=':', alpha=0.5)

    axes4[0, 1].set_ylabel('Peak power (mW)')
    axes4[0, 1].set_title('Peak power vs distance')

    axes4[1, 0].set_ylabel('Spectral width (GHz)')
    axes4[1, 0].set_title('3-dB spectral bandwidth')

    axes4[1, 1].set_ylabel('Time-bandwidth product')
    axes4[1, 1].set_title('TBP (0.44 = transform-limited Gaussian)')
    axes4[1, 1].axhline(0.44, color='gray', ls=':', alpha=0.5,
                         label='Transform limit')

    for ax in axes4.flat:
        ax.set_xlabel('Fiber length (km)')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig4.savefig('images/fiber_propagation/dispersion_penalty.png', dpi=150)
    print("  Saved: images/fiber_propagation/dispersion_penalty.png")

    # ── Figure 5: SLD effect on dispersion ────────────────────────────

    fig5, axes5 = plt.subplots(1, 3, figsize=(18, 5.5))
    fig5.suptitle(
        'How SLD Injection Modifies the Dispersion Penalty\n'
        'Ratio: SLD / Free-running at each fiber length',
        fontsize=13)

    for i, f_rep in enumerate(freqs):
        fg = f_rep * 1e-9
        fk = f'{fg:.0f}GHz'
        ax = axes5[i]

        fwhm_free = [results[f'{fk}_freerun'][L]['metrics']['fwhm']
                     for L in fiber_lengths_km]
        fwhm_sld = [results[f'{fk}_sld'][L]['metrics']['fwhm']
                    for L in fiber_lengths_km]
        pk_free = [results[f'{fk}_freerun'][L]['metrics']['peak_power']
                   for L in fiber_lengths_km]
        pk_sld = [results[f'{fk}_sld'][L]['metrics']['peak_power']
                  for L in fiber_lengths_km]

        ratio_fwhm = [s / f if f > 0 else 1.0
                      for s, f in zip(fwhm_sld, fwhm_free)]
        ratio_pk = [s / f if f > 0 else 1.0
                    for s, f in zip(pk_sld, pk_free)]

        ax.plot(fiber_lengths_km, ratio_fwhm, 'o-', color='C0', lw=2,
                ms=7, label='FWHM ratio')
        ax.plot(fiber_lengths_km, ratio_pk, 's-', color='C3', lw=2,
                ms=7, label='Peak power ratio')
        ax.axhline(1.0, color='gray', ls='--', lw=1, alpha=0.5)

        ax.set_xlabel('Fiber length (km)')
        ax.set_ylabel('SLD / Free-running')
        ax.set_title(f'{fg:.0f} GHz')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig5.savefig('images/fiber_propagation/sld_dispersion_effect.png', dpi=150)
    print("  Saved: images/fiber_propagation/sld_dispersion_effect.png")

    plt.close('all')

    # ── Summary table ─────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("  Summary: Fiber Propagation of Gain-Switched Pulses")
    print("=" * 70)

    print(f"\n  Pulse metrics at z = 0 km:")
    print(f"  {'Config':>20s}  {'FWHM':>8s}  {'P_pk':>8s}  "
          f"{'BW':>8s}  {'TBP':>6s}  {'Chirp':>8s}")
    print(f"  {'':>20s}  {'(ps)':>8s}  {'(mW)':>8s}  "
          f"{'(GHz)':>8s}  {'':>6s}  {'(GHz)':>8s}")
    print("  " + "-" * 60)

    for f_rep in freqs:
        fg = f_rep * 1e-9
        fk = f'{fg:.0f}GHz'
        for mode, label in [('freerun', 'free'), ('sld', 'SLD')]:
            key = f'{fk}_{mode}'
            m = results[key][0]['metrics']
            tag = f'{fg:.0f} GHz {label}'
            print(f"  {tag:>20s}  {m['fwhm']*1e12:8.1f}  "
                  f"{m['peak_power']*1e3:8.2f}  "
                  f"{m['spec_width_3dB']*1e-9:8.1f}  "
                  f"{m['tbp']:6.2f}  "
                  f"{m['chirp_at_peak']*1e-9:+8.1f}")

    print(f"\n  Broadening factor (FWHM at L / FWHM at 0):")
    print(f"  {'Config':>20s}" +
          "".join(f"  {L:>5d}km" for L in fiber_lengths_km))
    print("  " + "-" * (20 + 8 * len(fiber_lengths_km)))

    for f_rep in freqs:
        fg = f_rep * 1e-9
        fk = f'{fg:.0f}GHz'
        for mode, label in [('freerun', 'free'), ('sld', 'SLD')]:
            key = f'{fk}_{mode}'
            fwhm0 = results[key][0]['metrics']['fwhm']
            tag = f'{fg:.0f} GHz {label}'
            vals = "".join(
                f"  {results[key][L]['metrics']['fwhm']/fwhm0:7.2f}"
                for L in fiber_lengths_km)
            print(f"  {tag:>20s}{vals}")

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)
