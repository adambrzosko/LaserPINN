"""
Pulse and spectral analysis utilities for fiber-propagated fields.

    from fiber.analysis import pulse_metrics, spectral_centroid, band_power
"""
import numpy as np


def pulse_metrics(A, dt):
    """Extract metrics from a single pulse envelope, |A|^2 in Watts.

    Returns dict with: peak_power, fwhm, rms_width, energy,
    spec_width_3dB, rms_spec, tbp (time-bandwidth product),
    chirp_at_peak (instantaneous frequency at the power peak, rad/s... in Hz).
    """
    P = np.abs(A) ** 2
    peak_power = float(np.max(P))
    peak_idx = int(np.argmax(P))

    energy = float(np.sum(P) * dt)

    half_max = peak_power / 2.0
    above = np.where(P >= half_max)[0]
    fwhm = float((above[-1] - above[0]) * dt) if len(above) > 1 else dt

    t_arr = np.arange(len(A)) * dt
    t_mean = np.sum(t_arr * P) / np.sum(P) if np.sum(P) > 0 else 0
    rms_width = float(np.sqrt(np.sum((t_arr - t_mean) ** 2 * P) / np.sum(P))) if np.sum(P) > 0 else 0.0

    A_f = np.fft.fftshift(np.fft.fft(A))
    S_f = np.abs(A_f) ** 2
    f_arr = np.fft.fftshift(np.fft.fftfreq(len(A), d=dt))

    S_f_max = np.max(S_f)
    above_f = np.where(S_f >= S_f_max / 2.0)[0]
    spec_width_3dB = float(abs(f_arr[above_f[-1]] - f_arr[above_f[0]])) if len(above_f) > 1 else 1.0 / dt

    f_mean = np.sum(f_arr * S_f) / np.sum(S_f) if np.sum(S_f) > 0 else 0
    rms_spec = float(np.sqrt(np.sum((f_arr - f_mean) ** 2 * S_f) / np.sum(S_f))) if np.sum(S_f) > 0 else 0.0

    tbp = rms_width * rms_spec * 2 * np.pi

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


def spectral_centroid(A, dt):
    """Power-weighted mean angular-frequency offset (rad/s), in raw FFT
    (Omega = 2*pi*fftfreq) convention.

    Tracks the soliton self-frequency shift (SSFS) and other Raman-induced
    spectral shifts under propagation. Note: with this codebase's envelope
    convention (see fiber.raman_response.raman_gain_spectrum), the physical
    optical frequency at offset Omega is omega0 - Omega, so a physical
    *redshift* corresponds to this centroid *increasing* -- negate it
    (or use -spectral_centroid(...)) if you want a quantity where negative
    directly means redshift.
    """
    A_f = np.fft.fft(A)
    Omega = 2 * np.pi * np.fft.fftfreq(len(A), d=dt)
    S = np.abs(A_f) ** 2
    total = np.sum(S)
    return float(np.sum(Omega * S) / total) if total > 0 else 0.0


def band_power(A, dt, omega_lo, omega_hi):
    """Spectral energy within an angular-frequency offset band [omega_lo, omega_hi)
    (rad/s from the carrier) -- e.g. to isolate a Stokes or anti-Stokes band."""
    A_f = np.fft.fft(A)
    Omega = 2 * np.pi * np.fft.fftfreq(len(A), d=dt)
    S = np.abs(A_f) ** 2
    mask = (Omega >= omega_lo) & (Omega < omega_hi)
    n = len(A)
    return float(np.sum(S[mask]) * dt / n)
