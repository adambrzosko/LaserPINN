import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift, ifft

# Constants
c = 3e8  # speed of light in m/s
lambda0 = 1300e-9  # center wavelength in meters (1300 nm typical for SLD)
delta_lambda = 50e-9  # spectral FWHM in meters

# Derived parameters
freq0 = c / lambda0
delta_freq = c * delta_lambda / (lambda0 ** 2)

# Frequency axis
N = 2048
freq = np.linspace(freq0 - 5 * delta_freq, freq0 + 5 * delta_freq, N)
wavelength = c / freq

# Gaussian spectral power distribution
def gaussian(f, f0, df):
    return np.exp(-((f - f0) ** 2) / (2 * (df / 2.355) ** 2))

spectrum = gaussian(freq, freq0, delta_freq)

# Normalize spectrum
spectrum /= np.max(spectrum)

# Time domain coherence function (via IFFT)
time = np.linspace(-5e-12, 5e-12, N)  # time window in seconds
coherence = ifft(fftshift(spectrum))
coherence = np.abs(coherence)

# Plotting
plt.figure(figsize=(12, 5))

# Spectrum
plt.subplot(1, 2, 1)
plt.plot(wavelength * 1e9, spectrum)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Normalized Intensity')
plt.title('SLD Emission Spectrum')
plt.grid(True)

# Coherence
plt.subplot(1, 2, 2)
plt.plot(time * 1e12, coherence)
plt.xlabel('Time Delay (ps)')
plt.ylabel('Coherence Amplitude')
plt.title('Temporal Coherence Function')
plt.grid(True)

plt.tight_layout()
plt.show()
