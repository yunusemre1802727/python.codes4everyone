import numpy as np
import matplotlib.pyplot as plt

# Generate a signal with noise
np.random.seed(42)
t = np.linspace(0, 1, 1000, endpoint=False)

# Create an actual signal composed of two sine waves
actual_signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 15 * t)

# Add noise to the measured signal
measured_signal = actual_signal + 0.5 * np.random.normal(size=len(t))

# Plot the original signal
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(t, measured_signal, label='Measured Signal (with noise)')
plt.title('Measured Signal (with noise)', fontsize=16)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Amplitude', fontsize=14)
plt.legend()

# Compute the FFT of the measured signal
fft_result_measured = np.fft.fft(measured_signal)
frequencies_measured = np.fft.fftfreq(len(fft_result_measured), t[1] - t[0])
amplitudes_measured = np.abs(fft_result_measured)

# Plot the frequency spectrum of the measured signal
plt.subplot(3, 1, 2)
plt.plot(frequencies_measured[:50], amplitudes_measured[:50])
plt.title('Frequency Spectrum (Measured Signal)', fontsize=16)
plt.xlabel('Frequency', fontsize=14)
plt.ylabel('Amplitude', fontsize=14)

# Denoising Strategy: Low-pass filter
cutoff_frequency = 19
fft_result_filtered = fft_result_measured * (np.abs(frequencies_measured) < cutoff_frequency)
signal_filtered = np.fft.ifft(fft_result_filtered)

# Plot the actual signal and denoised signal using a low-pass filter
plt.subplot(3, 1, 3)
plt.plot(t, actual_signal, label='Actual Signal')
plt.plot(t, signal_filtered.real, label='Denoised (Low-pass filter)')
plt.title('Denoised Signal (Low-pass filter)', fontsize=16)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Amplitude', fontsize=14)
plt.legend()

plt.tight_layout()
plt.show()
