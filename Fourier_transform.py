import numpy as np
import matplotlib.pyplot as plt

# Generate a sample signal
fs = 1000  # Sampling frequency
time = 1
t = np.linspace(0, time, fs*time) # Time vector
frequencies = [10, 50, 100]  # Frequencies in the signal
signal = ( 
    1.0 * np.sin(2 * np.pi * frequencies[0] * t) +
    0.5 * np.sin(2 * np.pi * frequencies[1] * t) +
    0.2 * np.sin(2 * np.pi * frequencies[2] * t)
        )

# Perform FFT
fft_result =  np.fft.fft(signal) / (len(signal) / 2)
frequencies_fft = np.fft.fftfreq(len(t), 1/fs)  # Frequency axis

# Plot the original signal and its FFT result
plt.figure(figsize=(12, 6))

# Plot the original signal
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title('Original Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Plot the FFT result
plt.subplot(2, 1, 2)
plt.plot(frequencies_fft[:105], np.abs(fft_result)[:105])
plt.title('Frequency Content (FFT)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

plt.tight_layout()
plt.show()
