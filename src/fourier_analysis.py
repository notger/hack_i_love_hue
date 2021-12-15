import numpy as np


def get_major_frequencies_from_matrix(m: np.ndarray) -> float:
    # We do this by taking the one-pixel-shifted-differences as a simple edge-detector
    # and then analyse the main frequencies in row and columns.

    return 0.0, 0.0

def _get_major_frequency_from_array(arr: np.ndarray, ignore_constant=True) -> float:
    # Do the FFT, standard control theory notation:
    A = np.fft.fft(arr)
    frequencies = np.fft.fftfreq(len(A), d=1/len(arr))
    N = len(A) // 2

    # Resize the arrays to only contain the positive frequencies:
    A = A[:N]
    frequencies = frequencies[:N]
    amplitudes = np.abs(A)

    # In some cases we want to get the first harmonic and ignore the constant offset.
    if ignore_constant:
        amplitudes[0] = 0

    # Return the frequency for the maximal amplitude:
    return frequencies[np.argmax(amplitudes)]
