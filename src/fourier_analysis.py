import numpy as np
import typing


def get_major_frequencies_from_matrix(m: np.ndarray) -> typing.Tuple[float]:
    # We do this by taking the one-pixel-shifted-differences as a simple edge-detector
    # and then analyse the main frequencies in row and columns.
    # The first value returned is the main frequency along the first dimension, so along
    # the rows, going from top to bottom. The second frequency vice versa.
    # More precisely: For M in (m x n), we return f_m, f_n such.
    # Previous version was taking the mean for the respective axis, but that produced
    # twice the critical frequency, as you had the small dots in there as well and the
    # factor of 0.5 was dependent on there being dots in every row and in every column.
    # return (
    #     0.5 * _get_major_frequency_from_array(m.mean(axis=1)),
    #     0.5 * _get_major_frequency_from_array(m.mean(axis=0)),
    # )
    # Working version emphasises the differences by squaring the input:
    return (
        _get_major_frequency_from_array(m[:, 5] ** 2),
        _get_major_frequency_from_array(m[5, :] ** 2),
    )

def _get_major_frequency_from_array(arr: np.ndarray, ignore_constant=True) -> float:
    # Do the FFT, standard control theory notation:
    A = np.fft.fft(arr)
    frequencies = np.fft.fftfreq(len(A), d=1/len(arr))
    amplitudes = np.abs(A)
    N = len(A) // 2

    # Resize the arrays to only contain the positive frequencies:
    amplitudes = amplitudes[:N]
    frequencies = frequencies[:N]

    # In some cases we want to get the first harmonic and ignore the constant offset.
    if ignore_constant:
        amplitudes[0] = 0

    # Return the frequency for the maximal amplitude:
    return frequencies[np.argmax(amplitudes)]
