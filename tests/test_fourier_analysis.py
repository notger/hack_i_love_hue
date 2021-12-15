import unittest
import numpy as np
from src.fourier_analysis import get_major_frequencies_from_matrix, _get_major_frequency_from_array


class TestFourierAnalysis(unittest.TestCase):

    def setUp(self) -> None:
        # Create a sinusoidal signal:
        self.t = np.linspace(0, 1, 101)
        self.sinusoid = np.sin(4 * 3.1415 * self.t)  # Should have a frequency of 2 Hz

        # Create a flat signal with some spikes:
        self.peaks = np.zeros((100,), dtype=float)
        self.peaks[[20, 40, 60, 80]] = 1.0
        self.peaks[[19, 21, 39, 41, 59, 61, 79, 81]] = 0.5

        # Create a test matrix with peaks in a grid-like-fashion:
        self.matrix = np.zeros((100, 100), dtype=float)
        self.matrix[[20, 40, 60, 80], :] = 1.0
        self.matrix[[19, 21, 39, 41, 59, 61, 79, 81], :] = 0.5
        self.matrix[:, [33, 66]] = 1.0
        self.matrix[:, [32, 34, 65, 67]] = 0.5

        return super().setUp()

    def test_array_fourier_analysis(self):
        self.assertAlmostEqual(2.0, _get_major_frequency_from_array(self.sinusoid))
        self.assertAlmostEqual(5.0, _get_major_frequency_from_array(self.peaks))
        self.assertAlmostEqual(0.0, _get_major_frequency_from_array(self.peaks, ignore_constant=False))

    def test_matrix_fourier_analysis(self):
        frequencies = get_major_frequencies_from_matrix(self.matrix)
        self.assertAlmostEqual(5.0, frequencies[0])
        self.assertAlmostEqual(3.0, frequencies[1])
