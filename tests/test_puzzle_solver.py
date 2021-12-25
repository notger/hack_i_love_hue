import unittest
import numpy as np
from numpy.lib.type_check import _asfarray_dispatcher
from src.puzzle_solver import (
    _generate_fixed_tiles_mask, 
    _find_reference_tiles,
    _calculate_delta_to_reference_tiles,
    _extract_target_tile_coordinates,
)


class TestPuzzleSolver(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()

    def test_fixed_tiles_mask(self):
        tiling = (4, 4)
        fixed_tiles_tuple_list = [(0, 0), (2, 3), (3, 3)]
        mask = _generate_fixed_tiles_mask(fixed_tiles_tuple_list=fixed_tiles_tuple_list, tiling=tiling)

        for i in range(tiling[0]):
            for j in range(tiling[1]):
                self.assertAlmostEqual(
                    1.0 if (i, j) in fixed_tiles_tuple_list else 0.0, 
                    mask[i, j]
                )

    def test_find_reference_tiles(self):
        fixed_tiles_tuple_list = [(0, 0), (2, 0), (1, 1)]

        # The tile (1, 0) should have three reference tiles:
        self.assertEqual(
            set(fixed_tiles_tuple_list), 
            set(_find_reference_tiles(1, 0, fixed_tiles_tuple_list))
        )

        # The tile (3, 0) should have one reference tile:
        self.assertEqual(
            set([(2, 0)]), 
            set(_find_reference_tiles(3, 0, fixed_tiles_tuple_list))
        )

        # The tile (2, 1) should have two reference tiles:
        self.assertEqual(
            set([(2, 0), (1, 1)]), 
            set(_find_reference_tiles(2, 1, fixed_tiles_tuple_list))
        )

        # The tile (3, 3) should have no reference tiles:
        self.assertEqual([], _find_reference_tiles(3, 3, fixed_tiles_tuple_list))

    def test_calculate_delta_to_reference_tiles(self):
        # Create a colour matrix with all grey values and two white values in the first and the last place:
        colours = np.zeros((4, 4, 3), dtype=int) + 0.5
        colours[0, 0, :] = 1
        colours[-1, -1, :] = 1

        deltas = _calculate_delta_to_reference_tiles([(0, 0)], colours)

        # The delta-matrix with respect to the (0, 0)-white-value should be all 0.5, 
        # except for the first and the last entry:
        target_deltas = np.zeros((colours.shape[0], colours.shape[1]), dtype=float) + 0.5
        target_deltas[0, 0] = 0.0
        target_deltas[-1, -1] = 0.0

        self.assertTrue(np.allclose(target_deltas, deltas))

        # And it should of course have the correct shape:
        self.assertEqual((4, 4), deltas.shape)

    def test_extract_target_tiles_coordinates(self):
        deltas = np.asarray(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ],
            dtype=float,
        )

        mask = np.asarray(
            [
                [1, 1, 1],
                [1, 0, 0],
                [0, 0, 1]
            ],
            dtype=int,
        )

        self.assertEqual((1, 1), _extract_target_tile_coordinates(deltas, mask))
        self.assertEqual((2, 1), _extract_target_tile_coordinates(-deltas, mask))
