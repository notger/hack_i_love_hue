import unittest
import numpy as np
from numpy.lib.type_check import _asfarray_dispatcher
from src.puzzle_solver import (
    _generate_fixed_tiles_mask, 
    _find_reference_tiles,
    _calculate_delta_to_reference_tiles,
    _extract_target_tile_coordinates,
    find_final_ordering,
)
from src.image_manipulation import Image


class TestPuzzleSolver(unittest.TestCase):

    def setUp(self) -> None:
        self.image_path1 = 'images/test1.jpeg'
        self.image_path2 = 'images/test2-jpeg'
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

    def test_find_final_ordering(self):
        # Let's load an image to create the necessary structure:
        image = Image(self.image_path1)

        # But let's overwrite the values in there to make for a more easily controlled experiment.
        # We want a simple 3x3 image with some random colours, but the first and the last cell
        # should be black and white, respectively.
        image.fixed_tiles = [(0, 0), (2, 2)]
        image.tiling = (3, 3)
        image.tile_colours = np.asarray(
            [
                [[0, 0, 0], [240, 240, 240], [40, 40, 40]], 
                [[120, 120, 120], [200, 200, 200], [80, 80, 80]], 
                [[160, 160, 160], [100, 100, 100], [255, 255, 255]]
            ]
        )
        target_ordering = np.asarray(
            [
                [[0, 0], [0, 2], [1, 2]],
                [[2, 1], [1, 0], [2, 0]],
                [[1, 1], [0, 1], [2, 2]]
            ]
        )

        final_ordering, final_colouring = find_final_ordering(image)

        # For the test, we want to look at the flattened array, where all colours are averaged.
        # In that array, all the colours have to increase, which means each colour has a larger
        # value than the last one:
        flattened_colouring = final_colouring.mean(axis=2).flatten()
        self.assertTrue((flattened_colouring[1:] - flattened_colouring[:-1] > 0).all())

        # Compare the ordering-matrices:
        self.assertTrue((target_ordering == final_ordering).all())
