import unittest
import numpy as np
from src.image_manipulation import PILImage, Image, get_background_pixels, get_background_pixels, is_single_colour


class TestImageManipulation(unittest.TestCase):

    def setUp(self) -> None:
        self.image_path = 'images/test1.jpeg'
        return super().setUp()

    def test_load_image(self):
        self.assertIsNotNone(Image(self.image_path))
        self.assertIsNotNone(Image.load_image(self.image_path))

    def test_cut_to_size(self):
        image = PILImage.open(self.image_path)
        image_cut_to_size = Image.cut_to_size(image)
        self.assertIsNotNone(image_cut_to_size)
        for k, size in enumerate(image.size):
            self.assertGreaterEqual(size, image_cut_to_size.size[k])

    def test_get_background_pixel(self):
        # Let's test this with an established picture right away.
        # Alternatively, you can set up a synthetic test at some point.
        background_pixels = get_background_pixels(
            np.asarray(PILImage.open(self.image_path))
        )

        self.assertEqual([0, 0, 0], background_pixels[0].tolist())
        self.assertEqual([32, 25, 33], background_pixels[1].tolist())

    def test_is_single_colour_array(self):
        white = np.asarray([255, 255, 255])

        # Create a completely black matrix:
        single_colour_matrix = np.zeros((5, 10, 3), dtype=int)

        # Create a colour array with the first half having half the colours being [255, 255, 255]
        # and the second half having only one entry being white:
        dual_colour_matrix = np.zeros((10, 100, 3), dtype=int)
        dual_colour_matrix[:5, 50:, :] = 255
        dual_colour_matrix[5:, 99, :] = 255

        # Now run some tests ...
        # The single colour array should be of a single colour with default values:
        self.assertTrue((is_single_colour(single_colour_matrix) == True).all())

        # The dual colour arrays should not be of a single colour:
        self.assertFalse((is_single_colour(dual_colour_matrix) == True).all())

        print(is_single_colour(dual_colour_matrix, majority_vote_threshold=0.5).shape)        
        print(is_single_colour(dual_colour_matrix, majority_vote_threshold=0.75))

        # Given a lower threshold of exactly 50%, all rows should have enough black pixels:
        self.assertTrue((is_single_colour(dual_colour_matrix, majority_vote_threshold=0.5) == True).all())

        # But only half of the lines should have enough white pixels:
        self.assertAlmostEqual(
            0.5,
            is_single_colour(dual_colour_matrix, majority_vote_threshold=0.5, target_colour=white).mean()
        )

        # Now what happens if we flip the indices? Are the results preserved?
        self.assertAlmostEqual(
            0.5,
            is_single_colour(
                dual_colour_matrix.transpose(1, 0, 2),
                axis=1,
                majority_vote_threshold=0.5, 
                target_colour=white
            ).mean()
        )

    def test_is_single_colour_array_with_continuous_colours(self):
        # We want to test the colour-distance-threshold, so we want to generate a matrix of size
        # (10, 255, 3) with continuously increasing colours from left to right.
        # So it starts out all black left and then goes via the greyscale to white on the right.
        continuous_colour_matrix = np.zeros((10, 255, 3), dtype=int)
        for k in range(3):
            continuous_colour_matrix[:, :, k] = np.linspace(0, 255, 255, dtype=int)

        # With a continuous matrix like this, for a target colour black, we expect 50% to be 
        # within a mean distance of 127 or less. So for a majority vote threshold of 0.5 and
        # a colour_distance_threshold of 127, we should get all lines being labelled as being
        # sufficiently close to "black".
        self.assertAlmostEqual(
            1.0,
            is_single_colour(
                continuous_colour_matrix, colour_distance_threshold=127, majority_vote_threshold=0.5
            ).mean()
        )

        # If we lower the colour-distance-threshold, but keep the 50% requirement, we should get
        # no lines being labelled as black:
        self.assertAlmostEqual(
            0.0,
            is_single_colour(
                continuous_colour_matrix, colour_distance_threshold=126, majority_vote_threshold=0.5
            ).mean()
        )

    def test_count_tiling(self):
        image = PILImage.open(self.image_path)
        self.assertEqual(
            (10, 10),
            Image.count_tiling(image)
        )

    # def test_get_fixed_tile_positions(self):
    #     self.assertTrue(False)
