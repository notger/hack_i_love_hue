import unittest
from src.image_manipulation import Image


class TestImageManipulation(unittest.TestCase):

    def setUp(self) -> None:
        self.image = Image('images/default-2021-12-07-205120.jpeg')
        return super().setUp()

    def test_load_image(self):
        self.assertIsNotNone(self.image)
        self.assertIsNotNone(Image.load_image('images/default-2021-12-07-205120.jpeg'))

    def test_cut_to_size(self):
        self.assertTrue(False)

    def test_count_tiling(self):
        self.assertTrue(False)

    def test_get_fixed_tile_positions(self):
        self.assertTrue(False)
