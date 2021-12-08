import logging
from typing import List, Tuple
from PIL import Image as PILImage


class Image(object):
    # VERY ingenuous class to handle all things related to an image,
    # like loading, cutting it to size and extracting the amount of
    # tiles, fixed and moveable.

    def __init__(self, file_name) -> None:
        self.image = self.cut_to_size(
            self.load_image(file_name)
        )

        self.tiling = self.count_tiling(self.image)
        self.fixed_tiles = self.get_fixed_tile_positions(self.image, self.tiling)

        super().__init__()

    @staticmethod
    def load_image(file_name: str) -> PILImage:
        im = PILImage.open(file_name)
        logging.info(f'Filename {file_name} loaded.\n      Format: {im.size}, {im.format}, {im.mode}')
        return im

    @staticmethod
    def cut_to_size(image: PILImage) -> PILImage:
        # Takes a matplotlib-image and cuts it to size by removing the dark parts
        # which are top and bottom of the image.
        # Returns a matplotlib-image again, which should only contain the tiles.
        return image

    @staticmethod
    def count_tiling(image: PILImage) -> Tuple[int, int]:
        # Takes an image and by counting the different colours per row and column
        # determines the amount of different tiles we have.
        return (1, 1)

    @staticmethod
    def get_fixed_tile_positions(image: PILImage, tiling: Tuple[int, int]) -> List[Tuple[int, int]]:
        # From the image and based on the tiling-information, determine which of the tiles are
        # fixed and which are movable. The fixed ones have a dark spot in the middle.
        return [(0, 0), (0, 1)]
