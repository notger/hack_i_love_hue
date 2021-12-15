import logging
import numpy as np
from typing import List, Tuple
from PIL import Image as PILImage

from .fourier_analysis import get_major_frequencies_from_matrix


class Image(object):
    # VERY ingenuous class name for a class to handle all things related to an image,
    # like loading, cutting it to size and extracting the amount of tiles, fixed and moveable.

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
        pix = np.asarray(image)

        for pixel in get_background_pixels(pix):
            single_colour_rows = is_single_colour(
                pix, 
                target_colour=pixel,
                majority_vote_threshold=0.95,
            )
            pix = pix[~single_colour_rows, :, :]

            logging.debug(f'Cut image to size: {pix.shape}, removed all lines with colour {pixel}.')

        return PILImage.fromarray(pix)

    @staticmethod
    def count_tiling(image: PILImage) -> Tuple[int, int]:
        # Takes an image and by counting the different colours per row and column
        # determines the amount of different tiles we have.
        # We have to do this with a Fourier-analysis of the rows/columns, as the screenshots 
        # come in as lossily compressed images with lots of noise.
        # As we want to look at colour changes, we take the man colour values as indicator for that.
        return [round(f) for f in get_major_frequencies_from_matrix(np.asarray(image).mean(axis=2))]

    @staticmethod
    def get_fixed_tile_positions(image: PILImage, tiling: Tuple[int, int]) -> List[Tuple[int, int]]:
        # From the image and based on the tiling-information, determine which of the tiles are
        # fixed and which are movable. The fixed ones have a dark spot in the middle.
        return [(0, 0), (0, 1)]


# ======================== Some helper methods

def get_background_pixels(image_array: np.ndarray) -> Tuple[np.ndarray]:
    # Identifies the background pixels of an image by assuming that the first and the last
    # pixel are background.
    # The first pixel in a phone screenshot is the background of the head-bar, which is totally black.
    # The last pixel in a phone screenshot it the background of the app itself, which is usually not totally black.
    return image_array[0, 0, :], image_array[-1, -1, :]

def is_single_colour(
    matrix: np.ndarray,
    axis: int = 0,
    target_colour: np.ndarray = np.asarray([0, 0, 0]),
    colour_distance_threshold: float = 40.0,
    majority_vote_threshold: float = 1.00,
) -> np.ndarray:
    """
    This function determines whether a given row (axis = 0) or column (axis = 1) 
    is made up mostly of a given colour.

    :param matrix: Image matrix of shape (m, n, 3).
    :param axis: Axis along which to analyse. Chose axis = 0 = row or axis = 1 = column.
    :param target_colour: Colour-value to compare against.
    :param colour_distance_threshold: How far away in terms of mean channel difference to the target
        colour do we allow each pixel to be and still count it as "same"? Can go from 0 (exact match)
        to 255 (every colour is counted as a match)
    :param majority_vote_threshold: Sets the threshold of which share of colours has to be close to the
        target colour such that the aggregated dimension is considered as "single-colour".
    
    Example: 
    1) If you chose axis = 0 and majority_vote = 1.00, you search along each row, meaning that your output will
    be an array of the shape of (m,), indicating in which row all colours are matching the target colour.
    2) If you chose axis = 1 and majority_vote = 0.50, you search along each column, meaning that your output will
    be an array of the shape of (n,), indicating in which column at least 50% of the pixels match the target colour.
    """    
    # Calculate the colour-distance for each pixel:
    delta_colour = np.abs(matrix - target_colour).mean(axis=2)
    
    # We are only interested in matches which are sufficiently close to the target colour,
    # indicated by colour_distance_threshold.
    matches = delta_colour <= colour_distance_threshold

    # We now need aggregate along the axis of which we want to get the majority-vote on:
    aggregation_axis = 1 if axis == 0 else 0
    return matches.mean(axis=aggregation_axis) >= majority_vote_threshold