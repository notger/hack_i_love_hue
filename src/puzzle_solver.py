import logging
import numpy as np
from typing import List, Tuple
from .image_manipulation import Image


def _generate_fixed_tiles_mask(
    fixed_tiles_tuple_list: List[Tuple[int, int]], 
    tiling: List[int]
) -> np.ndarray:
    """
    Transforms the list of coordinate tuples which indicate the fixed tiles into a numpy matrix
    which can be used as a mask for numeric tasks where you want or do not want to include/exclude
    the fixed tiles.
    """
    mask = np.zeros((tiling[0], tiling[1]))

    for fixed_tile in fixed_tiles_tuple_list:
        mask[fixed_tile[0], fixed_tile[1]] = 1.0

    return mask

def _find_reference_tiles(tile_x: int, tile_y: int, fixed_tiles: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Helper function to determine the reference tiles for a given tile.
    A reference tile is a fixed tile which is bordering the tile in question in either
    left, right, up or down direction.
    """
    bordering_tiles = [
        tile for tile in [
            (tile_x - 1, tile_y), (tile_x + 1, tile_y), (tile_x, tile_y - 1), (tile_x, tile_y + 1)
        ] if tile in fixed_tiles
    ]
    return bordering_tiles

def _calculate_delta_to_reference_tiles(
    reference_tiles: List[Tuple[int, int]],
    tile_colours: np.ndarray,
) -> np.ndarray:
    """
    Calculates the deltas between the target tile and the reference tile and returns
    a mean of the absolute deltas in the colour channels, for all tiles.
    """
    return sum([np.abs(tile_colours - tile_colours[ref_tile[0], ref_tile[1], :]) for ref_tile in reference_tiles]).mean(axis=2) / len(reference_tiles)
    
def _extract_target_tile_and_colour(deltas: np.ndarray, fixed_tiles_mask: np.ndarray) -> Tuple[List[int], np.ndarray]:
    """
    Returns the tile's coordinate and colour with the lowest difference in the delta-matrix.
    """
    # TODO
    return None

def find_final_ordering(image: Image) -> np.ndarray:
    """
    Determines the final ordering of the tiles in an image by doing the following steps
    for each non-fixed tile:
    a) finding the fixed tiles bordering that tile
    b) calculating all the deltas between the main colour of the tile in question and
       all reference tiles with previously solved tiles being treated as reference tiles
    c) taking the absolute of the mean of the deltas
    d) extracting the original position and colour value of the tile that should be in 
       the tile position examined

    The result are two matrices which indicate the original position and colour value
    at the supposed final position. This can be used by the step-generator of the solver
    to determine the best swapping order. 
    """
    # Generate a numpy array with the same dimensions as the tiling from the image, but 
    # two entries in the 3rd dimension which will contain the coordinates / colours. 
    # This will make assignments and read-outs easier, later.
    final_ordering = np.zeros((image.tiling[0], image.tiling[1], 2))
    final_colouring = np.zeros((image.tiling[0], image.tiling[1], 3))

    # TODO: Also handle fixed tiles and put them in the final ordering, for cleanliness' sake.

    return final_ordering, final_colouring

def determine_swapping_order(final_ordering: np.ndarray) -> List[Tuple[Tuple[int, int]]]:
    return None