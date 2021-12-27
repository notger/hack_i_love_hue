import logging
import numpy as np
from typing import List, Tuple
from .image_manipulation import Image

# Set up logging:
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s: %(message)s'))
logger = logging.getLogger('puzzle_solver')
logger.setLevel('DEBUG')
logger.addHandler(console_handler)
logger.propagate = False


class Solution(object):

    def __init__(self, initial_ordering, initial_colouring, final_ordering, final_colouring) -> None:
        super().__init__()
        self.initial_ordering = initial_ordering
        self.initial_colouring = initial_colouring
        self.final_ordering = final_ordering
        self.final_colouring = final_colouring
        self.steps = []

    def solve(self, solver):
        self.steps = solver()

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
    
def _extract_target_tile_coordinates(
    deltas: np.ndarray, 
    fixed_tiles_mask: np.ndarray
) -> Tuple[int, int]:
    """
    Returns the tile's coordinate with the lowest difference in the delta-matrix.
    """
    # Isolate the eligible cells by setting the delta-values to very high values where the tiles are fixed.
    # This prevents us identifying a fixed tiles as the one with the lowest difference:
    tmp = deltas + 1e5 * fixed_tiles_mask
    
    location = np.unravel_index(np.argmin(tmp), tmp.shape)
    return location

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

    Known weaknesses / TODO: If the problem is not solvable from the top left to the bottom right,
    then the approach chosen here fails. E.g. if there are no fixed tiles in the cells
    (0, 0), (0, 1) or (1, 0).
    """
    # Generate a numpy array with the same dimensions as the tiling from the image, but 
    # two entries in the 3rd dimension which will contain the coordinates / colours. 
    # This will make assignments and read-outs easier, later.
    N_i, N_j = image.tiling
    final_ordering = -np.ones((N_i, N_j, 2), dtype=int)
    final_colouring = -np.ones((N_i, N_j, 3), dtype=int)

    mask = _generate_fixed_tiles_mask(image.fixed_tiles, image.tiling)

    # We will need to keep a ledger on all tiles already fixed or determined,
    # as tiles we already know about become "fixed tiles" in the sense of our
    # puzzle solver. This requires that we copy the fixed tiles to have it 
    # manipulatable without side effects.
    fixed_tiles_list = image.fixed_tiles[:]
    
    # Create a lookup which translates from new to old coordinates such that the
    # new coordinates are the keys and the old coordinates are the values: D[new] = old
    new_to_old_lookup = {}

    # Create the initial seeding by copying over the fixed tiles and their colour values:
    for i in range(N_i):
        for j in range(N_j):
            # Where the tile comes from that should be in position (i, j) depends on
            # whether the tile is fixed or not. For fixed tiles, we just copy the tile,
            # for swappable tiles, we find the one with the closest difference to the
            # reference tiles:
            if (i, j) in image.fixed_tiles:
                # Assign the source position and colours to the target and note the fixed
                # tiles in the list of already processed tiles:
                final_ordering[i, j, 0] = i
                final_ordering[i, j, 1] = j
                final_colouring[i, j, :] = image.tile_colours[i, j, :]
                new_to_old_lookup[(i, j)] = (i, j)
                
    # Now deal with the movable tiles:
    for i in range(N_i):
        for j in range(N_j):
            if (i, j) not in fixed_tiles_list:
                reference_tiles = _find_reference_tiles(i, j, fixed_tiles_list)
                
                # What would the reference tiles' coordinates be in the original image?
                reference_tiles_old_coordinates = [new_to_old_lookup[(i, j)] for (i, j) in reference_tiles]
                
                # Calculate the deltas of all the colours in the original image to the reference colours
                # indexed by the old coordinate frame:
                deltas = _calculate_delta_to_reference_tiles(reference_tiles_old_coordinates, image.tile_colours)

                # Extract the tile with the lowest distance, assuming the fixed-point-mask given in the old
                # coordinates:
                k, l = _extract_target_tile_coordinates(deltas, mask)

                # Assign the source position and colours to the target:
                final_ordering[i, j, 0] = k
                final_ordering[i, j, 1] = l
                final_colouring[i, j, :] = image.tile_colours[k, l, :]

                # Register the newly swapped in colour in the mask:
                new_to_old_lookup[(i, j)] = (k, l)
                mask[k, l] = 1.0
                fixed_tiles_list.append((i, j))

                logging.debug(f'Checked for position {(i, j)}: Target is originally at {(k, l)}.')

            else:
                logging.debug(f'Checked for position {(i, j)}: Fixed tile, nothing to do.')

    logging.info('Created final ordering.')

    return final_ordering, final_colouring

def determine_swapping_order(final_ordering: np.ndarray) -> List[Tuple[Tuple[int, int]]]:
    return None