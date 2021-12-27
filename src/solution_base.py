"""
Just defines a base class to hold the solution and a class to hold intermediate results.
They also contain integrity checks which can be used in tests as well as on-the-fly to
check whether a given solution gives sensible results.
"""
import logging
import numpy as np
from numpy.testing._private.utils import raises
from .final_ordering import find_final_ordering
from .image_manipulation import Image

# Set up logging:
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s: %(message)s'))
logger = logging.getLogger('solution_base')
logger.setLevel('INFO')
logger.addHandler(console_handler)
logger.propagate = False


class Solution(object):

    def __init__(
        self, 
        image: Image, 
    ) -> None:
        super().__init__()
        self.image = image
        self.initial_colouring = image.tile_colours
        final_ordering, final_colouring = find_final_ordering(image)
        self.final_ordering = final_ordering
        self.final_colouring = final_colouring
        self.steps = []
        self.initial_state = State(create_initial_ordering(self.final_ordering), self.initial_colouring)
        self.final_state = State(self.final_ordering, self.final_colouring)

        # Perform a sanity check on initial and final state. We want to do this every time we
        # create a solution, as we can never be sure whether we are handed a proper image or something
        # upstream has silently failed.
        if self.initial_state.is_sane() & self.final_state.is_sane():
            logger.info('Successfully created solution object with sane fringe states, ready for solver method.')
        else:
            raise ValueError(f'State sanity violated! Initial state sane: {self.initial_state.is_sane()}, final state sane: {self.final_state.is_sane()}')

    def solve(self, solver) -> None:
        self.steps = solver(self.initial_colouring, self.final_ordering)

    def __str__(self) -> str:
        return f'Solution for image {self.image.file_name} in {len(self.steps)} steps.'

class State(object):
    """
    Class to hold the intermediate states of the solution.
    """
    def __init__(
        self, 
        ordering: np.ndarray, 
        colouring: np.ndarray, 
        idx: int = -1,
    ) -> None:
        super().__init__()
        self.ordering = ordering
        self.colouring = colouring
        self.idx = idx

    def __str__(self) -> str:
        return f'State with index {self.idx}.'

    def is_sane(self):
        """
        Performs an integrity check on the data of the state.
        Data is considered sane, when 
        a) Every coordinate pair in the ordering appears only once.
        b) Every colour in the colouring appears only once.
        As the index is optional given it can be inferred from the ordering
        of the state in a list, it is not a criterion for sanity.
        """
        # Check a): Are all coordinate pairs generated unique?
        # We flatten the coordinate matrix and reshape it, to create an
        # array of the coordinate pairs and then get the set of those.
        # The number of items in that set has to match the number of 
        # items in the flatted matrix.
        flattened_coordinates = self.ordering.flatten().reshape((-1, 2)).tolist()
        flattened_coordinates_set = set(
            [(i, j) for i, j in flattened_coordinates]  # We have to do this, as lists are not hashable.
        )
        coordinates_are_unique = len(flattened_coordinates) == len(flattened_coordinates_set)

        # Check b): Is every colour unique?
        # Same approach as above.
        flattened_colours = self.colouring.flatten().reshape((-1, 3)).tolist()
        flattened_colours_set = set(
            [(c1, c2, c3) for c1, c2, c3 in flattened_colours]
        )
        colours_are_unique = len(flattened_colours) == len(flattened_colours_set)

        return coordinates_are_unique & colours_are_unique


def create_initial_ordering(ordering_template):
    """
    Helper function to create an initial ordering matrix where all 
    coordinates are in their original position, 
    i.e. ordering[i, j, 0] = i and ordering[i, j, 1] = j
    """
    ordering = np.zeros_like(ordering_template, dtype=int)
    N_i = ordering.shape[0]
    N_j = ordering.shape[1]
    for i in range(N_i):
        for j in range(N_j):
            ordering[i, j, 0] = i
            ordering[i, j, 1] = j

    return ordering