"""
Just defines a base class to hold the solution and a class to hold intermediate results.
They also contain integrity checks which can be used in tests as well as on-the-fly to
check whether a given solution gives sensible results.
"""
import logging
import numpy as np
from collections import Counter
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
        is_sane = self.is_sane()
        s = f'State with index {self.idx} is {"NOT" if not is_sane else ""} sane.'
        s += '' if is_sane else f'\nDetected duplicates: {self.get_duplicates()}'
        return s

    def is_sane(self):
        """
        Performs an integrity check on the data of the state.
        Data is considered sane, when 
        a) Every coordinate pair in the ordering appears only once.
        b) Every colour in the colouring appears only once.
        As the index is optional given it can be inferred from the ordering
        of the state in a list, it is not a criterion for sanity.
        """
        duplicates = self.get_duplicates()
        return len(duplicates['coord_pair_duplicates']) == 0 and len(duplicates['colour_duplicates']) == 0

    def get_duplicates(self):
        # As we want to check for coord/colour-tuples and as lists are not hashable and thus
        # can not be used in the Counter-object, we have to first flatten the matrices to arrays,
        # such that a matrix (N, M, x) becomes an array (N*M, x). We then convert to a list and from
        # there to a tuple, which we can then use in the Counter-class.
        coord_counter_dict = Counter(
            [
                tuple(coord_pair) 
                for coord_pair 
                in self.ordering.flatten().reshape((-1, 2)).tolist()
            ]
        )
        colour_counter_dict = Counter(
            [
                tuple(colour) 
                for colour 
                in self.colouring.flatten().reshape((-1, 3)).tolist()
            ]
        )
        return {
            'coord_pair_duplicates': [k for k, v in coord_counter_dict.items() if v >= 2],
            'colour_duplicates': [k for k, v in colour_counter_dict.items() if v >= 2]
        }


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