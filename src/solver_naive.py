"""
Solver to derive the steps from the initial to the final ordering
in a naive way: We just go step by step and swap each colour for
its target colour.
"""
import logging
import typing
import numpy as np
from .solution_base import State, create_initial_ordering

# Set up logging:
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s: %(message)s'))
logger = logging.getLogger('solver_naive')
logger.setLevel('INFO')
logger.addHandler(console_handler)
logger.propagate = False


def naive_method(
    initial_colouring: np.ndarray, 
    final_ordering: np.ndarray,
) -> typing.List[State]:

    states = []
    ordering = create_initial_ordering(final_ordering)
    
    # Create the colouring matrix to work on:
    colouring = initial_colouring.copy()

    # Go through from top left to bottom right and swap each element, that needs swapping:
    for i in range(ordering.shape[0]):
        for j in range(ordering.shape[1]):
            # If the indices are not yet the same ...
            if (final_ordering[i, j, :] != ordering[i, j, :]).any():
                # ... swap the elements: If eventually some element with the original
                # coordinate of (k, l) has to end up in (i, j), then after the swap,
                # the ordering-matrix needs to agree with the final-ordering at (i, j)
                # and the former entry in (i, j) has to be moved to the (k, l)-position.
                k, l = final_ordering[i, j, :].tolist()
                
                # Find the source coordinates, i.e. where in ordering does the element (k, j)
                # currently reside?
                i_in, j_in = tuple(
                    np.argwhere(
                        (ordering[:, :, 0] == k) * (ordering[:, :, 1] == l)
                    ).flatten().tolist()
                )

                logger.debug(f'----------------------------Switching step {len(states)}-------------------------')
                logger.debug(f'At tile {(i, j)}, we expect to be tile {(k, l)}.')
                logger.debug(f'Tile {(k, l)} can currently be found at position {(i_in, j_in)}.')
                logger.debug(f'Before switching, ordering looks like this: {ordering}')

                # Do the switches, ...:
                tmp = ordering[i, j, :].copy()
                ordering[i, j, :] = ordering[i_in, j_in, :]
                ordering[i_in, j_in, :] = tmp.copy()

                tmp = colouring[i, j, :].copy()
                colouring[i, j, :] = colouring[i_in, j_in, :]
                colouring[i_in, j_in, :] = tmp.copy()

                # ... log the new position of the original (i, j)-element:
                logger.debug(f'After switching, ordering looks like this: {ordering}')
                logger.debug(f'After switching, colouring looks like this: {colouring}')

                # and generate a new State and add it to the list:
                states.append(
                    State(
                        ordering.copy(),
                        colouring.copy(),
                        swapped_elements=[(i, j), (i_in, j_in)],
                        idx=len(states),
                    )
                )

                logger.debug(f'Processed tile {(i, j)}: Swapped in for tile {(k, l)}, created state number {len(states) - 1}.')

            else:
                logger.debug(f'Processed tile {(i, j)}: Fixed tile, nothing to do.')

    return states
