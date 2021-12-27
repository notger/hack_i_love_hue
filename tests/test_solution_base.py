import unittest
import numpy as np
from src.solution_base import State, create_initial_ordering


class TestState(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()

    def test_create_initial_ordering(self):
        initial_ordering = create_initial_ordering(np.zeros((4, 3, 2)))

        # Every pair of coordinates should only show up once:
        flattened_coordinates = initial_ordering.flatten().reshape((-1, 2)).tolist()
        flattened_coordinates_set = set(
            [(i, j) for i, j in flattened_coordinates]  # We have to do this, as lists are not hashable.
        )
        self.assertTrue(len(flattened_coordinates) == len(flattened_coordinates_set))

    def test_state_integrity_check(self):
        # Create an ordering and colouring which are sane:
        N_i = 3
        N_j = 3
        ordering = create_initial_ordering(np.zeros((N_i, N_j, 2)))
        colouring = np.random.choice(range(N_i * N_j * 3), N_i * N_j * 3, replace=False).reshape((N_i, N_j, 3))
        
        # Now create borked ones:
        ordering_broken = ordering.copy()
        ordering_broken[-1, -1, :] = ordering_broken[0, 1, :]
        colouring_broken = colouring.copy()
        colouring_broken[-1, -1, :] = colouring_broken[0, 1, :]

        self.assertTrue(State(ordering, colouring).is_sane())
        self.assertFalse(State(ordering_broken, colouring).is_sane())
        self.assertFalse(State(ordering, colouring_broken).is_sane())
