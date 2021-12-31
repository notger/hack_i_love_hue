import unittest
import numpy as np
from src.image_manipulation import Image
from src.solution_base import Solution
from src.solver_naive import naive_method


class TestNaiveMethod(unittest.TestCase):

    def test_naive_method(self):
        solution = Solution(Image('images/test1.jpeg'))
        solution.solve(naive_method)

        # Make sure, that all states are sane:
        self.assertTrue(solution.initial_state.is_sane())
        self.assertTrue(solution.final_state.is_sane())
        for k, state in enumerate(solution.steps):
            if not state.is_sane():
                print(f'Step {k}, state: {state}')

            self.assertTrue(state.is_sane())

        # Has the final state reached the final ordering?
        self.assertTrue((solution.final_ordering == solution.steps[-1].ordering).all())

    def test_naive_method_synthetic(self):
        initial_colouring = np.asarray(
            [
                [[0, 0, 0], [240, 240, 240], [40, 40, 40]], 
                [[120, 120, 120], [200, 200, 200], [80, 80, 80]], 
                [[160, 160, 160], [100, 100, 100], [255, 255, 255]]
            ]
        )
        final_ordering = np.asarray(
            [
                [[0, 0], [0, 2], [1, 2]],
                [[2, 1], [1, 0], [2, 0]],
                [[1, 1], [0, 1], [2, 2]]
            ]
        )

        states = naive_method(initial_colouring, final_ordering)

        for k, state in enumerate(states):
            if not state.is_sane():
                print(f'Step {k}, state: {state}')

            self.assertTrue(state.is_sane())

        # Has the final state reached the final ordering?
        self.assertTrue((final_ordering == states[-1].ordering).all())
