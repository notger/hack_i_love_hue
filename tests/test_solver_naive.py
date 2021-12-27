import unittest
from src.image_manipulation import Image
from src.solution_base import Solution
from src.solver_naive import naive_method


class TestNaiveMethod(unittest.TestCase):

    def setUp(self) -> None:
        self.image_path = 'images/test1.jpeg'
        self.solution = Solution(Image(self.image_path))
        return super().setUp()

    def test_naive_method(self):
        self.solution.solve(naive_method)

        # Make sure, that all states are sane:
        self.assertTrue(self.solution.initial_state.is_sane())
        self.assertTrue(self.solution.final_state.is_sane())
        for state in self.solution.steps:
            print(state)
            self.assertTrue(state.is_sane())
