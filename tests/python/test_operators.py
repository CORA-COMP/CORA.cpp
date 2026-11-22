"""test_operators - the operators of Zonotope and Interval in Python, as CORA writes them.

CORA authors: Matthias Althoff
Authors:       Claude Sonnet 5.5, Tobias Ladner
Written:       29-September-2026
Last update:   03-November-2026
Reviewed:      ---
"""
# CORA: contSet/@interval/mtimes.m
# CORA: contSet/@interval/plus.m
# CORA: contSet/@zonotope/mtimes.m
# CORA: contSet/@zonotope/plus.m

import unittest

import numpy as np


try:
    import torch
except ImportError:  # numpy-only build: test_numpy.py covers it
    raise unittest.SkipTest("needs torch") from None

import cora


if not cora.has_torch:  # torch is installed, but this build has no torch backend
    raise unittest.SkipTest("needs a build with torch")

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

class ZonotopeOperators(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("torch"))  # torch is never the default

    def test_a_matrix_at_a_zonotope_is_the_linear_map(self):
        M = cora.Tensor([[0.0, 1.0], [-1.0, 0.0]])
        for Z in (M @ box(), box().linear_map(M), box() @ M.T):
            torch.testing.assert_close(Z.center, torch.tensor([2.0, -1.0], dtype=torch.float64))
        torch.testing.assert_close((M @ box()).generators, box().linear_map(M).generators)
        # Z @ M maps every point x to x M, as numpy's x @ M
        torch.testing.assert_close((box() @ M).generators, box().generators @ M)

    def test_star_with_a_matrix_names_the_at_operator(self):
        M = cora.Tensor([[0.0, 1.0], [-1.0, 0.0]])
        for product in (lambda: M * box(), lambda: box() * M, lambda: box() * box()):
            with self.assertRaises(TypeError) as caught:
                product()
            self.assertIn("@", str(caught.exception))

    def test_a_number_scales(self):
        for Z in (-2.0 * box(), box() * -2.0, -2 * box()):
            torch.testing.assert_close(Z.center, torch.tensor([-2.0, -4.0], dtype=torch.float64))
            torch.testing.assert_close(Z.generators[0, 0], torch.tensor(-1.0, dtype=torch.float64))
        torch.testing.assert_close((-box()).center, torch.tensor([-1.0, -2.0], dtype=torch.float64))

    def test_zonotopes_add_and_a_vector_translates(self):
        S = box() + box()
        self.assertEqual(tuple(S.generators.shape), (4, 2))
        v = cora.Tensor([1.0, -1.0])
        for Z in (box() + v, v + box()):
            torch.testing.assert_close(Z.center, torch.tensor([2.0, 1.0], dtype=torch.float64))
        torch.testing.assert_close((box() - v).center, torch.tensor([0.0, 3.0], dtype=torch.float64))

    def test_an_interval_matrix_widens(self):
        # a [-1, 1] matrix mixes the two dimensions: 2 generators of the box become 4
        M = cora.Interval.matrix(-torch.ones(2, 2, dtype=torch.float64),
                                 torch.ones(2, 2, dtype=torch.float64))
        self.assertEqual(tuple((M @ box()).generators.shape), (4, 2))

    def test_numpy_arrays_work_on_the_eigen_backend(self):
        with cora.backend("eigen"):
            Z = cora.Zonotope(np.array([1.0, 2.0]), np.eye(2))
            R = np.array([[0.0, 1.0], [-1.0, 0.0]]) @ Z
            np.testing.assert_allclose(R.center, [2.0, -1.0])
            np.testing.assert_allclose((Z + np.array([1.0, 1.0])).center, [2.0, 3.0])

    def test_something_else_is_a_type_error(self):
        with self.assertRaises(TypeError):
            box() + "a string"
        with self.assertRaises(TypeError):
            "a string" * box()
        with self.assertRaises(TypeError):
            box() * cora.Tensor([[1.0, 0.0], [0.0, 1.0]])


class IntervalOperators(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("torch"))  # torch is never the default
        self.I = cora.Interval(cora.Tensor([-1.0, 0.0]), cora.Tensor([1.0, 2.0]))

    def test_intervals_add_and_a_vector_translates(self):
        S = self.I + self.I
        torch.testing.assert_close(S.sup, torch.tensor([2.0, 4.0], dtype=torch.float64))
        v = cora.Tensor([1.0, 1.0])
        torch.testing.assert_close((self.I + v).inf, torch.tensor([0.0, 1.0], dtype=torch.float64))
        torch.testing.assert_close((v + self.I).sup, torch.tensor([2.0, 3.0], dtype=torch.float64))
        torch.testing.assert_close((self.I - v).sup, torch.tensor([0.0, 1.0], dtype=torch.float64))

    def test_a_negative_number_swaps_the_bounds(self):
        N = -self.I
        torch.testing.assert_close(N.inf, torch.tensor([-1.0, -2.0], dtype=torch.float64))
        torch.testing.assert_close(N.sup, torch.tensor([1.0, 0.0], dtype=torch.float64))
        torch.testing.assert_close((2 * self.I).sup, torch.tensor([2.0, 4.0], dtype=torch.float64))

    def test_a_matrix_gives_the_hull_of_the_image(self):
        J = cora.Tensor([[0.0, 1.0], [1.0, 0.0]]) @ self.I
        torch.testing.assert_close(J.inf, torch.tensor([0.0, -1.0], dtype=torch.float64))

    # the same widening for a zonotope with generators: 2 become 4
    def test_an_interval_matrix_times_a_zonotope(self):
        M = cora.Interval.matrix(-torch.ones(2, 2, dtype=torch.float64),
                                 torch.ones(2, 2, dtype=torch.float64))
        Z = cora.Zonotope(cora.Tensor([1.0, 2.0]), cora.Tensor([[0.5, 0.0], [0.0, 0.25]]))
        self.assertEqual(tuple((M @ Z).generators.shape), (4, 2))


# -----------------------------------------  AUXILIARY  ------------------------------------------ #

def box():
    return cora.Zonotope(cora.Tensor([1.0, 2.0]), cora.Tensor([[0.5, 0.0], [0.0, 0.25]]))


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
