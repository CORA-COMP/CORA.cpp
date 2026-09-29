"""test_operators - the operators of Zonotope and Interval in Python, as CORA writes them."""
import unittest

import numpy as np
import torch

import cora


def box():
    return cora.Zonotope(cora.Tensor([1.0, 2.0]), cora.Tensor([[0.5, 0.0], [0.0, 0.25]]))


class ZonotopeOperators(unittest.TestCase):
    def test_a_matrix_times_a_zonotope_is_the_linear_map(self):
        M = cora.Tensor([[0.0, 1.0], [-1.0, 0.0]])
        for Z in (M * box(), M @ box(), box().mtimes(M)):
            torch.testing.assert_close(Z.c, torch.tensor([2.0, -1.0], dtype=torch.float64))
        torch.testing.assert_close((M * box()).G, box().mtimes(M).G)

    def test_a_number_scales(self):
        for Z in (-2.0 * box(), box() * -2.0, -2 * box()):
            torch.testing.assert_close(Z.c, torch.tensor([-2.0, -4.0], dtype=torch.float64))
            torch.testing.assert_close(Z.G[0, 0], torch.tensor(-1.0, dtype=torch.float64))
        torch.testing.assert_close((-box()).c, torch.tensor([-1.0, -2.0], dtype=torch.float64))

    def test_zonotopes_add_and_a_vector_translates(self):
        S = box() + box()
        self.assertEqual(tuple(S.G.shape), (2, 4))
        v = cora.Tensor([1.0, -1.0])
        for Z in (box() + v, v + box()):
            torch.testing.assert_close(Z.c, torch.tensor([2.0, 1.0], dtype=torch.float64))
        torch.testing.assert_close((box() - v).c, torch.tensor([0.0, 3.0], dtype=torch.float64))

    def test_an_interval_matrix_widens(self):
        M = cora.Interval.matrix(-torch.ones(2, 2, dtype=torch.float64),
                                 torch.ones(2, 2, dtype=torch.float64))
        self.assertEqual(tuple((M * box()).G.shape), (2, 4))

    def test_numpy_arrays_work_on_the_eigen_backend(self):
        cora.setBackend("eigen")
        try:
            Z = cora.Zonotope(np.array([1.0, 2.0]), np.eye(2))
            R = np.array([[0.0, 1.0], [-1.0, 0.0]]) * Z
            np.testing.assert_allclose(R.c, [2.0, -1.0])
            np.testing.assert_allclose((Z + np.array([1.0, 1.0])).c, [2.0, 3.0])
        finally:
            cora.setBackend("torch")

    def test_something_else_is_a_type_error(self):
        with self.assertRaises(TypeError):
            box() + "a string"
        with self.assertRaises(TypeError):
            "a string" * box()
        with self.assertRaises(TypeError):
            box() * cora.Tensor([[1.0, 0.0], [0.0, 1.0]])


class IntervalOperators(unittest.TestCase):
    def setUp(self):
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
        J = cora.Tensor([[0.0, 1.0], [1.0, 0.0]]) * self.I
        torch.testing.assert_close(J.inf, torch.tensor([0.0, -1.0], dtype=torch.float64))

    def test_an_interval_matrix_times_a_zonotope(self):
        M = cora.Interval.matrix(-torch.ones(2, 2, dtype=torch.float64),
                                 torch.ones(2, 2, dtype=torch.float64))
        Z = cora.Zonotope(cora.Tensor([1.0, 2.0]), cora.Tensor([[0.5, 0.0], [0.0, 0.25]]))
        self.assertEqual(tuple((M * Z).G.shape), (2, 4))


if __name__ == "__main__":
    unittest.main()
