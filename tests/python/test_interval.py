"""test_interval - the Interval class in Python: a box, or a matrix of intervals."""
import unittest

import numpy as np
import torch

import cora

F64 = dict(dtype=torch.float64)


class Interval(unittest.TestCase):
    def test_bounds_center_and_rad(self):
        I = cora.Interval(torch.tensor([-1.0, 0.0, 2.0], **F64), torch.tensor([3.0, 0.5, 2.0], **F64))
        torch.testing.assert_close(I.inf, torch.tensor([-1.0, 0.0, 2.0], **F64))
        torch.testing.assert_close(I.center(), torch.tensor([1.0, 0.25, 2.0], **F64))
        torch.testing.assert_close(I.rad(), torch.tensor([2.0, 0.25, 0.0], **F64))
        self.assertEqual(I.dim(), 3)

    def test_numpy_boxes_stay_numpy(self):
        I = cora.Interval(np.array([0.0, 1.0]), np.array([2.0, 3.0]))
        self.assertIsInstance(I.inf, np.ndarray)
        self.assertIsInstance(I.supportFunc(np.array([1.0, 1.0])), float)

    def test_supportFunc_is_the_best_corner(self):
        lo, hi = torch.tensor([-1.0, 0.0], **F64), torch.tensor([2.0, 3.0], **F64)
        I = cora.Interval(lo, hi)
        directions = ([1.0, 1.0], [-1.0, 2.0], [-2.0, -1.0])
        for d in (torch.tensor(d, **F64) for d in directions):
            self.assertAlmostEqual(float(I.supportFunc(d)), float(torch.maximum(d * lo, d * hi).sum()))

    def test_contains(self):
        I = cora.Interval(torch.tensor([0.0, -1.0], **F64), torch.tensor([2.0, 1.0], **F64))
        self.assertTrue(I.contains(torch.tensor([1.0, 0.0], **F64)))
        self.assertTrue(I.contains(torch.tensor([0.0, 1.0], **F64)))  # a corner belongs to the box
        self.assertFalse(I.contains(torch.tensor([2.1, 0.0], **F64)))

    def test_mtimes_is_the_hull_of_the_image(self):
        I = cora.Interval(torch.tensor([-1.0, 0.0, 1.0], **F64), torch.tensor([1.0, 2.0, 3.0], **F64))
        M = torch.randn(3, 3, **F64)
        image = I.mtimes(M)
        for _ in range(50):
            x = I.inf + torch.rand(3, **F64) * (I.sup - I.inf)
            self.assertTrue(image.contains(M @ x))

    def test_plus_adds_the_bounds(self):
        S = cora.Interval(torch.tensor([0.0, 1.0], **F64), torch.tensor([1.0, 3.0], **F64)).plus(
            cora.Interval(torch.tensor([-2.0, 0.5], **F64), torch.tensor([0.0, 0.5], **F64)))
        torch.testing.assert_close(S.inf, torch.tensor([-2.0, 1.5], **F64))
        torch.testing.assert_close(S.sup, torch.tensor([1.0, 3.5], **F64))

    def test_randPoint_fills_the_box(self):
        I = cora.Interval(torch.tensor([0.0, -1.0], **F64), torch.tensor([2.0, 1.0], **F64))
        P = I.randPoint(2000, cora.Rng(4))
        self.assertEqual(tuple(P.shape), (2, 2000))
        self.assertTrue(bool((P.T >= I.inf).all() and (P.T <= I.sup).all()))
        self.assertLess(float((P.mean(-1) - I.center()).abs().max()), 0.1)

    def test_generateRandom(self):
        I = cora.Interval.generateRandom(4, cora.Rng(2))
        self.assertEqual(I.dim(), 4)
        self.assertTrue(bool((I.inf <= I.sup).all()))

    def test_an_interval_can_be_a_matrix(self):
        # The correction matrix of a system is an Interval whose bounds are matrices.
        sys = cora.LinearSys(torch.tensor([[-0.2, 1.0], [-1.0, -0.2]], **F64))
        F = sys.correctionMatrixState(0.1, 8)
        self.assertEqual(tuple(F.inf.shape), (2, 2))
        self.assertTrue(bool((F.inf <= F.sup).all()))
        # MATLAB CORA's taylorMatrices gives F.sup(1, 1) = 0.0012000413732381331 for this system.
        self.assertAlmostEqual(float(F.sup[0, 0]), 0.0012000413732381331, places=14)


if __name__ == "__main__":
    unittest.main()
