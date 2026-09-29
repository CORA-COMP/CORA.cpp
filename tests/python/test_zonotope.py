"""test_zonotope - the Zonotope class in Python: one set or a batch, on torch or numpy."""
import itertools
import unittest

import numpy as np
import torch

import cora

F64 = dict(dtype=torch.float64)


def brute_support(c, G, d):
    """max over the corners of the generator cube of d'(c + G b)."""
    m = G.shape[-1]
    return max(float(d @ (c + G @ torch.tensor(b, **F64))) for b in itertools.product([-1, 1], repeat=m))


def random_zonotope(n=3, m=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, generator=g, **F64), torch.randn(n, m, generator=g, **F64)


class Construction(unittest.TestCase):
    def test_torch_arrays_stay_torch(self):
        c, G = random_zonotope()
        Z = cora.Zonotope(c, G)
        self.assertIsInstance(Z.c, torch.Tensor)
        self.assertIsInstance(Z.G, torch.Tensor)
        torch.testing.assert_close(Z.c, c)
        torch.testing.assert_close(Z.G, G)
        self.assertEqual(Z.dim(), 3)

    def test_numpy_arrays_stay_numpy(self):
        c, G = random_zonotope()
        Z = cora.Zonotope(c.numpy(), G.numpy())
        self.assertIsInstance(Z.c, np.ndarray)
        self.assertIsInstance(Z.G, np.ndarray)
        np.testing.assert_allclose(Z.c, c.numpy())
        self.assertIsInstance(Z.supportFunc(np.ones(3)), float)

    def test_a_batch_lives_in_the_object(self):
        c, G = torch.randn(5, 3, **F64), torch.randn(5, 3, 4, **F64)
        Z = cora.Zonotope(c, G)
        self.assertEqual(tuple(Z.c.shape), (5, 3))
        self.assertEqual(tuple(Z.G.shape), (5, 3, 4))
        self.assertEqual(Z.dim(), 3)
        self.assertEqual(tuple(Z.center().shape), (5, 3))

    def test_the_two_backends_do_not_mix(self):
        c, G = random_zonotope()
        with self.assertRaises(ValueError) as caught:
            cora.Zonotope(c, G).plus(cora.Zonotope(c.numpy(), G.numpy()))
        self.assertIn("backend", str(caught.exception))


class SupportFunc(unittest.TestCase):
    def test_against_the_corners(self):
        for seed in range(4):
            c, G = random_zonotope(3, 4, seed)
            Z = cora.Zonotope(c, G)
            for _ in range(5):
                d = torch.randn(3, **F64)
                self.assertAlmostEqual(float(Z.supportFunc(d)), brute_support(c, G, d), places=10)

    def test_a_batch_gives_one_value_per_set(self):
        c, G = torch.randn(4, 3, **F64), torch.randn(4, 3, 2, **F64)
        d = torch.randn(3, **F64)
        rho = cora.Zonotope(c, G).supportFunc(d)
        self.assertEqual(tuple(rho.shape), (4,))
        for b in range(4):
            self.assertAlmostEqual(float(rho[b]), brute_support(c[b], G[b], d), places=10)


class Operations(unittest.TestCase):
    def test_mtimes_by_a_matrix(self):
        c, G = random_zonotope()
        M = torch.randn(3, 3, **F64)
        MZ = cora.Zonotope(c, G).mtimes(M)
        torch.testing.assert_close(MZ.c, M @ c)
        torch.testing.assert_close(MZ.G, M @ G)

    def test_mtimes_by_an_interval_matrix_encloses_its_members(self):
        c, G = random_zonotope()
        center = torch.randn(3, 3, **F64)
        radius = 0.3 * torch.rand(3, 3, **F64)
        Z = cora.Zonotope(c, G)
        wide = Z.mtimes(cora.Interval.matrix(center - radius, center + radius))
        self.assertEqual(wide.G.shape[-1], G.shape[-1] + 3)  # one generator per dimension
        for _ in range(20):
            M = center + radius * (2 * torch.rand(3, 3, **F64) - 1)
            d = torch.randn(3, **F64)
            self.assertGreaterEqual(float(wide.supportFunc(d)), float(Z.mtimes(M).supportFunc(d)) - 1e-10)

    def test_plus_is_the_minkowski_sum(self):
        (c1, G1), (c2, G2) = random_zonotope(3, 3, 1), random_zonotope(3, 2, 2)
        S = cora.Zonotope(c1, G1).plus(cora.Zonotope(c2, G2))
        self.assertEqual(S.G.shape[-1], 5)
        d = torch.randn(3, **F64)
        parts = cora.Zonotope(c1, G1).supportFunc(d) + cora.Zonotope(c2, G2).supportFunc(d)
        self.assertAlmostEqual(float(S.supportFunc(d)), float(parts))

    def test_linComb_encloses_both(self):
        c, G = random_zonotope()
        M = 0.5 * torch.randn(3, 3, **F64)
        Z = cora.Zonotope(c, G)
        mapped = Z.mtimes(M)
        Z2 = cora.Zonotope(mapped.c + torch.tensor([1.0, -1.0, 0.5], **F64), mapped.G)
        hull = Z.linComb(Z2)
        self.assertEqual(hull.G.shape[-1], 2 * G.shape[-1] + 1)
        for _ in range(20):
            d = torch.randn(3, **F64)
            self.assertGreaterEqual(float(hull.supportFunc(d)),
                                    max(float(Z.supportFunc(d)), float(Z2.supportFunc(d))) - 1e-10)

    def test_interval_is_the_box_around_it(self):
        c, G = random_zonotope()
        I = cora.Zonotope(c, G).interval()
        torch.testing.assert_close(I.inf, c - G.abs().sum(-1))
        torch.testing.assert_close(I.sup, c + G.abs().sum(-1))


class RandomSets(unittest.TestCase):
    def test_randPoint_lies_in_the_set(self):
        c, G = random_zonotope()
        Z = cora.Zonotope(c, G)
        P = Z.randPoint(1000, cora.Rng(3))
        self.assertEqual(tuple(P.shape), (3, 1000))
        for _ in range(20):
            d = torch.randn(3, **F64)
            self.assertLessEqual(float((d @ P).max()), float(Z.supportFunc(d)) + 1e-10)

    def test_extreme_points_are_corners(self):
        c, G = random_zonotope(3, 3)
        P = cora.Zonotope(c, G).randPoint(30, cora.Rng(0), type="extreme")
        signs = itertools.product([-1.0, 1.0], repeat=3)
        corners = torch.stack([c + G @ torch.tensor(b, **F64) for b in signs])
        nearest = (P.T[:, None, :] - corners[None]).norm(dim=-1).min(dim=1).values
        self.assertLess(float(nearest.max()), 1e-10)

    def test_a_seed_repeats_a_draw(self):
        c, G = random_zonotope()
        Z = cora.Zonotope(c, G)
        self.assertTrue(torch.equal(Z.randPoint(10, cora.Rng(1)), Z.randPoint(10, cora.Rng(1))))
        self.assertFalse(torch.equal(Z.randPoint(10, cora.Rng(1)), Z.randPoint(10, cora.Rng(2))))

    def test_an_unknown_type_is_described(self):
        Z = cora.Zonotope(*random_zonotope())
        with self.assertRaises(ValueError) as caught:
            Z.randPoint(3, cora.Rng(0), type="corner")
        self.assertIn("standard", str(caught.exception))
        self.assertIn("extreme", str(caught.exception))

    def test_a_batch_draws_in_each_set(self):
        c, G = torch.randn(3, 2, **F64), torch.randn(3, 2, 2, **F64)
        Z = cora.Zonotope(c, G)
        P = Z.randPoint(200, cora.Rng(0))
        self.assertEqual(tuple(P.shape), (3, 2, 200))
        d = torch.randn(2, **F64)
        best = (d @ P).max(-1).values
        self.assertTrue(bool((best <= Z.supportFunc(d) + 1e-10).all()))

    def test_generateRandom_follows_the_current_backend(self):
        cora.setBackend("eigen")
        try:
            Z = cora.Zonotope.generateRandom(3, 4, cora.Rng(0))
            self.assertIsInstance(Z.c, np.ndarray)
            self.assertEqual(Z.G.shape, (3, 4))
        finally:
            cora.setBackend("torch")
        self.assertIsInstance(cora.Zonotope.generateRandom(3, 4, cora.Rng(0)).c, torch.Tensor)


class Stack(unittest.TestCase):
    def test_stack_batches_zonotopes_of_different_size(self):
        a = cora.Zonotope(cora.Tensor([1.0, 0.0]), cora.eye(2))
        b = cora.Zonotope(cora.Tensor([0.0, 2.0]), cora.Tensor([[1.0, 0.5, 0.0], [0.0, 0.5, 1.0]]))
        S = cora.Zonotope.stack([a, b])
        self.assertEqual(tuple(S.c.shape), (2, 2))
        self.assertEqual(tuple(S.G.shape), (2, 2, 3))
        torch.testing.assert_close(S.c[1], b.c)
        torch.testing.assert_close(S.G[0, :, :2], a.G)
        self.assertEqual(float(S.G[0, :, 2].abs().sum()), 0.0)  # padding
        d = cora.Tensor([1.0, -1.0])
        torch.testing.assert_close(S.supportFunc(d), torch.stack([a.supportFunc(d), b.supportFunc(d)]))

    def test_stack_of_intervals(self):
        a = cora.Interval(cora.Tensor([0.0, 0.0]), cora.Tensor([1.0, 1.0]))
        b = cora.Interval(cora.Tensor([-1.0, 0.0]), cora.Tensor([2.0, 3.0]))
        S = cora.Interval.stack([a, b])
        torch.testing.assert_close(S.inf[1], b.inf)
        torch.testing.assert_close(S.sup[0], a.sup)

    def test_an_empty_stack_is_described(self):
        with self.assertRaises(ValueError):
            cora.Zonotope.stack([])


if __name__ == "__main__":
    unittest.main()
