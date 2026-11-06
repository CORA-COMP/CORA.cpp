"""test_zonotope - the Zonotope class in Python: one set or a batch, on torch or numpy.

CORA authors: Tobias Ladner, Matthias Althoff, Mark Wetzlinger, Adrian Kulmburg,
              Severin Prenitzer, Niklas Kochdumper
Authors:       Claude Sonnet 5.5, Tobias Ladner, Claude Opus 5.5
Written:       29-September-2026
Last update:   04-November-2026
Reviewed:      ---
"""
# CORA: contSet/@contSet/linComb.m
# CORA: contSet/@zonotope/generateRandom.m
# CORA: contSet/@zonotope/interval.m
# CORA: contSet/@zonotope/mtimes.m
# CORA: contSet/@zonotope/plus.m
# CORA: contSet/@zonotope/randPoint_.m
# CORA: contSet/@zonotope/supportFunc_.m
# CORA: contSet/@zonotope/zonotope.m

import itertools
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

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

F64 = dict(dtype=torch.float64)


def brute_support(c, G, d):
    """max over the corners of the generator cube of d'(c + G b)."""
    m = G.shape[-1]
    return max(float(d @ (c + G @ torch.tensor(b, **F64))) for b in itertools.product([-1, 1], repeat=m))


def random_zonotope(n=3, m=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, generator=g, **F64), torch.randn(n, m, generator=g, **F64)


# ============================================  MAIN  ============================================ #

class Construction(unittest.TestCase):
    def test_torch_arrays_stay_torch(self):
        c, G = random_zonotope()
        Z = cora.Zonotope(c, G)
        self.assertIsInstance(Z.center, torch.Tensor)
        self.assertIsInstance(Z.generators, torch.Tensor)
        torch.testing.assert_close(Z.center, c)
        torch.testing.assert_close(Z.generators, G)
        self.assertEqual(Z.dim, 3)

    def test_numpy_arrays_stay_numpy(self):
        c, G = random_zonotope()
        Z = cora.Zonotope(c.numpy(), G.numpy())
        self.assertIsInstance(Z.center, np.ndarray)
        self.assertIsInstance(Z.generators, np.ndarray)
        np.testing.assert_allclose(Z.center, c.numpy())
        self.assertIsInstance(Z.support_function(np.ones(3)), float)

    def test_a_batch_lives_in_the_object(self):
        c, G = torch.randn(5, 3, **F64), torch.randn(5, 3, 4, **F64)
        Z = cora.Zonotope(c, G)
        self.assertEqual(tuple(Z.center.shape), (5, 3))
        self.assertEqual(tuple(Z.generators.shape), (5, 3, 4))
        self.assertEqual(Z.dim, 3)
        self.assertEqual(tuple(Z.center.shape), (5, 3))

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
                self.assertAlmostEqual(float(Z.support_function(d)), brute_support(c, G, d), places=10)

    def test_a_batch_gives_one_value_per_set(self):
        c, G = torch.randn(4, 3, **F64), torch.randn(4, 3, 2, **F64)
        d = torch.randn(3, **F64)
        rho = cora.Zonotope(c, G).support_function(d)
        self.assertEqual(tuple(rho.shape), (4,))
        for b in range(4):
            self.assertAlmostEqual(float(rho[b]), brute_support(c[b], G[b], d), places=10)


class Operations(unittest.TestCase):
    def test_mtimes_by_a_matrix(self):
        c, G = random_zonotope()
        M = torch.randn(3, 3, **F64)
        MZ = cora.Zonotope(c, G).linear_map(M)
        torch.testing.assert_close(MZ.center, M @ c)
        torch.testing.assert_close(MZ.generators, M @ G)

    def test_mtimes_by_an_interval_matrix_encloses_its_members(self):
        c, G = random_zonotope()
        center = torch.randn(3, 3, **F64)
        radius = 0.3 * torch.rand(3, 3, **F64)
        Z = cora.Zonotope(c, G)
        wide = Z.linear_map(cora.Interval.matrix(center - radius, center + radius))
        self.assertEqual(wide.generators.shape[-1], G.shape[-1] + 3)  # one generator per dimension
        for _ in range(20):
            M = center + radius * (2 * torch.rand(3, 3, **F64) - 1)
            d = torch.randn(3, **F64)
            self.assertGreaterEqual(float(wide.support_function(d)),
                                    float(Z.linear_map(M).support_function(d)) - 1e-10)

    def test_plus_is_the_minkowski_sum(self):
        (c1, G1), (c2, G2) = random_zonotope(3, 3, 1), random_zonotope(3, 2, 2)
        S = cora.Zonotope(c1, G1).plus(cora.Zonotope(c2, G2))
        self.assertEqual(S.generators.shape[-1], 5)
        d = torch.randn(3, **F64)
        parts = cora.Zonotope(c1, G1).support_function(d) + cora.Zonotope(c2, G2).support_function(d)
        self.assertAlmostEqual(float(S.support_function(d)), float(parts))

    def test_linear_combination_encloses_both(self):
        c, G = random_zonotope()
        M = 0.5 * torch.randn(3, 3, **F64)
        Z = cora.Zonotope(c, G)
        mapped = Z.linear_map(M)
        Z2 = cora.Zonotope(mapped.center + torch.tensor([1.0, -1.0, 0.5], **F64), mapped.generators)
        hull = Z.linear_combination(Z2)
        self.assertEqual(hull.generators.shape[-1], 2 * G.shape[-1] + 1)
        for _ in range(20):
            d = torch.randn(3, **F64)
            self.assertGreaterEqual(float(hull.support_function(d)),
                                    max(float(Z.support_function(d)), float(Z2.support_function(d))) - 1e-10)

    def test_interval_is_the_box_around_it(self):
        c, G = random_zonotope()
        I = cora.Zonotope(c, G).interval()
        torch.testing.assert_close(I.inf, c - G.abs().sum(-1))
        torch.testing.assert_close(I.sup, c + G.abs().sum(-1))


class RandomSets(unittest.TestCase):
    def test_rand_point_lies_in_the_set(self):
        c, G = random_zonotope()
        Z = cora.Zonotope(c, G)
        P = Z.rand_point(1000, cora.Rng(3))
        self.assertEqual(tuple(P.shape), (3, 1000))
        for _ in range(20):
            d = torch.randn(3, **F64)
            self.assertLessEqual(float((d @ P).max()), float(Z.support_function(d)) + 1e-10)

    def test_extreme_points_are_corners(self):
        c, G = random_zonotope(3, 3)
        P = cora.Zonotope(c, G).rand_point(30, cora.Rng(0), type="extreme")
        signs = itertools.product([-1.0, 1.0], repeat=3)
        corners = torch.stack([c + G @ torch.tensor(b, **F64) for b in signs])
        nearest = (P.T[:, None, :] - corners[None]).norm(dim=-1).min(dim=1).values
        self.assertLess(float(nearest.max()), 1e-10)

    def test_a_seed_repeats_a_draw(self):
        c, G = random_zonotope()
        Z = cora.Zonotope(c, G)
        self.assertTrue(torch.equal(Z.rand_point(10, cora.Rng(1)), Z.rand_point(10, cora.Rng(1))))
        self.assertFalse(torch.equal(Z.rand_point(10, cora.Rng(1)), Z.rand_point(10, cora.Rng(2))))

    def test_an_unknown_type_is_described(self):
        Z = cora.Zonotope(*random_zonotope())
        with self.assertRaises(ValueError) as caught:
            Z.rand_point(3, cora.Rng(0), type="corner")
        self.assertIn("standard", str(caught.exception))
        self.assertIn("extreme", str(caught.exception))

    def test_a_batch_draws_in_each_set(self):
        c, G = torch.randn(3, 2, **F64), torch.randn(3, 2, 2, **F64)
        Z = cora.Zonotope(c, G)
        P = Z.rand_point(200, cora.Rng(0))
        self.assertEqual(tuple(P.shape), (3, 2, 200))
        d = torch.randn(2, **F64)
        best = (d @ P).max(-1).values
        self.assertTrue(bool((best <= Z.support_function(d) + 1e-10).all()))

    def test_generate_random_follows_the_current_backend(self):
        with cora.backend("eigen"):
            Z = cora.Zonotope.generate_random(cora.Rng(0), 3, nr_generators=4)
            self.assertIsInstance(Z.center, np.ndarray)
            self.assertEqual(Z.generators.shape, (3, 4))
        Z = cora.Zonotope.generate_random(cora.Rng(0), 3, nr_generators=4)
        self.assertIsInstance(Z.center, torch.Tensor)

    def test_generate_random_takes_the_options_of_cora(self):
        rng = cora.Rng(0)
        Z = cora.Zonotope.generate_random(rng, center=[1.0, -2.0, 3.0])
        self.assertEqual(Z.generators.shape, (3, 6))
        self.assertTrue(np.allclose(np.asarray(Z.center), [1.0, -2.0, 3.0]))
        for distribution in ("uniform", "exp", "gamma"):
            W = cora.Zonotope.generate_random(rng, 2, nr_generators=5, distribution=distribution)
            self.assertEqual(W.generators.shape, (2, 5))
        with self.assertRaises(cora.errors.WrongValue):
            cora.Zonotope.generate_random(rng, 2, distribution="normal")

    def test_reduce_and_compact_default_as_the_generic_ones(self):
        Z = cora.Zonotope.generate_random(cora.Rng(1), 2, nr_generators=7)
        self.assertEqual(Z.reduce().generators.shape, cora.ContSet.reduce(Z).generators.shape)
        self.assertEqual(Z.compact().generators.shape, cora.ContSet.compact(Z).generators.shape)


class Stack(unittest.TestCase):
    def test_stack_batches_zonotopes_of_different_size(self):
        a = cora.Zonotope(cora.Tensor([1.0, 0.0]), cora.eye(2))
        b = cora.Zonotope(cora.Tensor([0.0, 2.0]), cora.Tensor([[1.0, 0.5, 0.0], [0.0, 0.5, 1.0]]))
        S = cora.Zonotope.stack([a, b])
        self.assertEqual(tuple(S.center.shape), (2, 2))
        self.assertEqual(tuple(S.generators.shape), (2, 2, 3))
        torch.testing.assert_close(S.center[1], b.center)
        torch.testing.assert_close(S.generators[0, :, :2], a.generators)
        self.assertEqual(float(S.generators[0, :, 2].abs().sum()), 0.0)  # padding
        d = cora.Tensor([1.0, -1.0])
        torch.testing.assert_close(S.support_function(d), torch.stack([a.support_function(d),
                                                                       b.support_function(d)]))

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

# ----------------------------------------  END OF CODE  ----------------------------------------- #
