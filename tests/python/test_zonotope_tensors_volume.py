"""test_zonotope_tensors_volume - tensor_multiplication, tensor_multiplication_zono,

underapproximate, volume and zonotope_norm of Zonotope in Python, on numpy and Eigen alone.

CORA authors: Matthias Althoff, Adrian Kulmburg
Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
Written:       03-November-2026
Last update:   ---
Reviewed:      ---
"""
# CORA: contSet/@zonotope/tensorMultiplication.m
# CORA: contSet/@zonotope/tensorMultiplication_zono.m
# CORA: contSet/@zonotope/underapproximate.m
# CORA: contSet/@zonotope/volume_.m
# CORA: contSet/@zonotope/zonotopeNorm.m

import itertools
import unittest
import warnings

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

def zonotope(c, G):
    return cora.Zonotope(np.array(c, dtype=float), np.array(G, dtype=float))


def triples(m):
    """CORA's lists for m generators: the ordered triples, the sorted ones, the column of each."""
    comb = [list(t) for t in itertools.combinations_with_replacement(range(m), 3)]
    perm = [list(t) for t in itertools.product(range(m), repeat=3)]
    return perm, comb, [comb.index(sorted(t)) for t in perm]


# ============================================  MAIN  ============================================ #

class TensorMultiplication(unittest.TestCase):
    def test_by_hand(self):
        Z = zonotope([0], [[1, 2]])
        perm, comb, mapping = triples(2)
        R = Z.tensor_multiplication(2 * np.ones((1, 1, 1, 1)), perm, comb, mapping)
        np.testing.assert_allclose(R.generators, [[2, 12, 24, 16]])
        S = Z.tensor_multiplication_zono(np.ones((1, 1, 1, 1)), [0.5 * np.ones((1, 1, 1, 1))],
                                         perm, comb, mapping)
        np.testing.assert_allclose(S.generators, [[1, 6, 12, 8, 0.5, 3, 6, 4]])

    # the refusals of a wrong tensor or list
    def test_errors(self):
        Z = zonotope([0], [[1, 2]])
        perm, comb, mapping = triples(2)
        one = np.ones((1, 1, 1, 1))
        with self.assertRaises(cora.errors.NotSupported):
            Z.tensor_multiplication(np.ones((1, 1, 1)), perm, comb, mapping)
        with self.assertRaises(cora.errors.DimensionMismatch):
            Z.tensor_multiplication(np.ones((1, 1, 1, 2)), perm, comb, mapping)
        with self.assertRaises(cora.errors.WrongValue):
            zonotope([1], [[1, 2]]).tensor_multiplication(one, perm, comb, mapping)
        with self.assertRaises(cora.errors.WrongValue):
            Z.tensor_multiplication(one, [[0, 0, 5]] + perm[1:], comb, mapping)


class Underapproximate(unittest.TestCase):
    def test_cora_unit_test(self):
        Z = zonotope([-4, 1], [[-3, -2, -1], [2, 3, 4]])
        np.testing.assert_allclose(Z.underapproximate(), [[-10, 2, -10, 2], [10, -8, 10, -8]], atol=1e-12)
        # the under-approximation
        np.testing.assert_allclose(Z.underapproximate(np.array([[1.0, 1], [0, 1]])),
                                   [[2, -10, -4, -4], [-8, 10, 6, -4]], atol=1e-12)

    def test_errors(self):
        Z = zonotope([0, 0], [[1, 0, 1], [0, 1, 1]])
        with self.assertRaises(cora.errors.DimensionMismatch):
            Z.underapproximate(np.ones((3, 1)))


class Volume(unittest.TestCase):
    def test_cora_unit_test(self):
        Z = zonotope([-4, 1], [[-3, -2, -1], [2, 3, 4]])
        self.assertAlmostEqual(Z.volume(), 80)
        # volumes
        self.assertAlmostEqual(Z.volume("alamo"), 48.9897948556635612)

    def test_degenerate(self):
        self.assertEqual(zonotope([0, 0, 0], [[1, 0], [0, 1], [0, 0]]).volume(), 0)
        self.assertEqual(zonotope([0, 0], [[1, 2], [2, 4]]).volume(), 0)
        self.assertEqual(zonotope([0, 0], np.zeros((2, 0))).volume("alamo"), 0)

    def test_the_volume_is_the_area_and_a_determinant(self):
        rng = np.random.default_rng(3)
        for m in (2, 3, 5):
            Z = zonotope(rng.standard_normal(2), rng.uniform(-1, 1, (2, m)))
            # the area of the polygon: the zonogon has 2m corners, the sum of 2 |det| over pairs
            area = 4 * sum(abs(np.linalg.det(Z.generators[:, list(s)]))
                           for s in itertools.combinations(range(m), 2))
            self.assertAlmostEqual(Z.volume(), area, places=9)
            self.assertLessEqual(Z.volume("alamo"), Z.volume() + 1e-9)
        G = rng.uniform(-1, 1, (3, 3))
        self.assertAlmostEqual(zonotope(np.zeros(3), G).volume(), 8 * abs(np.linalg.det(G)), places=9)

    def test_a_batch(self):
        rng = np.random.default_rng(4)
        A = zonotope(np.zeros(2), rng.uniform(-1, 1, (2, 4)))
        flat = zonotope(np.zeros(2), [[1, 2, 3, 1], [2, 4, 6, 2]])
        batch = cora.Zonotope.stack([A, flat])
        for method in ("exact", "alamo"):
            np.testing.assert_allclose(batch.volume(method), [A.volume(method), 0], atol=1e-12)

    def test_errors(self):
        Z = zonotope([0, 0], [[1, 0, 1], [0, 1, 1]])
        with self.assertRaises(cora.errors.WrongValue):
            Z.volume("fast")
        with self.assertRaises(cora.errors.WrongValue):
            Z.volume("exact", 0)
        with self.assertRaises(cora.errors.NotSupported):
            # reduce volumes
            Z.volume("reduce", 1)


class ZonotopeNorm(unittest.TestCase):
    G = [[1, -2, 2, 0], [-1, 1, 0, 1]]

    def test_cora_unit_test(self):
        for c in ([0, 0], [1, -2]):
            Z = zonotope(c, self.G)
            self.assertAlmostEqual(Z.zonotope_norm(np.array([5.0, 3])), 2.2)
            self.assertAlmostEqual(Z.zonotope_norm(np.array([-5.0, 3])), 1)
        point = zonotope([0, 0], np.zeros((2, 0)))
        self.assertEqual(point.zonotope_norm(np.zeros(2)), 0)
        self.assertEqual(point.zonotope_norm(np.array([1.0, 0])), np.inf)

    # a shifted zonotope warns
    def test_a_shifted_zonotope_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            zonotope([1, 0], [[1], [0]]).zonotope_norm_minimizer(np.array([1.0, 0]))
        self.assertEqual(len(caught), 1)

    def test_errors(self):
        Z = zonotope([0, 0], [[1, 0], [0, 1]])
        with self.assertRaises(cora.errors.NotSupported):
            cora.Zonotope.stack([Z, Z]).zonotope_norm(np.array([1.0, 0]))
        with self.assertRaises(cora.errors.DimensionMismatch):
            Z.zonotope_norm(np.ones(3))


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
