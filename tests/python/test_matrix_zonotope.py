"""test_matrix_zonotope - cora.Zonotope.matrix: the constructor, the box and the stacking, products

and sums, the powers and the matrix exponential (cora.mat_exp).

CORA authors: Matthias Althoff, Tobias Ladner, Victor Gassmann, Niklas Kochdumper
Authors:       Claude Sonnet 5.5, Tobias Ladner, Claude Opus 5.5
Written:       03-November-2026
Last update:   18-November-2026
Reviewed:      ---
"""
# CORA: matrixSet/@matZonotope/matZonotope.m
# CORA: matrixSet/@matZonotope/center.m
# CORA: matrixSet/@matZonotope/contains.m
# CORA: matrixSet/@matZonotope/dependentTerms.m
# CORA: matrixSet/@matZonotope/expm.m
# CORA: matrixSet/@matZonotope/expmInd.m
# CORA: matrixSet/@matZonotope/expmMixed.m
# CORA: matrixSet/@matZonotope/expmOneParam.m
# CORA: matrixSet/@matZonotope/intervalMatrix.m
# CORA: matrixSet/@matZonotope/mpower.m
# CORA: matrixSet/@matZonotope/mtimes.m
# CORA: matrixSet/@matZonotope/norm.m
# CORA: matrixSet/@matZonotope/plus.m
# CORA: matrixSet/@matZonotope/powers.m
# CORA: matrixSet/@matZonotope/representsa.m
# CORA: matrixSet/@matZonotope/transpose.m
# CORA: matrixSet/@matZonotope/zonotope.m

import math
import unittest

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

# C = [[1, 2], [3, 4]] with the generators [[1, 0], [0, 1]] and [[0, 1], [1, 0]]
EXAMPLE = cora.Zonotope.matrix(np.array([[1.0, 2.0], [3.0, 4.0]]),
                               np.array([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]))


def point(C):
    """The matrix zonotope of the one matrix C."""
    C = np.asarray(C, dtype=float)
    return cora.Zonotope.matrix(C, np.zeros((0,) + C.shape))


def random_set(rng, n, h, spread):
    return cora.Zonotope.matrix(0.5 * rng.standard_normal((n, n)),
                                rng.uniform(-spread, spread, (h, n, n)))


def count(A):
    """The number of generator matrices."""
    return A.generators.shape[-3]


def member(A, b):
    return A.center + np.tensordot(b, A.generators, axes=1)


def exponential(M):
    """exp(M) by the Taylor series, enough terms for small matrices."""
    term, total = np.eye(len(M)), np.eye(len(M))
    for k in range(1, 40):
        term = term @ M / k
        total = total + term
    return total


# ============================================  MAIN  ============================================ #

class Construction(unittest.TestCase):
    def test_forms(self):
        self.assertEqual((EXAMPLE.elem_rank, EXAMPLE.center.shape, count(EXAMPLE)), (2, (2, 2), 2))
        np.testing.assert_array_equal(EXAMPLE.center, [[1, 2], [3, 4]])
        self.assertEqual(count(point(np.ones((2, 3)))), 0)

    def test_checks(self):
        with self.assertRaises(cora.errors.WrongInputInConstructor):
            cora.Zonotope.matrix(np.ones((2, 2)), np.ones((1, 2, 3)))


class Conversions(unittest.TestCase):
    # the matrices flatten column-major to the center and generators of a zonotope
    def test_stacked(self):
        Z = EXAMPLE.stacked()
        np.testing.assert_array_equal(Z.center, [1, 3, 2, 4])
        np.testing.assert_array_equal(Z.generators, [[1, 0, 0, 1], [0, 1, 1, 0]])

    def test_interval(self):
        B = EXAMPLE.interval()
        np.testing.assert_array_equal(B.inf, [[0, 1], [2, 3]])
        np.testing.assert_array_equal(B.sup, [[2, 3], [4, 5]])
        rng = np.random.default_rng(2)
        A = random_set(rng, 3, 4, 0.5)
        box = A.interval()
        for _ in range(20):
            self.assertTrue(box.contains(member(A, rng.uniform(-1, 1, 4)), tol=1e-12))

    def test_norm(self):
        self.assertAlmostEqual(EXAMPLE.norm(1), 8.0)
        self.assertAlmostEqual(EXAMPLE.norm(np.inf), 9.0)
        self.assertAlmostEqual(EXAMPLE.norm("fro"), np.sqrt(54.0))
        with self.assertRaises(cora.errors.WrongValue):
            EXAMPLE.norm(2)


# the predicates of a set of matrices
class Predicates(unittest.TestCase):
    # members are contained
    def test_contains(self):
        self.assertTrue(EXAMPLE.contains(member(EXAMPLE, np.array([0.5, -0.3])), tol=1e-9))
        self.assertTrue(EXAMPLE.contains(member(EXAMPLE, np.array([1.0, 1.0])), tol=1e-9))
        self.assertFalse(EXAMPLE.contains(member(EXAMPLE, np.array([2.0, 0.0])), tol=1e-9))
        inside, outside = member(EXAMPLE, np.array([0.1, 0.2])), member(EXAMPLE, np.array([2.0, 0.0]))
        self.assertTrue(EXAMPLE.contains(np.stack([inside, inside]), tol=1e-9))
        self.assertFalse(EXAMPLE.contains(np.stack([inside, outside]), tol=1e-9))
        with self.assertRaises(cora.errors.WrongValue):
            EXAMPLE.contains(inside, "sampling")
        with self.assertRaises(cora.errors.DimensionMismatch):
            EXAMPLE.contains(np.ones((1, 3)))

    def test_representsa(self):
        self.assertTrue(point(np.zeros((2, 2))).representsa("origin"))
        self.assertTrue(point(np.ones((2, 2))).representsa("point"))
        self.assertFalse(EXAMPLE.representsa("point"))


class Products(unittest.TestCase):
    def test_sum_and_matrices(self):
        S = EXAMPLE + EXAMPLE
        np.testing.assert_array_equal(S.center, [[2, 4], [6, 8]])
        self.assertEqual(count(S), 4)
        # sums with matrices
        np.testing.assert_array_equal((EXAMPLE + np.ones((2, 2))).center, [[2, 3], [4, 5]])
        np.testing.assert_array_equal((np.ones((2, 2)) + EXAMPLE).center, [[2, 3], [4, 5]])
        M = np.array([[1.0, 1.0], [0.0, 1.0]])
        np.testing.assert_array_equal((EXAMPLE @ M).center, [[1, 3], [3, 7]])
        np.testing.assert_array_equal((M @ EXAMPLE).center, [[4, 6], [3, 4]])
        np.testing.assert_array_equal((2.0 * EXAMPLE).center, [[2, 4], [6, 8]])
        np.testing.assert_array_equal((EXAMPLE * 2.0).generators[0], [[2, 0], [0, 2]])
        T = EXAMPLE.transpose()
        np.testing.assert_array_equal(T.center, [[1, 3], [2, 4]])

    # the product of two sets of matrices
    def test_product_of_sets(self):
        A = cora.Zonotope.matrix(np.eye(2), np.array([[[0.0, 1.0], [0.0, 0.0]]]))
        B = cora.Zonotope.matrix(2 * np.eye(2),
                                 np.array([[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]]))
        P = A @ B
        np.testing.assert_array_equal(P.center, 2 * np.eye(2))
        self.assertEqual(count(P), 5)
        np.testing.assert_array_equal(P.generators[0], [[0, 2], [0, 0]])
        np.testing.assert_array_equal(P.generators[2], [[0, 0], [0, 1]])
        rng = np.random.default_rng(3)
        Q, R = random_set(rng, 3, 2, 0.5), random_set(rng, 3, 3, 0.5)
        P = Q @ R
        # the weights of the product generators
        ca, cb = rng.uniform(-1, 1, 2), rng.uniform(-1, 1, 3)
        # the center, G_A(i) C_B, C_A G_B(j), G_A(i) G_B(j) with j fastest
        weights = np.r_[1.0, ca, cb, np.outer(ca, cb).reshape(-1)]
        value = P.center * weights[0] + np.tensordot(weights[1:], P.generators, axes=1)
        np.testing.assert_allclose(value, member(Q, ca) @ member(R, cb), atol=1e-12)

    def test_product_with_a_zonotope(self):
        A = cora.Zonotope.matrix(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([np.eye(2)]))
        Z = cora.Zonotope(np.array([1.0, 1.0]), np.array([[1.0, 0.0]]))
        P = A @ Z
        np.testing.assert_array_equal(P.center, [3, 7])
        np.testing.assert_array_equal(P.generators, [[1, 3], [1, 1], [1, 0]])

    def test_checks(self):
        with self.assertRaises(cora.errors.DimensionMismatch):
            EXAMPLE @ np.ones((3, 3))
        with self.assertRaises(cora.errors.NotSupported):
            point(np.array([[2.0]])) @ EXAMPLE

    def test_powers(self):
        P0 = EXAMPLE ** 0
        np.testing.assert_array_equal(P0.center, np.eye(2))
        self.assertEqual(count(P0), 0)
        # powers
        self.assertEqual(count(EXAMPLE ** 2), 8)
        np.testing.assert_array_equal((EXAMPLE ** 2).center, (EXAMPLE @ EXAMPLE).center)
        powers = cora.mat_exp.powers(EXAMPLE, 3)
        self.assertEqual(len(powers), 3)
        np.testing.assert_array_equal(powers[2].center, (EXAMPLE ** 3).center)
        rng = np.random.default_rng(4)
        A = random_set(rng, 3, 2, 0.4)
        later = cora.mat_exp.powers(A, 4, 2, A @ A)
        self.assertEqual(len(later), 3)
        M = member(A, rng.uniform(-1, 1, 2))
        self.assertTrue(later[2].interval().contains(M @ M @ M @ M, tol=1e-12))
        with self.assertRaises(cora.errors.WrongValue):
            EXAMPLE ** -1
        with self.assertRaises(cora.errors.WrongValue):
            cora.mat_exp.powers(EXAMPLE, 2, 1)


class Exponential(unittest.TestCase):
    def test_dependent_terms(self):
        A = cora.Zonotope.matrix(np.array([[2.0]]), np.array([[[1.0]]]))
        sq, H = cora.mat_exp.dependent_terms(A, 1.0)
        np.testing.assert_allclose(sq.center, [[4.5]])
        # squares and dependent terms
        np.testing.assert_allclose(sq.generators.reshape(-1), [4, 0.5])
        np.testing.assert_allclose(H.center, [[5.25]])
        np.testing.assert_allclose(H.generators.reshape(-1), [3, 0.25])
        rng = np.random.default_rng(5)
        S = random_set(rng, 3, 3, 0.5)
        sq, H = cora.mat_exp.dependent_terms(S, 0.7)
        self.assertEqual(count(sq), 9)
        b = rng.uniform(-1, 1, 3)
        coeff = np.r_[b, 2 * b ** 2 - 1, [b[i] * b[j] for i in range(3) for j in range(i + 1, 3)]]
        M = member(S, b)
        np.testing.assert_allclose(member(sq, coeff), 0.49 * M @ M, atol=1e-12)
        np.testing.assert_allclose(member(H, coeff), np.eye(3) + 0.7 * M + 0.245 * M @ M, atol=1e-12)

    # expm and expm_independent share the center; the refusals of an order and a shape
    def test_expm_and_its_refusals(self):
        A = random_set(np.random.default_rng(6), 3, 2, 0.3)
        np.testing.assert_array_equal(cora.mat_exp.expm(A).center,
                                      cora.mat_exp.expm_independent(A, 10).center)
        with self.assertRaises(cora.errors.WrongValue):
            cora.mat_exp.expm(A, -1)
        with self.assertRaises(cora.errors.DimensionMismatch):
            cora.mat_exp.expm(point(np.ones((1, 3))))

    # the mixed form of the exponential
    def test_expm_mixed(self):
        rng = np.random.default_rng(7)
        A = random_set(rng, 3, 2, 0.3)
        R = cora.mat_exp.expm_mixed(A, 1.0, 3, 9)
        box = R.enclosure.interval() + R.enclosure_rest
        for _ in range(20):
            self.assertTrue(box.contains(exponential(member(A, rng.uniform(-1, 1, 2))), tol=1e-12))
        self.assertEqual((len(R.powers), len(R.powers2)), (3, 6))
        with self.assertRaises(cora.errors.WrongValue):
            cora.mat_exp.expm_mixed(A, 1.0, 1, 9)
        with self.assertRaises(cora.errors.WrongValue):
            cora.mat_exp.expm_mixed(A, 0.0, 2, 9)

    def test_expm_one_param(self):
        A = cora.Zonotope.matrix(np.array([[1.0]]), np.array([[[1.0]]]))
        R = cora.mat_exp.expm_one_param(A, 1.0, 2)
        # the exponential of one parameter
        np.testing.assert_allclose(R.enclosure.center, [[2.75]])
        np.testing.assert_allclose(R.enclosure.generators.reshape(-1), [2, 0.25])
        self.assertIsNone(R.constant_input)
        rng = np.random.default_rng(8)
        S = random_set(rng, 3, 1, 0.5)
        r, N = 0.6, 6
        R = cora.mat_exp.expm_one_param(S, r, N)
        for b in (-1.0, 0.3, 1.0):
            M = member(S, np.array([b]))
            coeff = [b ** l if l % 2 else 2 * b ** l - 1 for l in range(1, N + 1)]
            taylor = sum(np.linalg.matrix_power(M, k) * r ** k / math.factorial(k) for k in range(N + 1))
            np.testing.assert_allclose(member(R.enclosure, np.array(coeff)), taylor, atol=1e-12)
            self.assertTrue(R.remainder.contains(exponential(r * M) - taylor, tol=1e-12))
        c, g = np.array([1.0, -2.0, 0.5]), np.array([0.3, 0.1, -0.4])
        U = cora.mat_exp.expm_one_param(S, r, N, cora.Zonotope(c, g[None, :]))
        V = cora.mat_exp.expm_one_param(S, r, N, c)
        self.assertEqual(U.constant_input.generators.shape, (N, 3))
        self.assertEqual(V.constant_input.generators.shape, (N, 3))
        with self.assertRaises(cora.errors.WrongValue):
            cora.mat_exp.expm_one_param(EXAMPLE, 1.0, 3)


# the class below differentiates the operations through torch autograd
class Gradients(unittest.TestCase):
    """Autograd through the operations equals central differences."""

    # torch is optional
    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("needs torch")
        if not cora.has_torch:
            self.skipTest("needs a build with torch")

    def check(self, f, center, generators):
        import torch
        center = torch.tensor(center, dtype=torch.float64, requires_grad=True)
        generators = torch.tensor(generators, dtype=torch.float64, requires_grad=True)

        def result(c, g):
            R = f(cora.Zonotope.matrix(c, g))
            if isinstance(R, torch.Tensor):
                return R
            if isinstance(R, cora.Interval):
                return torch.stack([R.inf, R.sup])
            return torch.cat([R.center.reshape(-1), R.generators.reshape(-1)])

        self.assertTrue(torch.autograd.gradcheck(result, (center, generators), atol=1e-5, rtol=1e-4))

    def test_gradients(self):
        center = [[1.0, 0.4], [0.3, 1.2]]
        generators = [[[0.2, 0.1], [0.05, 0.3]], [[0.1, -0.2], [0.15, 0.1]]]
        # the center and the gradients of the operations
        self.check(lambda A: A.center, center, generators)
        self.check(lambda A: A.transpose(), center, generators)
        self.check(lambda A: A + A, center, generators)
        self.check(lambda A: A @ A, center, generators)
        self.check(lambda A: A ** 3, center, generators)
        self.check(lambda A: cora.mat_exp.powers(A, 3)[2], center, generators)
        self.check(lambda A: A.stacked().generators, center, generators)
        # the box, the norm and the series
        self.check(lambda A: A.interval(), center, generators)
        self.check(lambda A: A.norm("fro"), center, generators)
        self.check(lambda A: cora.mat_exp.dependent_terms(A, 0.5)[1], center, generators)
        self.check(lambda A: cora.mat_exp.expm(A, 4), center, generators)
        self.check(lambda A: cora.mat_exp.expm_mixed(A, 1.0, 3, 6).enclosure, center, generators)

    def test_gradient_of_one_parameter(self):
        center = [[1.0, 0.4], [0.3, 1.2]]
        generators = [[[0.2, 0.1], [0.05, 0.3]]]
        self.check(lambda A: cora.mat_exp.expm_one_param(A, 0.5, 4).enclosure, center, generators)
        import torch
        u = torch.tensor([1.0, 2.0], dtype=torch.float64)
        self.check(lambda A: cora.mat_exp.expm_one_param(A, 0.5, 4, u).constant_input.generators,
                   center, generators)


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
