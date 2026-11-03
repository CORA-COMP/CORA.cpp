"""test_mat_zonotope - cora.MatZonotope: the constructor, the box and the stacking, products and
sums, the powers and the matrix exponential."""
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
# PROPERTY: matrixSet/@matZonotope/matZonotope.m
# PROPERTY: matrixSet/@matZonotope/center.m
# PROPERTY: matrixSet/@matZonotope/contains.m
# PROPERTY: matrixSet/@matZonotope/dependentTerms.m
# PROPERTY: matrixSet/@matZonotope/expm.m
# PROPERTY: matrixSet/@matZonotope/expmInd.m
# PROPERTY: matrixSet/@matZonotope/expmMixed.m
# PROPERTY: matrixSet/@matZonotope/expmOneParam.m
# PROPERTY: matrixSet/@matZonotope/intervalMatrix.m
# PROPERTY: matrixSet/@matZonotope/mpower.m
# PROPERTY: matrixSet/@matZonotope/mtimes.m
# PROPERTY: matrixSet/@matZonotope/norm.m
# PROPERTY: matrixSet/@matZonotope/plus.m
# PROPERTY: matrixSet/@matZonotope/powers.m
# PROPERTY: matrixSet/@matZonotope/representsa.m
# PROPERTY: matrixSet/@matZonotope/transpose.m
# PROPERTY: matrixSet/@matZonotope/zonotope.m

import math
import unittest

import numpy as np

import cora

# C = [[1, 2], [3, 4]] with the generators [[1, 0], [0, 1]] and [[0, 1], [1, 0]]
EXAMPLE = cora.MatZonotope(np.array([[1.0, 2.0], [3.0, 4.0]]),
                           np.array([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]))


def random_set(rng, n, h, spread):
    return cora.MatZonotope(0.5 * rng.standard_normal((n, n)), rng.uniform(-spread, spread, (h, n, n)))


def member(A, b):
    return A.center + np.tensordot(b, A.generators, axes=1)


def exponential(M):
    """exp(M) by the Taylor series, enough terms for small matrices."""
    term, total = np.eye(len(M)), np.eye(len(M))
    for k in range(1, 40):
        term = term @ M / k
        total = total + term
    return total


class Construction(unittest.TestCase):
    def test_forms(self):
        self.assertEqual((EXAMPLE.rows, EXAMPLE.cols, EXAMPLE.num_generators), (2, 2, 2))
        np.testing.assert_array_equal(EXAMPLE.center, [[1, 2], [3, 4]])
        point = cora.MatZonotope(np.ones((2, 3)))
        self.assertEqual((point.rows, point.cols, point.num_generators), (2, 3, 0))
        empty = cora.MatZonotope()
        self.assertEqual((empty.rows, empty.cols, empty.num_generators), (0, 0, 0))

    def test_checks(self):
        with self.assertRaises(cora.errors.WrongInputInConstructor):
            cora.MatZonotope(np.ones((2, 2)), np.ones((1, 2, 3)))
        with self.assertRaises(cora.errors.WrongValue):
            cora.MatZonotope(np.array([[1.0, np.nan]]))
        with cora.checks(False):
            self.assertEqual(cora.MatZonotope(np.ones((2, 2))).rows, 2)

    def test_center_is_the_member_at_zero(self):
        rng = np.random.default_rng(1)
        A = random_set(rng, 3, 2, 0.4)
        np.testing.assert_allclose(member(A, np.zeros(2)), A.center)


class Conversions(unittest.TestCase):
    def test_zonotope(self):
        Z = EXAMPLE.zonotope()
        np.testing.assert_array_equal(Z.center, [1, 3, 2, 4])
        np.testing.assert_array_equal(Z.generators, [[1, 0], [0, 1], [0, 1], [1, 0]])

    def test_interval_matrix(self):
        B = EXAMPLE.interval_matrix()
        np.testing.assert_array_equal(B.infimum, [[0, 1], [2, 3]])
        np.testing.assert_array_equal(B.interval.sup, [[2, 3], [4, 5]])
        rng = np.random.default_rng(2)
        A = random_set(rng, 3, 4, 0.5)
        box = A.interval_matrix()
        for _ in range(20):
            self.assertTrue(box.contains(member(A, rng.uniform(-1, 1, 4)), tol=1e-12))

    def test_norm(self):
        self.assertAlmostEqual(EXAMPLE.norm(1), 8.0)
        self.assertAlmostEqual(EXAMPLE.norm(np.inf), 9.0)
        self.assertAlmostEqual(EXAMPLE.norm("fro"), np.sqrt(54.0))
        with self.assertRaises(cora.errors.WrongValue):
            EXAMPLE.norm(2)


class Predicates(unittest.TestCase):
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
        self.assertTrue(cora.MatZonotope(np.zeros((2, 2))).representsa("origin"))
        self.assertTrue(cora.MatZonotope(np.ones((2, 2))).representsa("point"))
        self.assertFalse(EXAMPLE.representsa("point"))


class Products(unittest.TestCase):
    def test_sum_and_matrices(self):
        S = EXAMPLE + EXAMPLE
        np.testing.assert_array_equal(S.center, [[2, 4], [6, 8]])
        self.assertEqual(S.num_generators, 4)
        np.testing.assert_array_equal((EXAMPLE + np.ones((2, 2))).center, [[2, 3], [4, 5]])
        np.testing.assert_array_equal((np.ones((2, 2)) + EXAMPLE).center, [[2, 3], [4, 5]])
        M = np.array([[1.0, 1.0], [0.0, 1.0]])
        np.testing.assert_array_equal((EXAMPLE @ M).center, [[1, 3], [3, 7]])
        np.testing.assert_array_equal((M @ EXAMPLE).center, [[4, 6], [3, 4]])
        np.testing.assert_array_equal((2.0 * EXAMPLE).center, [[2, 4], [6, 8]])
        np.testing.assert_array_equal((EXAMPLE * 2.0).generators[0], [[2, 0], [0, 2]])
        T = EXAMPLE.transpose()
        np.testing.assert_array_equal(T.center, [[1, 3], [2, 4]])

    def test_product_of_sets(self):
        A = cora.MatZonotope(np.eye(2), np.array([[[0.0, 1.0], [0.0, 0.0]]]))
        B = cora.MatZonotope(2 * np.eye(2), np.array([[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]]))
        P = A @ B
        np.testing.assert_array_equal(P.center, 2 * np.eye(2))
        self.assertEqual(P.num_generators, 5)
        np.testing.assert_array_equal(P.generators[0], [[0, 2], [0, 0]])
        np.testing.assert_array_equal(P.generators[3], [[0, 0], [0, 1]])
        rng = np.random.default_rng(3)
        Q, R = random_set(rng, 3, 2, 0.5), random_set(rng, 3, 3, 0.5)
        P = Q @ R
        ca, cb = rng.uniform(-1, 1, 2), rng.uniform(-1, 1, 3)
        weights = np.outer(np.r_[1.0, ca], np.r_[1.0, cb]).T.reshape(-1)  # A's index fastest
        point = P.center * weights[0] + np.tensordot(weights[1:], P.generators, axes=1)
        np.testing.assert_allclose(point, member(Q, ca) @ member(R, cb), atol=1e-12)

    def test_product_with_a_zonotope(self):
        A = cora.MatZonotope(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([np.eye(2)]))
        Z = cora.Zonotope(np.array([1.0, 1.0]), np.array([[1.0], [0.0]]))
        P = A @ Z
        np.testing.assert_array_equal(P.center, [3, 7])
        np.testing.assert_array_equal(P.generators, [[1, 1, 1], [3, 1, 0]])

    def test_checks(self):
        with self.assertRaises(cora.errors.DimensionMismatch):
            EXAMPLE @ np.ones((3, 3))
        with self.assertRaises(cora.errors.NotSupported):
            cora.MatZonotope(np.array([[2.0]])) @ EXAMPLE
        with self.assertRaises(cora.errors.DimensionMismatch):
            EXAMPLE + np.ones((1, 3))

    def test_powers(self):
        P0 = EXAMPLE ** 0
        np.testing.assert_array_equal(P0.center, np.eye(2))
        self.assertEqual(P0.num_generators, 0)
        self.assertEqual((EXAMPLE ** 2).num_generators, 8)
        np.testing.assert_array_equal((EXAMPLE ** 2).center, (EXAMPLE @ EXAMPLE).center)
        powers = EXAMPLE.powers(3)
        self.assertEqual(len(powers), 3)
        np.testing.assert_array_equal(powers[2].center, (EXAMPLE ** 3).center)
        rng = np.random.default_rng(4)
        A = random_set(rng, 3, 2, 0.4)
        later = A.powers(4, 2, A @ A)
        self.assertEqual(len(later), 3)
        M = member(A, rng.uniform(-1, 1, 2))
        self.assertTrue(later[2].interval_matrix().contains(M @ M @ M @ M, tol=1e-12))
        with self.assertRaises(cora.errors.WrongValue):
            EXAMPLE ** -1
        with self.assertRaises(cora.errors.WrongValue):
            EXAMPLE.powers(2, 1)


class Exponential(unittest.TestCase):
    def test_dependent_terms(self):
        A = cora.MatZonotope(np.array([[2.0]]), np.array([[[1.0]]]))
        sq, H = A.dependent_terms(1.0)
        np.testing.assert_allclose(sq.center, [[4.5]])
        np.testing.assert_allclose(sq.generators.reshape(-1), [4, 0.5])
        np.testing.assert_allclose(H.center, [[5.25]])
        np.testing.assert_allclose(H.generators.reshape(-1), [3, 0.25])
        rng = np.random.default_rng(5)
        S = random_set(rng, 3, 3, 0.5)
        sq, H = S.dependent_terms(0.7)
        self.assertEqual(sq.num_generators, 9)
        b = rng.uniform(-1, 1, 3)
        coeff = np.r_[b, 2 * b ** 2 - 1, [b[i] * b[j] for i in range(3) for j in range(i + 1, 3)]]
        M = member(S, b)
        np.testing.assert_allclose(member(sq, coeff), 0.49 * M @ M, atol=1e-12)
        np.testing.assert_allclose(member(H, coeff), np.eye(3) + 0.7 * M + 0.245 * M @ M, atol=1e-12)

    def test_expm_encloses_the_exponentials(self):
        rng = np.random.default_rng(6)
        A = random_set(rng, 3, 2, 0.3)
        for E in (A.expm(), A.expm_independent(7)):
            box = E.interval_matrix()
            for _ in range(20):
                self.assertTrue(box.contains(exponential(member(A, rng.uniform(-1, 1, 2))), tol=1e-12))
        np.testing.assert_array_equal(A.expm().center, A.expm_independent(10).center)
        with self.assertRaises(cora.errors.WrongValue):
            A.expm(-1)
        with self.assertRaises(cora.errors.DimensionMismatch):
            cora.MatZonotope(np.ones((1, 3))).expm()

    def test_expm_mixed(self):
        rng = np.random.default_rng(7)
        A = random_set(rng, 3, 2, 0.3)
        R = A.expm_mixed(1.0, 3, 9)
        total = R.enclosure.interval_matrix().interval + R.enclosure_rest.interval
        box = cora.IntervalMatrix(total)
        for _ in range(20):
            self.assertTrue(box.contains(exponential(member(A, rng.uniform(-1, 1, 2))), tol=1e-12))
        self.assertEqual((len(R.powers), len(R.powers2)), (3, 6))
        with self.assertRaises(cora.errors.WrongValue):
            A.expm_mixed(1.0, 1, 9)
        with self.assertRaises(cora.errors.WrongValue):
            A.expm_mixed(0.0, 2, 9)

    def test_expm_one_param(self):
        A = cora.MatZonotope(np.array([[1.0]]), np.array([[[1.0]]]))
        R = A.expm_one_param(1.0, 2)
        np.testing.assert_allclose(R.enclosure.center, [[2.75]])
        np.testing.assert_allclose(R.enclosure.generators.reshape(-1), [2, 0.25])
        self.assertIsNone(R.constant_input)
        rng = np.random.default_rng(8)
        S = random_set(rng, 3, 1, 0.5)
        r, N = 0.6, 6
        R = S.expm_one_param(r, N)
        for b in (-1.0, 0.3, 1.0):
            M = member(S, np.array([b]))
            coeff = [b ** l if l % 2 else 2 * b ** l - 1 for l in range(1, N + 1)]
            taylor = sum(np.linalg.matrix_power(M, k) * r ** k / math.factorial(k) for k in range(N + 1))
            np.testing.assert_allclose(member(R.enclosure, np.array(coeff)), taylor, atol=1e-12)
            self.assertTrue(R.remainder.contains(exponential(r * M) - taylor, tol=1e-12))
        c, g = np.array([1.0, -2.0, 0.5]), np.array([0.3, 0.1, -0.4])
        U = S.expm_one_param(r, N, cora.Zonotope(c, g[:, None]))
        V = S.expm_one_param(r, N, c)
        self.assertEqual(U.constant_input.generators.shape, (3, N))
        self.assertEqual(V.constant_input.generators.shape, (3, N))
        with self.assertRaises(cora.errors.WrongValue):
            EXAMPLE.expm_one_param(1.0, 3)


class Gradients(unittest.TestCase):
    """Autograd through the operations equals central differences."""

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
            R = f(cora.MatZonotope(c, g))
            if isinstance(R, torch.Tensor):
                return R
            if isinstance(R, cora.IntervalMatrix):
                return torch.stack([R.interval.inf, R.interval.sup])
            return torch.cat([R.center.reshape(-1), R.generators.reshape(-1)])

        self.assertTrue(torch.autograd.gradcheck(result, (center, generators), atol=1e-5, rtol=1e-4))

    def test_gradients(self):
        center = [[1.0, 0.4], [0.3, 1.2]]
        generators = [[[0.2, 0.1], [0.05, 0.3]], [[0.1, -0.2], [0.15, 0.1]]]
        self.check(lambda A: A.center, center, generators)
        self.check(lambda A: A.transpose(), center, generators)
        self.check(lambda A: A + A, center, generators)
        self.check(lambda A: A @ A, center, generators)
        self.check(lambda A: A ** 3, center, generators)
        self.check(lambda A: A.powers(3)[2], center, generators)
        self.check(lambda A: A.zonotope().generators, center, generators)
        self.check(lambda A: A.interval_matrix(), center, generators)
        self.check(lambda A: A.norm("fro"), center, generators)
        self.check(lambda A: A.dependent_terms(0.5)[1], center, generators)
        self.check(lambda A: A.expm(4), center, generators)
        self.check(lambda A: A.expm_mixed(1.0, 3, 6).enclosure, center, generators)

    def test_gradient_of_one_parameter(self):
        center = [[1.0, 0.4], [0.3, 1.2]]
        generators = [[[0.2, 0.1], [0.05, 0.3]]]
        self.check(lambda A: A.expm_one_param(0.5, 4).enclosure, center, generators)
        import torch
        u = torch.tensor([1.0, 2.0], dtype=torch.float64)
        self.check(lambda A: A.expm_one_param(0.5, 4, u).constant_input.generators, center, generators)


if __name__ == "__main__":
    unittest.main()
