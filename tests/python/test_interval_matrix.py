"""test_interval_matrix - cora.IntervalMatrix: the constructors, the bounds, the norms, the
squares and the matrix exponential."""
# CORA: matrixSet/@intervalMatrix/abs.m
# CORA: matrixSet/@intervalMatrix/center.m
# CORA: matrixSet/@intervalMatrix/contains.m
# CORA: matrixSet/@intervalMatrix/dependentTerms.m
# CORA: matrixSet/@intervalMatrix/exactSquare.m
# CORA: matrixSet/@intervalMatrix/expm.m
# CORA: matrixSet/@intervalMatrix/expmInd.m
# CORA: matrixSet/@intervalMatrix/expmMixed.m
# CORA: matrixSet/@intervalMatrix/exponentialRemainder.m
# CORA: matrixSet/@intervalMatrix/infNorm.m
# CORA: matrixSet/@intervalMatrix/infimum.m
# CORA: matrixSet/@intervalMatrix/intervalMatrix.m
# CORA: matrixSet/@intervalMatrix/mpower.m
# CORA: matrixSet/@intervalMatrix/mtimes.m
# CORA: matrixSet/@intervalMatrix/norm.m
# PROPERTY: matrixSet/@intervalMatrix/abs.m
# PROPERTY: matrixSet/@intervalMatrix/center.m
# PROPERTY: matrixSet/@intervalMatrix/contains.m
# PROPERTY: matrixSet/@intervalMatrix/dependentTerms.m
# PROPERTY: matrixSet/@intervalMatrix/exactSquare.m
# PROPERTY: matrixSet/@intervalMatrix/expm.m
# PROPERTY: matrixSet/@intervalMatrix/expmInd.m
# PROPERTY: matrixSet/@intervalMatrix/expmMixed.m
# PROPERTY: matrixSet/@intervalMatrix/exponentialRemainder.m
# PROPERTY: matrixSet/@intervalMatrix/infNorm.m
# PROPERTY: matrixSet/@intervalMatrix/infimum.m
# PROPERTY: matrixSet/@intervalMatrix/intervalMatrix.m
# PROPERTY: matrixSet/@intervalMatrix/mpower.m
# PROPERTY: matrixSet/@intervalMatrix/mtimes.m
# PROPERTY: matrixSet/@intervalMatrix/norm.m

import unittest

import numpy as np

import cora

# [[[1, 2], [-1, 1]], [[0, 3], [2, 2]]]
EXAMPLE = cora.IntervalMatrix(cora.Interval.matrix(np.array([[1.0, -1.0], [0.0, 2.0]]),
                                                   np.array([[2.0, 1.0], [3.0, 2.0]])))


def random_matrix(rng, n, spread):
    return cora.IntervalMatrix(0.5 * rng.standard_normal((n, n)), rng.uniform(0, spread, (n, n)))


def member(A, rng):
    inf, sup = A.interval.inf, A.interval.sup
    return inf + (sup - inf) * rng.uniform(0, 1, inf.shape)


def exponential(M):
    """exp(M) by the Taylor series, enough terms for small matrices."""
    term, total = np.eye(len(M)), np.eye(len(M))
    for k in range(1, 40):
        term = term @ M / k
        total = total + term
    return total


class Bounds(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("eigen"))

    def test_constructors(self):
        A = cora.IntervalMatrix(np.array([[1.0, -2.0], [0.0, 3.0]]), np.array([[0.5, -1.0], [0.0, 2.0]]))
        np.testing.assert_allclose(A.interval.inf, [[0.5, -3.0], [0.0, 1.0]])
        np.testing.assert_allclose(A.interval.sup, [[1.5, -1.0], [0.0, 5.0]])
        P = cora.IntervalMatrix(np.array([[1.0, 2.0]]))
        np.testing.assert_allclose(P.infimum, [[1.0, 2.0]])
        self.assertEqual((EXAMPLE.rows, EXAMPLE.cols), (2, 2))
        with self.assertRaises(cora.errors.WrongInputInConstructor):
            cora.IntervalMatrix(np.ones((2, 2)), np.ones((1, 3)))

    def test_abs_center_infimum(self):
        np.testing.assert_allclose(EXAMPLE.abs(), [[2, 1], [3, 2]])
        np.testing.assert_allclose(EXAMPLE.center, [[1.5, 0], [1.5, 2]])
        np.testing.assert_allclose(EXAMPLE.infimum, [[1, -1], [0, 2]])
        rng = np.random.default_rng(1)
        A = random_matrix(rng, 3, 1.0)
        self.assertTrue(A.contains(A.center))
        self.assertTrue(all(np.all(np.abs(member(A, rng)) <= A.abs() + 1e-15) for _ in range(20)))

    def test_contains(self):
        out = np.array([[1.0, -1.0], [3.5, 2.0]])
        self.assertTrue(EXAMPLE.contains(EXAMPLE.center))
        self.assertFalse(EXAMPLE.contains(out))
        self.assertTrue(EXAMPLE.contains(out, tol=0.2))
        self.assertFalse(EXAMPLE.contains(out, tol=0.05))
        self.assertTrue(EXAMPLE.contains(EXAMPLE.center, "approx"))
        with self.assertRaises(cora.errors.DimensionMismatch):
            EXAMPLE.contains(np.ones((1, 3)))
        with self.assertRaises(cora.errors.WrongValue):
            EXAMPLE.contains(EXAMPLE.center, "other")
        with self.assertRaises(cora.errors.WrongValue):
            EXAMPLE.contains(EXAMPLE.center, tol=-1.0)

    def test_norms(self):
        self.assertEqual(EXAMPLE.infinity_norm(), 5.0)
        self.assertEqual(EXAMPLE.norm(1), 5.0)
        self.assertEqual(EXAMPLE.norm(float("inf")), 5.0)
        self.assertAlmostEqual(EXAMPLE.norm("fro"), np.sqrt(18.0))
        with self.assertRaises(cora.errors.WrongValue):
            EXAMPLE.norm()  # CORA's default 2 is refused


class Squares(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("eigen"))

    def test_exact_square(self):
        S = EXAMPLE.exact_square().interval
        np.testing.assert_allclose(S.inf, [[-2, -4], [0, 1]])
        np.testing.assert_allclose(S.sup, [[7, 4], [12, 7]])
        with self.assertRaises(cora.errors.DimensionMismatch):
            cora.IntervalMatrix(np.ones((1, 3))).exact_square()

    def test_dependent_terms(self):
        M = np.array([[1.0, 2.0], [3.0, 4.0]])
        square, H = cora.IntervalMatrix(M).dependent_terms(0.5)
        np.testing.assert_allclose(square.center, 0.25 * M @ M)
        np.testing.assert_allclose(H.center, np.eye(2) + 0.5 * M + 0.125 * M @ M)
        rng = np.random.default_rng(2)
        A = random_matrix(rng, 3, 0.7)
        square, H = A.dependent_terms(1.0)
        for _ in range(30):
            X = member(A, rng)
            self.assertTrue(square.contains(X @ X, tol=1e-12))
            self.assertTrue(H.contains(np.eye(3) + X + X @ X / 2, tol=1e-12))

    def test_matrix_power(self):
        A = cora.IntervalMatrix(cora.Interval.matrix(np.array([[1.0, 0.0], [0.0, 1.0]]),
                                                     np.array([[2.0, 1.0], [0.0, 1.0]])))
        np.testing.assert_allclose(A.matrix_power(2).interval.sup, [[4, 3], [0, 1]])
        np.testing.assert_allclose((A ** 3).interval.sup, [[8, 7], [0, 1]])
        np.testing.assert_allclose(A.matrix_power(0).interval.inf, np.eye(2))
        S = cora.IntervalMatrix(cora.Interval.matrix(np.array([[-1.0]]), np.array([[1.0]])))
        np.testing.assert_allclose(S.matrix_power(2).interval.inf, [-1.0])
        with self.assertRaises(cora.errors.WrongValue):
            A.matrix_power(-1)

    def test_products(self):
        M = np.array([[1.0, 0.0], [2.0, -1.0]])
        np.testing.assert_allclose((EXAMPLE @ M).interval.inf, [[-1, -1], [4, -2]])
        np.testing.assert_allclose((M @ EXAMPLE).interval.sup, [[2, 1], [4, 0]])
        np.testing.assert_allclose((-2.0 * EXAMPLE).interval.inf, [[-4, -2], [-6, -4]])
        np.testing.assert_allclose((EXAMPLE * 2.0).interval.sup, [[4, 2], [6, 4]])
        P = EXAMPLE @ EXAMPLE
        self.assertTrue(P.contains(EXAMPLE.center @ EXAMPLE.center))
        Z = cora.Zonotope(np.array([1.0, 1.0]), np.eye(2))
        R = EXAMPLE @ Z
        np.testing.assert_allclose(R.center, [1.5, 3.5])
        self.assertEqual((EXAMPLE @ cora.EmptySet(2)).dim, 2)
        with self.assertRaises(cora.errors.DimensionMismatch):
            EXAMPLE @ np.ones((3, 1))
        with self.assertRaises(TypeError):
            EXAMPLE @ "text"


class Exponential(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("eigen"))

    def test_remainder(self):
        S = cora.IntervalMatrix(np.array([[0.0]]), np.array([[1.0]]))
        W = np.e - 2.5
        E = S.exponential_remainder(2).interval
        np.testing.assert_allclose([E.inf[0], E.sup[0]], [-W, W])  # a 1 x 1 bound reads as a vector
        with self.assertRaises(cora.errors.DimensionMismatch):
            cora.IntervalMatrix(np.ones((1, 3))).exponential_remainder(2)

    def test_expm_variants_enclose_the_exponential(self):
        rng = np.random.default_rng(3)
        A = random_matrix(rng, 3, 0.3)
        results = [A.expm(), A.expm(8), A.expm(8, r=0.5), A.expm(8, r=2.0),
                   A.expm_independent(8)]
        for _ in range(20):
            E = exponential(member(A, rng))
            for R in results:
                self.assertTrue(R.enclosure.contains(E, tol=1e-12))
        tail = A.expm(8, r=1.0, initial_order=3, initial_power=A.matrix_power(3))
        X = member(A, rng)
        self.assertTrue(tail.enclosure.contains(exponential(X) - np.eye(3) - X - X @ X / 2, tol=1e-12))
        self.assertEqual((results[0].first_order, len(results[0].powers)), (2, 9))
        zero = cora.IntervalMatrix(np.zeros((2, 2))).expm(5)
        np.testing.assert_allclose(zero.enclosure.center, np.eye(2), atol=1e-14)
        with self.assertRaises(cora.errors.WrongValue):
            A.expm(4, r=0.0)
        with self.assertRaises(cora.errors.WrongValue):
            A.expm(4, initial_order=2)

    def test_expm_independent_from_a_power(self):
        rng = np.random.default_rng(4)
        A = random_matrix(rng, 2, 0.3)
        R = A.expm_independent(8, initial_order=3, initial_power=A.matrix_power(3))
        X = member(A, rng)
        self.assertTrue(R.enclosure.contains(exponential(X) - np.eye(2) - X - X @ X / 2, tol=1e-12))
        with self.assertRaises(cora.errors.WrongValue):
            A.expm_independent(-1)

    def test_expm_mixed(self):
        rng = np.random.default_rng(5)
        A = random_matrix(rng, 3, 0.3)
        R = A.expm_mixed(1.0, 3, 9)
        total = cora.IntervalMatrix(R.enclosure.interval + R.enclosure_rest.interval)
        for _ in range(20):
            self.assertTrue(total.contains(exponential(member(A, rng)), tol=1e-12))
        self.assertEqual((len(R.powers), len(R.powers2)), (3, 6))
        with self.assertRaises(cora.errors.WrongValue):
            A.expm_mixed(1.0, 1, 9)


class Gradients(unittest.TestCase):
    """Autograd through the operations equals central differences (piecewise ones off their kinks)."""

    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("needs torch")
        if not cora.has_torch:
            self.skipTest("needs a build with torch")

    def check(self, f, center, delta):
        import torch
        center = torch.tensor(center, dtype=torch.float64, requires_grad=True)
        delta = torch.tensor(delta, dtype=torch.float64, requires_grad=True)

        def result(c, d):
            R = f(cora.IntervalMatrix(c, d))
            return R if isinstance(R, torch.Tensor) else torch.stack([R.interval.inf, R.interval.sup])

        self.assertTrue(torch.autograd.gradcheck(result, (center, delta), atol=1e-5, rtol=1e-4))

    def test_gradients(self):
        center, delta = [[1.0, 0.4], [0.3, 1.2]], [[0.2, 0.1], [0.05, 0.3]]
        self.check(lambda A: A.abs(), center, delta)
        self.check(lambda A: A.center, center, delta)
        self.check(lambda A: A.infimum, center, delta)
        self.check(lambda A: A.infinity_norm(), center, delta)
        self.check(lambda A: A.norm("fro"), center, delta)
        self.check(lambda A: A.exact_square(), center, delta)
        self.check(lambda A: A.matrix_power(3), center, delta)
        self.check(lambda A: A @ A, center, delta)
        self.check(lambda A: A.dependent_terms(0.5)[1], center, delta)
        self.check(lambda A: A.expm(6).enclosure, center, delta)


if __name__ == "__main__":
    unittest.main()
