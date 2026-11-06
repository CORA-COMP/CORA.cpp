"""test_ellipsoid - the Ellipsoid class in Python: constructors, queries and set operations.

CORA authors: Adrian Kulmburg, Mark Wetzlinger, Matthias Althoff, Niklas Kochdumper,
              Victor Gassmann
Authors:       Claude Sonnet 5.5
Written:       04-November-2026
Last update:   ---
Reviewed:      ---
"""
# CORA: contSet/@ellipsoid/and_.m
# CORA: contSet/@ellipsoid/cartProd_.m
# CORA: contSet/@ellipsoid/center.m
# CORA: contSet/@ellipsoid/contains_.m
# CORA: contSet/@ellipsoid/display.m
# CORA: contSet/@ellipsoid/ellipsoid.m
# CORA: contSet/@ellipsoid/ellipsoidNorm.m
# CORA: contSet/@ellipsoid/empty.m
# CORA: contSet/@ellipsoid/enclose.m
# CORA: contSet/@ellipsoid/enclosePoints.m
# CORA: contSet/@ellipsoid/generators.m
# CORA: contSet/@ellipsoid/intersectStrip.m
# CORA: contSet/@ellipsoid/interval.m
# CORA: contSet/@ellipsoid/isBadDir.m
# CORA: contSet/@ellipsoid/isBigger.m
# CORA: contSet/@ellipsoid/isBounded.m
# CORA: contSet/@ellipsoid/isFullDim.m
# PROPERTY: contSet/@ellipsoid/and_.m
# PROPERTY: contSet/@ellipsoid/cartProd_.m
# PROPERTY: contSet/@ellipsoid/center.m
# PROPERTY: contSet/@ellipsoid/contains_.m
# PROPERTY: contSet/@ellipsoid/ellipsoidNorm.m
# PROPERTY: contSet/@ellipsoid/enclosePoints.m
# PROPERTY: contSet/@ellipsoid/generators.m
# PROPERTY: contSet/@ellipsoid/intersectStrip.m
# PROPERTY: contSet/@ellipsoid/interval.m
# PROPERTY: contSet/@ellipsoid/isBadDir.m
# PROPERTY: contSet/@ellipsoid/isBigger.m
# PROPERTY: contSet/@ellipsoid/isFullDim.m

import unittest

import cora
import numpy as np

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# ----------------------------------------  BEGIN CODE  ---------------------------------------- #

# ----------------------------------------  AUXILIARY  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

def ellipsoid(Q, q=None, tol=None):
    return cora.Ellipsoid(np.array(Q, dtype=float), None if q is None else np.array(q, dtype=float), tol)


def random_ellipsoid(rng, n, rank=None):
    """Q = A A' (+ I/2 if full rank) and a normal center."""
    A = rng.standard_normal((n, n if rank is None else rank))
    Q = A @ A.T + (0.5 * np.eye(n) if rank is None else 0.0)
    return cora.Ellipsoid(0.5 * (Q + Q.T), rng.standard_normal(n))


def boundary_points(E, rng, count):
    """Points q + G u of the boundary, G without the generators of rounding size."""
    G = E.generators()
    G = G[:, np.linalg.norm(G, axis=0) > 1e-6]
    U = rng.standard_normal((G.shape[1], count))
    return E.q[:, None] + G @ (U / np.linalg.norm(U, axis=0))


# ===========================================  MAIN  =========================================== #


# ============================================  MAIN  ============================================ #

class Ellipsoid(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("eigen"))

    def test_the_constructor(self):
        E = ellipsoid([[4.0, 0.0], [0.0, 1.0]], [1.0, 2.0])
        self.assertEqual(E.dim, 2)
        self.assertEqual(E.class_name, "Ellipsoid")
        np.testing.assert_array_equal(E.Q, [[4.0, 0.0], [0.0, 1.0]])
        np.testing.assert_array_equal(E.q, [1.0, 2.0])
        np.testing.assert_array_equal(ellipsoid([[1.0]]).q, [0.0])
        self.assertEqual(ellipsoid([[1.0]], tol=1e-6).tol, 1e-6)
        self.assertEqual(ellipsoid([[1.0]]).tol, cora.default_tol())
        self.assertIn("ellipsoid", repr(E))
        for Q, q in [([[1.0, 2.0, 3.0], [0.0, 1.0, 1.0]], None), ([[1.0, 0.5], [0.0, 1.0]], None),
                     ([[1.0, 0.0], [0.0, -1.0]], None), ([[1.0, 0.0], [0.0, 1.0]], [0.0])]:
            with self.assertRaises(cora.errors.WrongInputInConstructor):
                ellipsoid(Q, q)
        with self.assertRaises(cora.errors.WrongValue):
            ellipsoid([[1.0]], tol=-1.0)
        with cora.checks(False):
            np.testing.assert_array_equal(ellipsoid([[1.0, 0.5], [0.0, 1.0]]).Q, [[1.0, 0.5], [0.0, 1.0]])

    def test_empty_bounded_and_full_dim(self):
        E = cora.Ellipsoid.empty(3)
        self.assertEqual(E.dim, 3)
        self.assertEqual(cora.Ellipsoid.empty().dim, 0)
        self.assertEqual(E.Q.shape, (0, 0))
        self.assertEqual(E.q.shape[0], 3)
        self.assertTrue(E.is_bounded())
        self.assertFalse(E.is_full_dim())
        with self.assertRaises(cora.errors.WrongValue):
            cora.Ellipsoid.empty(-1)
        self.assertTrue(ellipsoid([[4.0, 0.0], [0.0, 1.0]]).is_full_dim())
        self.assertFalse(ellipsoid([[1.0, 0.0], [0.0, 0.0]]).is_full_dim())
        thin = [[1.0, 0.0], [0.0, 1e-8]]
        self.assertTrue(ellipsoid(thin).is_full_dim())
        self.assertFalse(ellipsoid(thin, tol=1e-6).is_full_dim())
        self.assertFalse(ellipsoid(thin).is_full_dim(tol=1e-6))
        with self.assertRaises(cora.errors.WrongValue):
            ellipsoid(thin).is_full_dim(tol=-1.0)

    def test_center_interval_and_generators(self):
        E = ellipsoid([[4.0, 1.0], [1.0, 1.0]], [1.0, 2.0])
        np.testing.assert_array_equal(E.center, [1.0, 2.0])
        box = E.interval()
        np.testing.assert_array_equal(box.inf, [-1.0, 1.0])
        np.testing.assert_array_equal(box.sup, [3.0, 3.0])
        G = E.generators()
        np.testing.assert_allclose(G @ G.T, E.Q, atol=1e-12)
        self.assertEqual(cora.Ellipsoid.empty(2).generators().shape, (2, 0))
        rng = np.random.default_rng(1)
        for k in range(5):
            F = random_ellipsoid(rng, 3, 2 if k == 4 else None)
            B = boundary_points(F, rng, 20)
            box = F.interval()
            self.assertTrue(np.all(B >= box.inf[:, None] - 1e-12) and np.all(B <= box.sup[:, None] + 1e-12))
            np.testing.assert_allclose(F.generators() @ F.generators().T, F.Q, atol=1e-10)

    def test_ellipsoid_norm(self):
        E = ellipsoid([[4.0, 0.0], [0.0, 1.0]])
        np.testing.assert_allclose(E.ellipsoid_norm([2.0, 0.0]), 1.0, atol=1e-12)
        np.testing.assert_allclose(E.ellipsoid_norm([2.0, 3.0]), np.sqrt(10.0), atol=1e-12)
        segment = ellipsoid([[1.0, 0.0], [0.0, 0.0]])
        np.testing.assert_allclose(segment.ellipsoid_norm([2.0, 1e-12]), 2.0, atol=1e-12)
        self.assertEqual(segment.ellipsoid_norm([0.0, 1.0]), np.inf)
        with self.assertRaises(cora.errors.WrongValue):
            E.ellipsoid_norm([1.0])
        with self.assertRaises(cora.errors.EmptySet):
            cora.Ellipsoid.empty(2).ellipsoid_norm([0.0, 0.0])
        rng = np.random.default_rng(2)
        for k in range(4):
            F = random_ellipsoid(rng, 3, 2 if k == 3 else None)
            for p in boundary_points(F, rng, 5).T:
                np.testing.assert_allclose(F.ellipsoid_norm(p - F.q), 1.0, atol=1e-8)

    def test_is_bigger_and_is_bad_dir(self):
        Q = np.diag([4.0, 4.0])
        E = ellipsoid(Q)
        self.assertTrue(E.is_bigger(ellipsoid(Q * 0.25)))
        self.assertFalse(ellipsoid(Q * 0.25).is_bigger(E))
        self.assertFalse(E.is_bigger(ellipsoid(np.diag([5.0, 1.0]))))
        self.assertFalse(ellipsoid([[1.0]]).is_bigger(ellipsoid([[4.0]])))  # CORA says true
        segment = ellipsoid([[1.0, 0.0], [0.0, 0.0]])
        self.assertTrue(segment.is_bigger(ellipsoid([[0.25, 0.0], [0.0, 0.0]])))
        self.assertFalse(segment.is_bigger(E))
        with self.assertRaises(cora.errors.DimensionMismatch):
            E.is_bigger(ellipsoid([[1.0]]))
        with self.assertRaises(cora.errors.EmptySet):
            E.is_bigger(cora.Ellipsoid.empty(2))
        E1, E2 = ellipsoid(np.eye(2)), ellipsoid(np.diag([0.25, 1.0]))
        self.assertEqual(list(E1.is_bad_dir(E2, np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]))),
                         [True, False, True])
        rng = np.random.default_rng(3)
        F = random_ellipsoid(rng, 3)
        L = rng.standard_normal((3, 6))
        self.assertTrue(all(F.is_bad_dir(ellipsoid(4.0 * F.Q), L)))
        self.assertFalse(any(F.is_bad_dir(ellipsoid(0.25 * F.Q), L)))
        with self.assertRaises(cora.errors.WrongValue):
            E1.is_bad_dir(E2, np.ones((3, 1)))

    def test_enclose_points(self):
        segment = cora.Ellipsoid.enclose_points(np.array([[1.0, 3.0], [0.0, 0.0]]))
        np.testing.assert_allclose(segment.q, [2.0, 0.0], atol=1e-12)
        self.assertFalse(segment.is_full_dim())
        point = cora.Ellipsoid.enclose_points(np.array([[2.0, 2.0], [3.0, 3.0]]))
        np.testing.assert_array_equal(point.q, [2.0, 3.0])
        np.testing.assert_array_equal(point.Q, np.zeros((2, 2)))
        line = cora.Ellipsoid.enclose_points(np.array([[1.0, 2.0, 7.0]]))
        np.testing.assert_allclose([line.q[0], line.Q[0, 0]], [4.0, 9.0], atol=1e-12)
        rng = np.random.default_rng(4)
        for count in (2, 3, 5, 40):
            P = rng.standard_normal((4, count))
            E = cora.Ellipsoid.enclose_points(P)
            self.assertTrue(all(E.ellipsoid_norm(p - E.q) <= 1 + 1e-9 for p in P.T))
        with self.assertRaises(cora.errors.WrongValue):
            cora.Ellipsoid.enclose_points(np.zeros((2, 0)))
        with self.assertRaises(cora.errors.WrongValue):
            cora.Ellipsoid.enclose_points(np.ones((2, 3)), method="mean")
        with self.assertRaises(cora.errors.NotSupported):
            cora.Ellipsoid.enclose_points(rng.standard_normal((2, 5)), method="min-vol")

    def test_intersect_strip(self):
        E = ellipsoid([[1.0]])
        C, y = np.array([[1.0]]), np.array([2.0])
        R, sigma_sq = E.intersect_strip(C, 3.0, y)
        self.assertAlmostEqual(sigma_sq, (1 + 2e-4) ** 2, places=12)
        self.assertTrue(all(R.ellipsoid_norm(np.array([p]) - R.q) <= 1 + 1e-12 for p in (-1.0, 1.0)))
        same, level = E.intersect_strip(C, 3.0, np.array([0.0]), 0.25)
        np.testing.assert_array_equal(same.Q, E.Q)
        self.assertEqual(level, 0.25)
        sys = {"V": np.array([[1.0]]), "bar_V": np.array([[1.0]])}
        L, sigma_liu = E.intersect_strip(C, sys, np.array([0.5]), 1.0, method="Liu2016")
        self.assertAlmostEqual(sigma_liu, 15.0 / 16.0, places=12)
        np.testing.assert_allclose([L.Q[0, 0], L.q[0]], [15.0 / 16.0, 0.25], atol=1e-12)
        with self.assertRaises(cora.errors.WrongValue):
            E.intersect_strip(C, 3.0, y, method="Liu2016")
        with self.assertRaises(cora.errors.WrongValue):
            E.intersect_strip(C, sys, y, method="Gollamudi1996")
        with self.assertRaises(cora.errors.WrongFieldValue):
            E.intersect_strip(C, {"V": np.array([[1.0]])}, y)
        with self.assertRaises(cora.errors.DegenerateSet):
            ellipsoid([[0.0]]).intersect_strip(C, 3.0, y)
        # the points of E in the strip are in the result
        rng = np.random.default_rng(5)
        F = random_ellipsoid(rng, 3)
        C3 = rng.standard_normal((2, 3))
        y3 = C3 @ F.q + 0.45 * rng.standard_normal(2)
        P = F.q[:, None] + F.generators() @ (rng.standard_normal((3, 2000)) * 0.3)
        P = P[:, [F.ellipsoid_norm(p - F.q) <= 1 for p in P.T]]
        P = P[:, np.linalg.norm(C3 @ P - y3[:, None], axis=0) <= 0.3]
        R3, _ = F.intersect_strip(C3, 0.3, y3)
        self.assertTrue(all(R3.ellipsoid_norm(p - R3.q) <= 1 + 1e-9 for p in P.T))

    def test_intersect_cart_prod_contains_and_enclose(self):
        E = ellipsoid([[4.0, 0.0], [0.0, 1.0]], [1.0, 2.0])
        inside = E.intersect(np.array([2.0, 2.5]))
        np.testing.assert_array_equal(inside.q, [2.0, 2.5])
        np.testing.assert_array_equal(inside.Q, np.zeros((2, 2)))
        self.assertEqual(E.intersect(np.array([5.0, 2.0])).Q.shape, (0, 0))
        point = ellipsoid(np.zeros((2, 2)), [1.0, 2.5])
        np.testing.assert_array_equal(E.intersect(point).q, [1.0, 2.5])
        self.assertEqual((E & point).Q.shape, (2, 2))
        with self.assertRaises(cora.errors.NotSupported):
            E.intersect(ellipsoid(np.eye(2)))
        with self.assertRaises(cora.errors.WrongValue):
            E.intersect(E, mode="exact")
        one = ellipsoid([[4.0]], [1.0])
        product = one.cart_prod(np.array([5.0, 6.0]))
        np.testing.assert_array_equal(product.q, [1.0, 5.0, 6.0])
        np.testing.assert_array_equal(product.Q, np.diag([4.0, 0.0, 0.0]))
        both = one.cart_prod(ellipsoid([[1.0]]))
        np.testing.assert_allclose(both.Q, np.diag([8.0, 2.0]), atol=1e-12)
        np.testing.assert_allclose(both.q, [1.0, 0.0], atol=1e-12)
        for mode in ("inner", "exact"):
            with self.assertRaises(cora.errors.NotSupported):
                one.cart_prod(one, mode=mode)
        self.assertEqual(cora.cart_prod(one, one).dim, 2)
        self.assertTrue(E.contains(np.array([3.0, 2.0])))
        self.assertTrue(np.array([1.0, 2.0]) in E)
        self.assertFalse(E.contains(np.array([3.0, 3.0])))
        inner = cora.Zonotope(np.array([1.0, 2.0]), np.array([[1.0, 0.0], [0.0, 0.5]]))
        outer = cora.Zonotope(np.array([1.0, 2.0]), np.array([[2.5, 0.0], [0.0, 0.0]]))
        self.assertTrue(E.contains(inner))
        self.assertFalse(E.contains(outer))
        self.assertTrue(cora.contains(E, inner))
        self.assertTrue(E.contains(cora.Interval(np.array([0.0, 1.5]), np.array([2.0, 2.5]))))
        self.assertTrue(point.contains(ellipsoid(np.zeros((2, 2)), [1.0, 2.5])))
        with self.assertRaises(cora.errors.NotSupported):
            E.contains(inner, method="approx")
        with self.assertRaises(cora.errors.NotSupported):
            E.contains(ellipsoid(0.25 * np.eye(2)))
        with self.assertRaises(cora.errors.NotSupported):
            E.enclose(point)
        with self.assertRaises(cora.errors.NotSupported):
            E.enclose(np.eye(2), point)
        with self.assertRaises(cora.errors.EmptySet):
            E.enclose(cora.Ellipsoid.empty(2))


if __name__ == "__main__":
    unittest.main()

# ---------------------------------------  END OF CODE  ---------------------------------------- #

# ----------------------------------------  END OF CODE  ----------------------------------------- #
