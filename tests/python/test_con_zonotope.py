"""test_con_zonotope - the ConZonotope class in Python: constructors, queries and set operations

CORA authors: Dmitry Grebenyuk, Niklas Kochdumper, Lukas Koller, Adrian Kulmburg, Mark Wetzlinger,
              Tobias Ladner, Matthias Althoff, Ivan Brkan
Authors:       Claude Sonnet 5.5
Written:       04-November-2026
Last update:   ---
Reviewed:      ---
"""
# CORA: contSet/@conZonotope/and_.m
# CORA: contSet/@conZonotope/approximateBoundsWithGPU.m
# CORA: contSet/@conZonotope/cartProd_.m
# CORA: contSet/@conZonotope/center.m
# CORA: contSet/@conZonotope/compact_.m
# CORA: contSet/@conZonotope/conIntersect.m
# CORA: contSet/@conZonotope/conZonotope.m
# CORA: contSet/@conZonotope/contains_.m
# CORA: contSet/@conZonotope/convHull_.m
# CORA: contSet/@conZonotope/cubMap.m
# CORA: contSet/@conZonotope/display.m
# CORA: contSet/@conZonotope/empty.m
# CORA: contSet/@conZonotope/enclose.m
# CORA: contSet/@conZonotope/intersectStrip.m
# CORA: contSet/@conZonotope/interval.m
# CORA: contSet/@conZonotope/isBounded.m
# CORA: contSet/@conZonotope/isFullDim.m
# CORA: contSet/@conZonotope/isIntersecting_.m
# CORA: contSet/@conZonotope/isequal.m
# CORA: contSet/@conZonotope/lift_.m
# CORA: contSet/@conZonotope/minkDiff.m
# CORA: contSet/@conZonotope/mtimes.m
# CORA: contSet/@conZonotope/or.m
# CORA: contSet/@conZonotope/origin.m
# CORA: contSet/@conZonotope/plus.m
# CORA: contSet/@conZonotope/project.m
# CORA: contSet/@conZonotope/quadMap.m
# CORA: contSet/@conZonotope/reduce.m
# CORA: contSet/@conZonotope/representsa_.m
# CORA: contSet/@conZonotope/rescale.m
# CORA: contSet/@conZonotope/split.m
# CORA: contSet/@interval/conZonotope.m
# CORA: contSet/@zonotope/conZonotope.m

import unittest

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

def con_zonotope(c, G, A=None, b=None):
    A = None if A is None else np.array(A, dtype=float)
    b = None if b is None else np.array(b, dtype=float)
    return cora.ConZonotope(np.array(c, dtype=float), np.array(G, dtype=float), A, b)


def segment():
    """x = xi on xi1 + xi2 = 1: the segment from (0, 1) to (1, 0)."""
    return con_zonotope([0.0, 0.0], np.eye(2), [[1.0, 1.0]], [1.0])


def is_empty_box(box):
    """Whether a box has no point: bounds without columns."""
    return box.inf.ndim == 2 and box.inf.shape[1] == 0


def random_set(rng, n, m, p):
    """A random set with the constraints [I B] x = b, whose points are drawn by rejection."""
    B = 0.3 * rng.standard_normal((p, m - p))
    xi = rng.uniform(-0.5, 0.5, m)
    A = np.hstack([np.eye(p), B])
    return con_zonotope(rng.standard_normal(n), rng.standard_normal((n, m)), A, A @ xi)


def sample(cZ, rng, count):
    """Points of a set made by random_set, as columns (n, count)."""
    G, A, b, c = cZ.generators, cZ.A, cZ.b, cZ.c
    p, m = A.shape
    points = []
    while len(points) < count:
        xi = rng.uniform(-1, 1, m)
        xi[:p] = b - A[:, p:] @ xi[p:]
        if np.all(np.abs(xi) <= 1):
            points.append(c + G @ xi)
    return np.array(points).T


# -===========================================  MAIN  ===========================================- #


# ============================================  MAIN  ============================================ #

class ConZonotope(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("eigen"))

    def test_the_constructor_and_the_conversions(self):
        Z = cora.ConZonotope(np.array([1.0, 2.0]), np.eye(2))
        self.assertEqual(Z.dim, 2)
        self.assertEqual(Z.class_name, "ConZonotope")
        self.assertEqual(Z.A.shape, (0, 2))
        self.assertEqual(Z.b.shape, (0,))
        np.testing.assert_array_equal(Z.c, [1.0, 2.0])
        S = segment()
        np.testing.assert_array_equal(S.A, [[1.0, 1.0]])
        np.testing.assert_array_equal(S.b, [1.0])
        legacy = cora.ConZonotope.from_matrix(np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                                              np.array([[1.0, 1.0]]), np.array([1.0]))
        np.testing.assert_array_equal(legacy.generators, S.generators)
        self.assertEqual(cora.ConZonotope.from_matrix(np.array([[1.0], [2.0]])).generators.shape, (2, 0))
        from_zonotope = cora.ConZonotope(cora.Zonotope(np.array([1.0, 2.0]), np.eye(2)))
        np.testing.assert_array_equal(from_zonotope.generators, np.eye(2))
        from_box = cora.ConZonotope(cora.Interval(np.array([0.0, 1.0]), np.array([2.0, 5.0])))
        # a box as a constrained zonotope
        np.testing.assert_array_equal(from_box.c, [1.0, 3.0])
        np.testing.assert_array_equal(from_box.generators, [[1.0, 0.0], [0.0, 2.0]])
        self.assertIn("constrained zonotope", repr(S))
        with self.assertRaises(cora.errors.WrongInputInConstructor):
            cora.ConZonotope(np.array([1.0, 2.0]), np.eye(3))
        with self.assertRaises(cora.errors.WrongInputInConstructor):
            con_zonotope([0.0, 0.0], np.eye(2), [[1.0]], [1.0])
        with self.assertRaises(cora.errors.WrongInputInConstructor):
            cora.ConZonotope(np.zeros(2), np.eye(2), np.array([[1.0, 1.0]]))

    def test_empty_and_is_bounded(self):
        E = cora.ConZonotope.empty(3)
        self.assertEqual(E.dim, 3)
        self.assertTrue(is_empty_box(E.interval()))
        self.assertFalse(E.is_full_dim())
        self.assertEqual(E.center.shape[-1], 0)
        self.assertTrue(E.is_bounded())
        self.assertTrue(segment().is_bounded())
        self.assertEqual(cora.ConZonotope.empty().dim, 0)
        with self.assertRaises(cora.errors.WrongValue):
            cora.ConZonotope.empty(-1)

    # center and interval
    def test_center_interval_and_is_full_dim(self):
        S = segment()
        np.testing.assert_allclose(S.center, [0.5, 0.5], atol=1e-9)
        box = S.interval()
        np.testing.assert_allclose(box.inf, [0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(box.sup, [1.0, 1.0], atol=1e-9)
        self.assertFalse(S.is_full_dim())
        res, basis = S.is_full_dim_subspace()
        self.assertFalse(res)
        np.testing.assert_allclose(np.abs(basis), [[np.sqrt(0.5)], [np.sqrt(0.5)]], atol=1e-9)
        square = con_zonotope([0.0, 0.0], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0]], [0.0])
        self.assertTrue(square.is_full_dim())
        np.testing.assert_allclose(square.center, [0.0, 0.0], atol=1e-9)
        # CORA's rank test says true for a set that the box pins to a point
        pinned = con_zonotope([0.0], [[1.0, 2.0]], [[1.0, 1.0]], [2.0])
        self.assertFalse(pinned.is_full_dim())
        np.testing.assert_allclose(pinned.interval().inf, [3.0])
        none = con_zonotope([0.0], [[1.0]], [[1.0]], [3.0])
        self.assertTrue(is_empty_box(none.interval()))
        with self.assertRaises(cora.errors.WrongValue):
            S.is_full_dim(-1.0)

    def test_compact(self):
        cZ = con_zonotope([1.0, 2.0], [[1.0, 0.0, 2.0], [0.0, 0.0, 3.0]],
                          [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]], [0.0, 1.0])
        for method in ("zeros", "all"):
            c = cZ.compact(method)
            np.testing.assert_array_equal(c.generators, [[1.0, 2.0], [0.0, 3.0]])
            np.testing.assert_array_equal(c.A, [[1.0, 1.0]])
        np.testing.assert_array_equal(cZ.compact("aligned").generators, cZ.generators)
        with self.assertRaises(cora.errors.WrongValue):
            cZ.compact("rows")

    def test_cart_prod_and_intersect(self):
        A = con_zonotope([1.0], [[2.0]], [[1.0]], [0.5])
        P = A.cart_prod(con_zonotope([3.0, 4.0], np.eye(2)))
        # Cartesian products
        np.testing.assert_array_equal(P.c, [1.0, 3.0, 4.0])
        np.testing.assert_array_equal(P.A, [[1.0, 0.0, 0.0]])
        self.assertEqual(A.cart_prod(np.array([9.0])).dim, 2)
        self.assertEqual(A.cart_prod(cora.Interval(np.array([0.0]), np.array([1.0]))).dim, 2)
        self.assertIsInstance(cora.cart_prod(A, cora.Zonotope(np.array([1.0]), np.eye(1))), cora.ConZonotope)
        a, b = con_zonotope([1.0], [[1.0]]), con_zonotope([2.0], [[1.0]])
        R = a.intersect(b)
        np.testing.assert_array_equal(R.A, [[1.0, -1.0]])
        np.testing.assert_allclose(R.interval().inf, [1.0], atol=1e-9)
        np.testing.assert_allclose(R.interval().sup, [2.0], atol=1e-9)
        self.assertIsInstance(a & b, cora.ConZonotope)
        self.assertTrue(is_empty_box(a.intersect(cora.Interval(np.array([5.0]), np.array([6.0]))).interval()))
        C = a.con_intersect(b, np.array([[2.0]]))
        np.testing.assert_allclose(C.interval().inf, [0.5], atol=1e-9)
        np.testing.assert_allclose(C.interval().sup, [1.5], atol=1e-9)
        with self.assertRaises(cora.errors.DimensionMismatch):
            a.con_intersect(b, np.array([[1.0, 2.0]]))
        with self.assertRaises(cora.errors.DimensionMismatch):
            a.intersect(con_zonotope([1.0, 2.0], [[1.0], [1.0]]))

    def test_enclose_and_convex_hull(self):
        # enclose with a matrix and a vector
        A = con_zonotope([0.0], [[1.0]])
        R = A.enclose(np.array([[2.0]]), np.array([4.0]))
        np.testing.assert_array_equal(R.c, [2.0])
        np.testing.assert_array_equal(R.generators, [[1.5, -2.0, -0.5]])
        self.assertEqual(A.enclose(A.enclose(np.array([[2.0]]), np.array([4.0]))).dim, 1)
        with self.assertRaises(cora.errors.SpecialError):
            con_zonotope([0.0], [[1.0, 1.0]], [[1.0, 1.0]], [1.0]).enclose(A)
        a, b = con_zonotope([0.5], [[0.5]]), con_zonotope([2.5], [[0.5]])
        for method in ("exact", "exact:Raghuraman", "exact:Kochdumper"):
            H = a.convex_hull(b, method)
            np.testing.assert_allclose(H.interval().inf, [0.0], atol=1e-9)
            np.testing.assert_allclose(H.interval().sup, [3.0], atol=1e-9)
        c = con_zonotope([-4.0], [[0.5]])
        L = a.convex_hull([b, c])
        np.testing.assert_allclose(L.interval().inf, [-4.5], atol=1e-9)
        np.testing.assert_allclose(L.interval().sup, [3.0], atol=1e-9)
        self.assertEqual(a.convex_hull(np.array([5.0])).interval().sup[0], 5.0)
        self.assertIsInstance(cora.convex_hull(a, b), cora.ConZonotope)
        with self.assertRaises(cora.errors.WrongValue):
            a.convex_hull(b, "outer")

    # contains points
    def test_contains(self):
        S = segment()
        self.assertTrue(S.contains(np.array([0.5, 0.5])))
        self.assertTrue(S.contains(np.array([[0.5, 1.0], [0.5, 0.0]])))
        self.assertFalse(S.contains(np.array([0.5, 0.6])))
        self.assertTrue(S.contains(cora.Zonotope(np.array([0.5, 0.5]), np.array([[0.25], [-0.25]]))))
        self.assertFalse(S.contains(cora.Interval(np.array([0.0, 0.0]), np.array([1.0, 1.0]))))
        self.assertTrue(np.array([0.5, 0.5]) in S)
        self.assertTrue(S.contains(np.array([0.5, 0.5 + 1e-9]), tol=1e-6))
        self.assertFalse(cora.ConZonotope.empty(2).contains(np.array([0.0, 0.0])))
        with self.assertRaises(cora.errors.NoSpecificAlg):
            S.contains(np.array([0.5, 0.5]), method="nothing")
        for method in ("approx", "exact:polymax", "sampling"):
            with self.assertRaises(cora.errors.NotSupported):
                S.contains(np.array([0.5, 0.5]), method=method)
        with self.assertRaises(cora.errors.DimensionMismatch):
            S.contains(np.array([[1.0, 2.0, 3.0]]))

    def test_cubic_map(self):
        T = [[np.array([[0.0, 1.0], [0.0, 0.0]]), np.zeros((2, 2))]]
        plain = con_zonotope([1.0, 2.0], np.eye(2))
        # the cubic map
        expected = cora.Zonotope(np.array([1.0, 2.0]), np.eye(2)).cubic_map(T)
        np.testing.assert_array_equal(plain.cubic_map(T).generators, expected.generators)
        constrained = con_zonotope([1.0, 2.0], np.eye(2), [[1.0, 1.0]], [0.5])
        C = constrained.cubic_map(T)
        np.testing.assert_array_equal(C.generators, expected.generators)
        np.testing.assert_array_equal(C.A[:, :2], [[1.0, 1.0]])
        self.assertEqual(float(np.abs(C.A[:, 2:]).sum()), 0.0)
        mixed = constrained.cubic_map(plain, plain, T)
        self.assertEqual(mixed.A.shape[0], 1)
        with self.assertRaises(cora.errors.WrongValue):
            constrained.cubic_map([[np.eye(3)]])

    def test_intersect_strip(self):
        cZ = con_zonotope([0.0, 0.0], np.eye(2))
        C, phi, y = np.array([[1.0, 0.0]]), np.array([0.1]), np.array([0.5])
        R = cZ.intersect_strip(C, phi, y, np.array([[1.0], [0.0]]))
        np.testing.assert_allclose(R.c, [0.5, 0.0], atol=1e-15)
        np.testing.assert_allclose(R.A, [[1.0, 0.0, -0.1]], atol=1e-15)
        np.testing.assert_allclose(R.b, [0.5], atol=1e-15)
        np.testing.assert_allclose(R.interval().inf, [0.4, -1.0], atol=1e-9)
        self.assertEqual(cZ.intersect_strip(C, phi, y).A.shape, (1, 3))
        # strip intersection by method
        self.assertEqual(cZ.intersect_strip(C, phi, y, "alamo-FRad").A.shape, (1, 3))
        observer = {"E": np.eye(2), "F": np.ones((1, 1)), "A": np.eye(2), "C": np.array([[1.0, 0.0]])}
        self.assertEqual(cZ.intersect_strip(C, phi, y, observer).dim, 2)
        self.assertEqual(cZ.intersect_strip(C, phi, y, "bravo").A.shape[0], 0)
        with self.assertRaises(cora.errors.NotSupported):
            cZ.intersect_strip(C, phi, y, "svd")
        with self.assertRaises(cora.errors.WrongValue):
            cZ.intersect_strip(C, phi, y, "nothing")
        with self.assertRaises(cora.errors.DimensionMismatch):
            cZ.intersect_strip(np.array([[1.0]]), phi, y)

    def test_approximate_bounds(self):
        c = np.zeros((1, 2))
        G = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        A = np.array([[[1.0, 1.0]]])
        b = np.array([[-1.5]])
        l, u, bl, bu = cora.ConZonotope.approximate_bounds(c, G, A, b)
        np.testing.assert_allclose(l, [[-1.0, -1.0]], atol=1e-12)
        np.testing.assert_allclose(u, [[-0.5, -0.5]], atol=1e-12)
        np.testing.assert_allclose(bu, [[-0.5, -0.5]], atol=1e-12)
        e = cora.ConZonotope.approximate_bounds(c, G, A, b, method="exact")
        # approximate bounds
        np.testing.assert_allclose(e[0], l, atol=1e-9)
        wide = cora.ConZonotope.approximate_bounds(c, G, A, b, dr=np.full((1, 2), 0.25))
        np.testing.assert_allclose(wide[0], [[-1.25, -1.25]], atol=1e-12)
        with self.assertRaises(cora.errors.NotSupported):
            cora.ConZonotope.approximate_bounds(c, G, A, b, method="dual-iter")
        with self.assertRaises(cora.errors.WrongFieldValue):
            cora.ConZonotope.approximate_bounds(c, G, A, b, method="nothing")

    def test_origin_plus_and_linear_map(self):
        O = cora.ConZonotope.origin(2)
        np.testing.assert_array_equal(O.c, [0.0, 0.0])
        self.assertEqual(O.generators.shape, (2, 0))
        with self.assertRaises(cora.errors.WrongValue):
            cora.ConZonotope.origin(0)
        a = segment()
        s = con_zonotope([10.0, 20.0], [[2.0], [3.0]], [[4.0]], [1.0])
        R = a.plus(s)
        np.testing.assert_array_equal(R.c, [10.0, 20.0])
        np.testing.assert_array_equal(R.A, [[1.0, 1.0, 0.0], [0.0, 0.0, 4.0]])
        np.testing.assert_array_equal((a + s).b, [1.0, 1.0])
        np.testing.assert_array_equal((a + np.array([1.0, 1.0])).c, [1.0, 1.0])
        # sums with points and sets
        np.testing.assert_array_equal((np.array([1.0, 1.0]) + a).c, [1.0, 1.0])
        np.testing.assert_array_equal((a - np.array([1.0, 1.0])).c, [-1.0, -1.0])
        self.assertEqual(a.plus(cora.Zonotope(np.zeros(2), np.eye(2))).generators.shape, (2, 4))
        self.assertEqual((a + cora.Interval(np.zeros(2), np.ones(2))).generators.shape, (2, 4))
        with self.assertRaises(cora.errors.DimensionMismatch):
            a.plus(con_zonotope([0.0], [[1.0]]))
        M = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]])
        L = a.linear_map(M)
        np.testing.assert_array_equal(L.generators, M)
        np.testing.assert_array_equal((M @ a).c, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal((2 * a).generators, 2 * a.generators)
        np.testing.assert_array_equal((-a).generators, -a.generators)
        with self.assertRaises(cora.errors.DimensionMismatch):
            a.linear_map(np.ones((1, 3)))
        rng = np.random.default_rng(51)
        A = random_set(rng, 3, 4, 2)
        B = random_set(rng, 3, 4, 1)
        total = A.plus(B)
        self.assertTrue(total.contains(sample(A, rng, 10) + sample(B, rng, 10), tol=1e-7))
        self.assertEqual(A.linear_map(np.ones((2, 3))).dim, 2)

    # projection, lift and reduce
    def test_project_lift_and_reduce(self):
        a = con_zonotope([1.0, 2.0, 3.0], [[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]], [[1.0, 1.0]], [0.5])
        P = a.project([2, 0])
        np.testing.assert_array_equal(P.c, [3.0, 1.0])
        np.testing.assert_array_equal(P.A, a.A)
        with self.assertRaises(cora.errors.WrongValue):
            a.project([3])
        L = a.lift(3, [1, 2, 0])
        np.testing.assert_array_equal(L.c, [3.0, 1.0, 2.0])
        with self.assertRaises(cora.errors.WrongValue):
            a.lift(3, [1, 1, 0])
        many = con_zonotope([3.0], [[1.0, -2.0, 0.5]], [[1.0, 1.0, 1.0]], [0.5])
        r = many.reduce(1)
        np.testing.assert_array_equal(r.generators, [[3.5, 0.0]])
        np.testing.assert_array_equal(r.A, [[0.0, 3.0]])
        np.testing.assert_array_equal(many.reduce(1, "girard").generators, r.generators)
        with self.assertRaises(cora.errors.NotSupported):
            many.reduce(1, "pca")
        with self.assertRaises(cora.errors.WrongValue):
            many.reduce(0)
        # reduce keeps the dimension
        rng = np.random.default_rng(52)
        A = random_set(rng, 2, 7, 2)
        R = A.reduce(2)
        self.assertEqual(R.generators.shape, (2, 6))
        self.assertTrue(R.contains(sample(A, rng, 10), tol=1e-7))

    def test_quad_map_and_rescale(self):
        unit = con_zonotope([0.0], [[1.0]])
        q = unit.quad_map([np.array([[1.0]])])
        np.testing.assert_array_equal(q.c, [0.5])
        np.testing.assert_array_equal(q.generators, [[0.5, 0.0]])
        with self.assertRaises(cora.errors.DimensionMismatch):
            unit.quad_map([np.eye(2)])
        rng = np.random.default_rng(53)
        A = random_set(rng, 2, 3, 1)
        B = random_set(rng, 2, 3, 1)
        Q = [np.array([[1.0, 0.5], [-1.0, 2.0]]), np.array([[0.0, 1.0], [1.0, 0.0]])]
        x, y = sample(A, rng, 10), sample(B, rng, 10)
        image = np.vstack([np.sum(x * (Qi @ x), axis=0) for Qi in Q])
        self.assertTrue(A.quad_map(Q).contains(image, tol=1e-7))
        cross = np.vstack([np.sum(x * (Qi @ y), axis=0) for Qi in Q])
        # quadratic maps and rescale
        self.assertTrue(A.quad_map(B, Q).contains(cross, tol=1e-7))
        a = con_zonotope([0.0, 0.0], np.eye(2), [[1.0, 1.0]], [1.5])
        for method in ("exact", "iter"):
            r = a.rescale(method)
            np.testing.assert_allclose(r.c, [0.75, 0.75], atol=1e-9)
            np.testing.assert_allclose(r.A, [[0.25, 0.25]], atol=1e-9)
            np.testing.assert_allclose(r.b, [0.0], atol=1e-9)
        with self.assertRaises(cora.errors.SpecialError):
            unit.rescale()
        with self.assertRaises(cora.errors.EmptySet):
            con_zonotope([0.0], [[1.0]], [[1.0]], [3.0]).rescale()
        with self.assertRaises(cora.errors.WrongValue):
            a.rescale("fast")

    def test_representsa_isequal_and_is_intersecting(self):
        box = con_zonotope([0.0, 0.0], [[1.0, 0.0], [0.0, 2.0]])
        S = segment()
        self.assertTrue(box.representsa("zonotope") and box.representsa("interval"))
        self.assertFalse(S.representsa("zonotope"))
        self.assertFalse(S.representsa("emptySet"))
        self.assertTrue(con_zonotope([0.0], [[1.0]], [[1.0]], [3.0]).representsa("emptySet"))
        # representsa
        self.assertTrue(cora.ConZonotope.origin(2).representsa("origin"))
        with self.assertRaises(cora.errors.NotSupported):
            box.representsa("capsule")
        with self.assertRaises(cora.errors.WrongValue):
            box.representsa("cube")
        swapped = con_zonotope([0.0, 0.0], [[0.0, 1.0], [1.0, 0.0]], [[1.0, 1.0]], [1.0])
        self.assertTrue(S.isequal(swapped))
        self.assertTrue(S == swapped)
        self.assertFalse(S.isequal(con_zonotope([0.0, 0.0], np.eye(2), [[1.0, 1.0]], [0.5])))
        square = con_zonotope([0.0, 0.0], np.eye(2))
        pinned = con_zonotope([0.0, 0.0], [[1.0, 0.0, 5.0], [0.0, 1.0, 5.0]], [[0.0, 0.0, 1.0]], [0.0])
        self.assertTrue(square.isequal(pinned))
        self.assertTrue(square.isequal(cora.Interval(-np.ones(2), np.ones(2))))
        self.assertFalse(S.isequal(box))
        with self.assertRaises(cora.errors.WrongValue):
            S.isequal(S, tol=-1.0)
        meets = cora.Interval(np.array([0.4, 0.4]), np.array([2.0, 2.0]))
        misses = cora.Interval(np.array([0.6, 0.6]), np.array([2.0, 2.0]))
        self.assertTrue(S.is_intersecting(meets))
        self.assertFalse(S.is_intersecting(misses))
        # intersection
        self.assertTrue(S.is_intersecting(misses, "approx"))
        self.assertTrue(S.is_intersecting(np.array([0.5, 0.5])))
        self.assertFalse(S.is_intersecting(S.plus(np.array([2.0, 0.0]))))
        self.assertTrue(cora.is_intersecting(S, meets))
        with self.assertRaises(cora.errors.WrongValue):
            S.is_intersecting(meets, "fast")

    def test_split_unite_and_minkowski_difference(self):
        S = segment()
        low, high = S.split(0)
        np.testing.assert_allclose(low.interval().sup, [0.5, 1.0], atol=1e-9)
        np.testing.assert_allclose(high.interval().inf, [0.5, 0.0], atol=1e-9)
        self.assertEqual(len(S.split()), 2)
        with self.assertRaises(cora.errors.WrongValue):
            S.split(2)
        with self.assertRaises(cora.errors.EmptySet):
            cora.ConZonotope.empty(2).split(0)
        left = con_zonotope([-2.0, 0.0], np.eye(2))
        right = con_zonotope([2.0, 0.0], np.eye(2))
        u = left.unite(right)
        self.assertEqual(u.A.shape[0], 0)
        # union
        self.assertTrue(u.contains(np.array([0.0, 0.0])))
        self.assertTrue(left.unite([right, left]).contains(np.array([2.0, 0.0])))
        self.assertTrue(left.unite(np.array([5.0, 5.0])).contains(np.array([5.0, 5.0])))
        self.assertIsInstance(left | right, cora.ConZonotope)
        for method, error in (("tedrake", cora.errors.NotSupported), ("other", cora.errors.WrongValue)):
            with self.assertRaises(error):
                left.unite(right, method)
        with self.assertRaises(cora.errors.NotSupported):
            left.unite(right, order=2)
        big = con_zonotope([0.0, 0.0], 2.0 * np.eye(2))
        small = cora.Interval(-np.ones(2), np.ones(2))
        expected = con_zonotope([0.0, 0.0], np.eye(2))
        for kind in ("exact", "inner:vertices", "inner:Vinod"):
            self.assertTrue(big.minkowski_difference(small, kind).isequal(expected))
        np.testing.assert_array_equal(big.minkowski_difference(np.array([1.0, 1.0])).c, [-1.0, -1.0])
        with self.assertRaises(cora.errors.WrongValue):
            big.minkowski_difference(small, "other")


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
