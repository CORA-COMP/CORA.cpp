"""test_polytope - the Polytope class in Python: constructors, queries and set operations."""
# CORA: contSet/@polytope/Inf.m
# CORA: contSet/@polytope/and_.m
# CORA: contSet/@polytope/box.m
# CORA: contSet/@polytope/cartProd_.m
# CORA: contSet/@polytope/center.m
# CORA: contSet/@polytope/compact_.m
# CORA: contSet/@polytope/conZonotope.m
# CORA: contSet/@polytope/constraints.m
# CORA: contSet/@polytope/contains_.m
# CORA: contSet/@polytope/convHull_.m
# CORA: contSet/@polytope/display.m
# CORA: contSet/@polytope/empty.m
# CORA: contSet/@polytope/enclose.m
# CORA: contSet/@polytope/enclosePoints.m
# CORA: contSet/@polytope/hausdorffDist.m
# CORA: contSet/@polytope/interval.m
# CORA: contSet/@polytope/polytope.m
# CORA: contSet/@interval/polytope.m
# PROPERTY: contSet/@polytope/Inf.m
# PROPERTY: contSet/@polytope/and_.m
# PROPERTY: contSet/@polytope/box.m
# PROPERTY: contSet/@polytope/cartProd_.m
# PROPERTY: contSet/@polytope/center.m
# PROPERTY: contSet/@polytope/compact_.m
# PROPERTY: contSet/@polytope/conZonotope.m
# PROPERTY: contSet/@polytope/constraints.m
# PROPERTY: contSet/@polytope/contains_.m
# PROPERTY: contSet/@polytope/convHull_.m
# PROPERTY: contSet/@polytope/empty.m
# PROPERTY: contSet/@polytope/enclose.m
# PROPERTY: contSet/@polytope/enclosePoints.m
# PROPERTY: contSet/@polytope/hausdorffDist.m
# PROPERTY: contSet/@polytope/interval.m
# PROPERTY: contSet/@polytope/polytope.m
# PROPERTY: contSet/@interval/polytope.m

import unittest

import numpy as np

import cora

SQUARE = np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
EYE = np.eye(2)
SQUARE_H = (np.vstack([EYE, -EYE]), np.array([1.0, 1.0, 0.0, 0.0]))


def square_vertices():
    return cora.Polytope(SQUARE)


def square_halfspaces():
    return cora.Polytope(*SQUARE_H)


def triangle():
    return cora.Polytope(np.array([[-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]]), np.array([0.0, 0.0, 2.0]))


def column(*x):
    return np.array(x, dtype=float).reshape(-1, 1)


class Polytope(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("eigen"))

    def test_the_constructors(self):
        H = square_halfspaces()
        self.assertEqual((H.dim, H.class_name), (2, "Polytope"))
        self.assertEqual(H.A.shape, (4, 2))
        self.assertEqual(H.b.shape, (4,))
        self.assertEqual(H.Ae.shape, (0, 2))
        V = square_vertices()
        self.assertEqual(V.V.shape, (2, 4))
        eq = cora.Polytope(*SQUARE_H, np.array([[1.0, 1.0]]), np.array([1.0]))
        np.testing.assert_array_equal(eq.Ae, [[1.0, 1.0]])
        np.testing.assert_array_equal(eq.be, [1.0])
        box = cora.Polytope(cora.Interval(np.array([0.0, 1.0]), np.array([2.0, 3.0])))
        np.testing.assert_array_equal(box.b, [2.0, 3.0, 0.0, -1.0])
        np.testing.assert_array_equal(cora.Polytope(np.array([[3.0, 1.0, 2.0]])).V, [[1.0, 3.0]])
        self.assertIn("polytope", repr(H).lower())
        with self.assertRaises(cora.errors.SpecialError):
            H.V
        with self.assertRaises(cora.errors.SpecialError):
            V.A
        with self.assertRaises(cora.errors.WrongInputInConstructor):
            cora.Polytope(EYE, np.array([1.0]))
        with self.assertRaises(cora.errors.WrongInputInConstructor):
            cora.Polytope(np.array([[0.0, np.nan]]))
        with self.assertRaises(cora.errors.WrongInputInConstructor):
            cora.Polytope(EYE, np.ones(2), np.ones((1, 2)))

    def test_empty_and_inf(self):
        E = cora.Polytope.empty(2)
        self.assertEqual(E.V.shape, (2, 0))
        self.assertFalse(E.contains(np.zeros((2, 1))))
        W = cora.Polytope.inf(2)
        self.assertEqual(W.A.shape, (0, 2))
        self.assertTrue(W.contains(np.array([[1e9], [-1e9]])))
        self.assertEqual(cora.Polytope.empty().dim, 0)
        with self.assertRaises(cora.errors.WrongValue):
            cora.Polytope.empty(-1)

    def test_constraints_interval_box_center(self):
        A, b, Ae, be = square_vertices().constraints()
        self.assertEqual((A.shape, b.shape, Ae.shape, be.shape), ((4, 2), (4,), (0, 2), (0,)))
        self.assertTrue(np.all(A @ SQUARE <= b[:, None] + 1e-9))
        box = square_vertices().interval()
        np.testing.assert_allclose(box.inf, [0.0, 0.0])
        np.testing.assert_allclose(box.sup, [1.0, 1.0])
        T = triangle()
        np.testing.assert_allclose(T.interval().sup, [2.0, 2.0], atol=1e-9)
        np.testing.assert_allclose(T.box().b, [2.0, 2.0, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(square_vertices().center_by("avg"), [0.5, 0.5])
        np.testing.assert_allclose(square_halfspaces().center, [0.5, 0.5], atol=1e-8)
        np.testing.assert_allclose(T.center_by("avg"), [2 / 3, 2 / 3], atol=1e-9)
        with self.assertRaises(cora.errors.WrongValue):
            T.center_by("mean")

    def test_compact(self):
        P = cora.Polytope(np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0],
                                    [1.0, 1.0]]), np.array([1.0, 4.0, 1.0, 0.0, 0.0, 5.0]))
        self.assertEqual(P.compact().A.shape[0], 4)
        self.assertEqual(P.compact("aligned").A.shape[0], 5)
        self.assertEqual(P.compact("zeros").A.shape[0], 6)
        self.assertEqual(P.compact("A").A.shape[0], 4)
        self.assertEqual(P.compact("AtoAe").A.shape[0], 6)
        self.assertEqual(square_vertices().compact("V").V.shape, (2, 4))
        self.assertEqual(P.compact("all", 1e-6).A.shape[0], 4)
        with self.assertRaises(cora.errors.WrongValue):
            P.compact("nope")
        with self.assertRaises(cora.errors.NotSupported):
            P.compact("V")

    def test_con_zonotope(self):
        T = triangle()
        for method in ("exact:supportFunc", "exact:vertices"):
            cz = T.con_zonotope(method)
            self.assertTrue(cz.contains(column(0.5, 0.5)))
            self.assertFalse(cz.contains(column(1.5, 1.5)))
        wide = cora.Interval(np.array([-1.0, -1.0]), np.array([3.0, 3.0]))
        self.assertTrue(T.con_zonotope("exact:supportFunc", wide).contains(column(1.0, 0.5)))
        with self.assertRaises(cora.errors.WrongValue):
            T.con_zonotope("nope")
        with self.assertRaises(cora.errors.SpecialError):
            cora.Polytope.inf(2).con_zonotope()

    def test_contains(self):
        H, V = square_halfspaces(), square_vertices()
        for P in (H, V):
            self.assertTrue(P.contains(column(0.5, 0.5)))
            self.assertFalse(P.contains(column(1.5, 0.5)))
            self.assertIn(np.array([[0.5], [0.5]]), P)
        points = np.array([[0.5, 2.0], [0.5, 0.5]])
        self.assertFalse(H.contains(points))
        for method in ("exact:venum", "exact:polymax", "approx"):
            self.assertTrue(H.contains(column(0.5, 0.5), method))
        self.assertTrue(H.contains(cora.Interval(np.array([0.2, 0.2]), np.array([0.8, 0.8]))))
        self.assertFalse(H.contains(cora.Interval(np.array([0.5, 0.5]), np.array([1.5, 1.0]))))
        self.assertTrue(H.contains(cora.Zonotope(np.array([0.5, 0.5]), 0.2 * EYE)))
        self.assertTrue(H.contains(cora.Polytope(np.array([[0.2, 0.8, 0.8], [0.2, 0.2, 0.8]]))))
        self.assertTrue(H.contains(column(1.0 + 1e-6, 0.5), "exact", 1e-5))
        with self.assertRaises(cora.errors.NoSpecificAlg):
            H.contains(column(0.5, 0.5), "nope")
        with self.assertRaises(cora.errors.DimensionMismatch):
            H.contains(column(0.5))

    def test_set_operations(self):
        H = square_halfspaces()
        Q = cora.Polytope(np.vstack([-EYE, EYE]), np.array([-0.5, -0.5, 2.0, 2.0]))
        R = H.intersect(Q)
        self.assertTrue(R.contains(column(0.75, 0.75)))
        self.assertFalse(R.contains(column(0.25, 0.75)))
        clip = cora.Interval(np.array([0.5, 0.5]), np.array([3.0, 3.0]))
        self.assertTrue(H.intersect(clip).contains(column(0.75, 0.75)))
        np.testing.assert_array_equal(H.intersect(column(0.5, 0.5)).V, [[0.5], [0.5]])
        self.assertEqual(H.intersect(column(3.0, 3.0)).V.shape, (2, 0))
        self.assertEqual(R.dim, (H & Q).dim)
        product = H.cart_prod(Q)
        self.assertEqual(product.dim, 4)
        self.assertTrue(product.contains(column(0.5, 0.5, 1.0, 1.0)))
        self.assertEqual(H.cart_prod(np.array([3.0])).dim, 3)
        self.assertEqual(H.cart_prod(cora.Interval(np.array([0.0]), np.array([1.0]))).dim, 3)
        V1 = square_vertices()
        V2 = cora.Polytope(SQUARE + np.array([[2.0], [0.0]]))
        hull = V1.convex_hull(V2)
        self.assertEqual(hull.V.shape, (2, 4))
        self.assertTrue(hull.contains(column(1.5, 0.5)))
        self.assertEqual(V1.enclose(V2).V.shape, (2, 4))
        self.assertEqual(V1.convex_hull(np.array([2.0, 0.5])).V.shape, (2, 5))
        self.assertEqual(V1.convex_hull(cora.Polytope.empty(2)).V.shape, (2, 4))
        with self.assertRaises(cora.errors.DimensionMismatch):
            H.intersect(cora.Polytope.inf(3))
        with self.assertRaises(cora.errors.DimensionMismatch):
            V1.convex_hull(column(1.0, 2.0, 3.0))

    def test_enclose_points_and_hausdorff(self):
        cloud = np.hstack([SQUARE, [[0.5], [0.5]]])
        P = cora.Polytope.enclose_points(cloud)
        np.testing.assert_array_equal(P.V, SQUARE)
        self.assertEqual(cora.Polytope.enclose_points(np.zeros((3, 4))).V.shape, (3, 1))
        shifted = cora.Polytope(SQUARE + np.array([[2.0], [0.0]]))
        self.assertAlmostEqual(square_vertices().hausdorff_distance(shifted), 2.0, places=8)
        self.assertAlmostEqual(square_halfspaces().hausdorff_distance(square_vertices()), 0.0, places=8)
        big = cora.Interval(np.array([0.0, 0.0]), np.array([2.0, 2.0]))
        self.assertAlmostEqual(square_vertices().hausdorff_distance(big), np.sqrt(2.0), places=8)
        self.assertAlmostEqual(square_vertices().hausdorff_distance(np.array([[3.0], [0.0]])), 2.0, places=8)
        with self.assertRaises(cora.errors.WrongValue):
            cora.Polytope.enclose_points(np.zeros((2, 2, 3)))

    def test_the_generic_interface(self):
        S = cora.intersect(square_halfspaces(), square_halfspaces())
        self.assertIsInstance(S, cora.Polytope)
        self.assertIsInstance(cora.convex_hull(square_vertices(), square_vertices()), cora.Polytope)
        self.assertIsInstance(cora.cart_prod(square_halfspaces(), square_halfspaces()), cora.Polytope)
        self.assertTrue(cora.contains(square_halfspaces(), square_vertices()))
        with cora.checks(False):
            self.assertEqual(square_vertices().interval().sup.shape, (2,))

    def test_random_polytopes(self):
        rng = np.random.default_rng(4)
        for n in (2, 3):
            points = rng.standard_normal((n, n + 4))
            points /= np.linalg.norm(points, axis=0)
            V = cora.Polytope(points)
            A, b, _, _ = V.constraints()
            H = cora.Polytope(A, b)
            self.assertTrue(np.all(A @ points <= b[:, None] + 1e-9))
            np.testing.assert_allclose(H.interval().inf, V.interval().inf, atol=1e-7)
            np.testing.assert_allclose(H.interval().sup, V.interval().sup, atol=1e-7)
            for probe in rng.uniform(-1.2, 1.2, (30, n)):
                self.assertEqual(V.contains(probe.reshape(-1, 1)), H.contains(probe.reshape(-1, 1)))
            mean = points.mean(axis=1)
            np.testing.assert_allclose(V.center_by("avg"), mean, atol=1e-12)
            shift = rng.uniform(-1.0, 1.0, (n, 1))
            moved = cora.Polytope(points + shift)
            self.assertAlmostEqual(V.hausdorff_distance(moved), np.linalg.norm(shift), places=6)
            cz = H.con_zonotope()
            for probe in rng.uniform(-1.2, 1.2, (30, n)):
                self.assertEqual(cz.contains(probe.reshape(-1, 1)), H.contains(probe.reshape(-1, 1)))
            inner = points @ np.full((n + 4, 3), 1.0 / (n + 4))
            enclosed = cora.Polytope.enclose_points(np.hstack([points, inner]))
            np.testing.assert_array_equal(enclosed.V, points)

    def test_batches(self):
        A = np.broadcast_to(np.vstack([EYE, -EYE]), (3, 4, 2)).copy()
        b = np.broadcast_to(np.array([1.0, 1.0, 0.0, 0.0]), (3, 4)).copy()
        P = cora.Polytope(A, b)
        self.assertEqual(P.A.shape, (3, 4, 2))
        self.assertEqual(P.cart_prod(P).A.shape, (3, 8, 4))
        self.assertEqual(P.intersect(P).A.shape, (3, 8, 2))
        V = cora.Polytope(np.broadcast_to(SQUARE, (3, 2, 4)).copy())
        self.assertEqual(V.interval().inf.shape, (3, 2))


if __name__ == "__main__":
    unittest.main()
