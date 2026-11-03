"""test_zonotope_more_operations - exact_plus, generator_length, is_bounded, is_full_dim, isequal,
is_intersecting, lift, norm, minnorm, intersect_strip, enclose_points, enclose_many and mink_diff
of Zonotope in Python, on numpy and Eigen alone."""
# CORA: contSet/@zonotope/exactPlus.m
# CORA: contSet/@zonotope/generatorLength.m
# CORA: contSet/@zonotope/isBounded.m
# CORA: contSet/@zonotope/isFullDim.m
# CORA: contSet/@zonotope/isequal.m
# CORA: contSet/@zonotope/isIntersecting_.m
# CORA: contSet/@zonotope/lift_.m
# CORA: contSet/@zonotope/norm_.m
# CORA: contSet/@zonotope/minnorm.m
# CORA: contSet/@zonotope/intersectStrip.m
# CORA: contSet/@zonotope/enclosePoints.m
# CORA: contSet/@zonotope/encloseMany.m
# CORA: contSet/@zonotope/minkDiff.m
# PROPERTY: contSet/@zonotope/exactPlus.m
# PROPERTY: contSet/@zonotope/generatorLength.m
# PROPERTY: contSet/@zonotope/isFullDim.m
# PROPERTY: contSet/@zonotope/isequal.m
# PROPERTY: contSet/@zonotope/isIntersecting_.m
# PROPERTY: contSet/@zonotope/lift_.m
# PROPERTY: contSet/@zonotope/norm_.m
# PROPERTY: contSet/@zonotope/minnorm.m
# PROPERTY: contSet/@zonotope/intersectStrip.m
# PROPERTY: contSet/@zonotope/enclosePoints.m
# PROPERTY: contSet/@zonotope/encloseMany.m
# PROPERTY: contSet/@zonotope/minkDiff.m

import itertools
import unittest
import warnings

import numpy as np

import cora


def zonotope(c, G):
    return cora.Zonotope(np.array(c, dtype=float), np.array(G, dtype=float))


def random_zonotope(rng, n=3, m=4):
    return zonotope(10 * rng.standard_normal(n), rng.uniform(-1, 1, (n, m)))


def support(Z, d):
    return Z.support_function(d)


class ExactPlus(unittest.TestCase):
    def test_by_hand(self):
        Z = zonotope([0, 1], [[1, 2], [0, 1]])
        Z2 = zonotope([1, 1], [[1], [1]])
        S = Z.exact_plus(Z2)
        np.testing.assert_allclose(S.center, [1, 2])
        np.testing.assert_allclose(S.generators, [[2, 2], [1, 1]])
        np.testing.assert_allclose(Z.exact_plus(Z2, 0).generators, [[1, 2, 1], [0, 1, 1]])
        np.testing.assert_allclose(Z.exact_plus(Z2, k=5).generators, S.generators)

    def test_the_result_is_in_the_plus(self):
        rng = np.random.default_rng(3)
        Z, Z2 = random_zonotope(rng, 3, 5), random_zonotope(rng, 3, 4)
        for k in (0, 2, 4):
            S = Z.exact_plus(Z2, k)
            self.assertEqual(S.generators.shape[1], 9 - min(4, k))
            for d in rng.standard_normal((30, 3)):
                self.assertLessEqual(support(S, d), support(Z + Z2, d) + 1e-9)

    def test_errors(self):
        Z = zonotope([0, 0], [[1], [0]])
        with self.assertRaises(cora.errors.DimensionMismatch):
            Z.exact_plus(zonotope([0], [[1]]))
        with self.assertRaises(cora.errors.WrongValue):
            Z.exact_plus(Z, -1)


class Measures(unittest.TestCase):
    def test_generator_length(self):
        rng = np.random.default_rng(4)
        Z = random_zonotope(rng, 4, 6)
        np.testing.assert_allclose(Z.generator_length(), np.linalg.norm(Z.generators, axis=0))
        self.assertEqual(Z.generator_length().shape, (6,))

    def test_is_bounded(self):
        self.assertTrue(random_zonotope(np.random.default_rng(1)).is_bounded())

    def test_is_full_dim(self):
        self.assertTrue(zonotope([0, 0], [[1, 0], [0, 1]]).is_full_dim())
        flat = zonotope([0, 0], [[1, 2], [1, 2]])
        self.assertFalse(flat.is_full_dim())
        thin = zonotope([0, 0], [[1, 0], [0, 1e-7]])
        self.assertFalse(thin.is_full_dim(tol=1e-6))  # CORA's tolerance
        self.assertTrue(thin.is_full_dim(tol=1e-8) and thin.is_full_dim())
        res, basis = flat.is_full_dim_subspace()
        self.assertFalse(res)
        np.testing.assert_allclose(np.abs(basis), np.sqrt(0.5) * np.ones((2, 1)))
        res, basis = thin.is_full_dim_subspace(1e-8)
        self.assertTrue(res)
        np.testing.assert_allclose(basis, np.eye(2))

    def test_the_rank_decides(self):
        rng = np.random.default_rng(6)
        for n, m in ((3, 2), (3, 3), (4, 7)):
            Z = random_zonotope(rng, n, m)
            self.assertEqual(Z.is_full_dim(), np.linalg.matrix_rank(Z.generators, tol=1e-6) == n)

    def test_errors(self):
        with self.assertRaises(cora.errors.WrongValue):
            zonotope([0], [[1]]).is_full_dim(tol=-1)

    def test_norm(self):
        Z = zonotope([1, -1], [[1, 0], [0, 2]])  # the box [0, 2] x [-3, 1]
        self.assertAlmostEqual(Z.norm(), np.sqrt(13))
        self.assertEqual(Z.norm(1), 5)
        self.assertEqual(Z.norm(float("inf")), 3)
        for mode in ("exact", "ub_convex"):
            with self.assertRaises(cora.errors.NotSupported):
                Z.norm(2, mode)
        with self.assertRaises(cora.errors.WrongValue):
            Z.norm(3)

    def test_norm_bounds_the_points(self):
        rng = np.random.default_rng(8)
        Z = random_zonotope(rng, 3, 4)
        points = Z.rand_point(300, cora.Rng(8), "extreme")
        for kind, bound in ((1, Z.norm(1)), (2, Z.norm(2)), (np.inf, Z.norm(np.inf))):
            self.assertLessEqual(np.linalg.norm(points, kind, axis=0).max(), bound + 1e-9)

    def test_minnorm(self):
        value, x = zonotope([0, 0], [[1, 0], [0, 1]]).minnorm()
        self.assertAlmostEqual(value, 1)
        np.testing.assert_allclose(x, [0, -1], atol=1e-14)
        value, x = zonotope([3, 4], [[1], [1]]).minnorm()
        self.assertEqual(value, 0)
        np.testing.assert_allclose(x, [3, 4])

    def test_minnorm_is_the_least_facet_distance(self):
        # CORA's polytope facets: cross products of n - 1 generators
        rng = np.random.default_rng(9)
        for n, m in ((2, 4), (3, 5)):
            Z = random_zonotope(rng, n, m)
            G = Z.generators
            best = np.inf
            for subset in itertools.combinations(range(m), n - 1):
                Q = G[:, subset]
                v = np.array([(-1) ** i * np.linalg.det(np.delete(Q, i, axis=0)) for i in range(n)])
                best = min(best, np.abs(v @ G).sum() / np.linalg.norm(v))
            value, x = Z.minnorm()
            self.assertAlmostEqual(value, best, places=10)
            self.assertAlmostEqual(np.linalg.norm(x - Z.center), best, places=10)
            self.assertTrue(Z.contains(x, tol=1e-9))

    def test_minnorm_batch_is_refused(self):
        Z = zonotope([0, 0], [[1, 0], [0, 1]])
        with self.assertRaises(cora.errors.NotSupported):
            cora.Zonotope.stack([Z, Z]).minnorm()


class Equality(unittest.TestCase):
    def test_isequal(self):
        Z = zonotope([1, 2], [[1, 0], [0, 3]])
        self.assertTrue(Z.isequal(Z))
        self.assertTrue(Z.isequal(zonotope([1, 2], [[0, -1], [-3, 0]])))
        self.assertFalse(Z.isequal(zonotope([1, 2.5], [[1, 0], [0, 3]])))
        self.assertTrue(zonotope([0], [[1, 1]]).isequal(zonotope([0], [[2]])))
        self.assertTrue(Z.isequal(zonotope([1 + 1e-9, 2], Z.generators), tol=1e-6))
        self.assertFalse(Z.isequal(zonotope([1 + 1e-9, 2], Z.generators)))
        self.assertFalse(Z.isequal(zonotope([1], [[1]])))

    def test_isequal_of_a_box_and_a_point(self):
        box = zonotope([1, 1.25], [[1, 0], [0, 0.25]])
        self.assertTrue(box.isequal(cora.Interval(np.array([0, 1.0]), np.array([2, 1.5]))))
        self.assertFalse(box.isequal(cora.Interval(np.array([0, 1.0]), np.array([2, 2.0]))))
        point = zonotope([1, 2], np.zeros((2, 0)))
        self.assertTrue(point.isequal(np.array([1.0, 2.0])))
        self.assertFalse(point.isequal(np.array([1.0, 2.5])))

    def test_a_reordered_copy_is_equal(self):
        rng = np.random.default_rng(10)
        for _ in range(10):
            Z = random_zonotope(rng, 3, 5)
            W = zonotope(Z.center, Z.generators[:, rng.permutation(5)] * rng.choice([-1, 1], 5))
            self.assertTrue(Z.isequal(W) and W.isequal(Z))

    def test_errors(self):
        Z = zonotope([0], [[1]])
        with self.assertRaises(cora.errors.WrongValue):
            Z.isequal(Z, tol=-1)
        with self.assertRaises(cora.errors.NotSupported):
            cora.Zonotope.stack([Z, Z]).isequal(Z)


class Intersecting(unittest.TestCase):
    def test_by_hand(self):
        square = lambda x: zonotope([x, 0], [[1, 0], [0, 1]])
        Z = square(0)
        self.assertTrue(Z.is_intersecting(square(1.5)))
        self.assertTrue(Z.is_intersecting(square(2)))
        self.assertFalse(Z.is_intersecting(square(3)))
        self.assertFalse(Z.is_intersecting(square(2 + 1e-6)))
        self.assertTrue(Z.is_intersecting(square(2 + 1e-6), tol=1e-5))
        diamond = zonotope([0, 0], [[0.5, 0.5], [0.5, -0.5]])
        corner = cora.Interval(np.array([0.6, 0.6]), np.array([1.0, 1.0]))
        self.assertFalse(diamond.is_intersecting(corner))
        self.assertTrue(diamond.is_intersecting(corner, "approx"))
        self.assertTrue(diamond.is_intersecting(np.array([0.5, 0.5])))
        self.assertFalse(diamond.is_intersecting(np.array([[0.0, 0.6], [0.0, 0.6]])))
        self.assertTrue(cora.is_intersecting(Z, square(1.5)))

    def test_agrees_with_a_separating_direction(self):
        rng = np.random.default_rng(11)
        for _ in range(30):
            Z1, Z2 = random_zonotope(rng, 3, 4), random_zonotope(rng, 3, 3) + 15 * rng.uniform(-1, 1, 3)
            res = Z1.is_intersecting(Z2)
            self.assertEqual(res, Z2.is_intersecting(Z1))
            for d in rng.standard_normal((100, 3)):
                if support(Z1, d) + support(Z2, -d) < 0:
                    self.assertFalse(res)

    def test_errors(self):
        Z = zonotope([0, 0], [[1], [0]])
        with self.assertRaises(cora.errors.DimensionMismatch):
            Z.is_intersecting(zonotope([0], [[1]]))
        with self.assertRaises(cora.errors.WrongValue):
            Z.is_intersecting(Z, "other")
        with self.assertRaises(cora.errors.NotSupported):
            cora.Zonotope.stack([Z, Z]).is_intersecting(Z)


class Lift(unittest.TestCase):
    def test_by_hand(self):
        Z = zonotope([1, 2, 3], [[1, 0], [0, 2], [0, 0]])
        L = Z.lift(3, [1, 2, 0])
        np.testing.assert_allclose(L.center, [3, 1, 2])
        np.testing.assert_allclose(L.generators, [[0, 0], [1, 0], [0, 2]])

    def test_is_a_permutation(self):
        rng = np.random.default_rng(12)
        Z = random_zonotope(rng, 4, 5)
        dims = [2, 0, 3, 1]
        L = Z.lift(4, dims)
        for d in rng.standard_normal((20, 4)):
            self.assertAlmostEqual(support(L, d), support(Z, d[dims]))

    def test_errors(self):
        Z = zonotope([1, 2], [[1], [0]])
        with self.assertRaises(cora.errors.WrongValue):
            Z.lift(1, [0, 0])
        with self.assertRaises(cora.errors.WrongValue):
            Z.lift(3, [0, 3])
        with self.assertRaises(cora.errors.NotDefined):
            Z.lift(3, [0, 2])


class IntersectStrip(unittest.TestCase):
    def setUp(self):
        self.Z = zonotope([0, 0], [[1, 0], [0, 1]])
        self.C, self.phi, self.y = np.array([[1.0, 0.0]]), np.array([0.5]), np.array([0.25])

    def test_gains_by_hand(self):
        Z, C, phi, y = self.Z, self.C, self.phi, self.y
        R = Z.intersect_strip(C, phi, y, np.array([[1.0], [0.0]]))
        np.testing.assert_allclose(R.center, [0.25, 0])
        np.testing.assert_allclose(R.generators, [[0, 0, 0.5], [0, 1, 0]])
        N = Z.intersect_strip(C, phi, y)
        np.testing.assert_allclose(N.center, [0.2, 0])
        np.testing.assert_allclose(N.generators, [[0.2, 0, 0.4], [0, 1, 0]])
        F = Z.intersect_strip(C, phi, y, "alamo-FRad")
        np.testing.assert_allclose(F.generators, N.generators)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            B = Z.intersect_strip(C, phi, y, "bravo")
        np.testing.assert_allclose(B.generators, [[0.5, 0], [0, 1]])
        observer = dict(E=np.zeros((2, 1)), F=np.array([[1.0]]), A=np.eye(2), C=C)
        W = Z.intersect_strip(C, phi, y, observer)
        np.testing.assert_allclose(W.center, [0.125, 0])
        np.testing.assert_allclose(W.generators, [[0.5, 0, 0.25], [0, 1, 0]])

    def test_contains_the_intersection(self):
        rng = np.random.default_rng(13)
        Z = random_zonotope(rng, 3, 5)
        C = np.array([[1.0, 0.5, 0.0], [0.0, -1.0, 2.0]])
        phi, y = np.array([0.8, 1.5]), C @ Z.center + np.array([0.3, -0.2])
        points = Z.rand_point(2000, cora.Rng(13))
        inside = points[:, np.all(np.abs(C @ points - y[:, None]) <= phi[:, None], axis=0)]
        self.assertGreater(inside.shape[1], 50)
        for method in ("normGen", np.zeros((3, 2))):
            self.assertTrue(Z.intersect_strip(C, phi, y, method).contains(inside, tol=1e-9))

    def test_errors(self):
        Z, C, phi, y = self.Z, self.C, self.phi, self.y
        for method in ("svd", "radius", "alamo-volume"):
            with self.assertRaises(cora.errors.NotSupported):
                Z.intersect_strip(C, phi, y, method)
        with self.assertRaises(cora.errors.WrongValue):
            Z.intersect_strip(C, phi, y, "other")
        with self.assertRaises(cora.errors.DimensionMismatch):
            Z.intersect_strip(np.ones((1, 3)), phi, y)
        with self.assertRaises(cora.errors.WrongFieldValue):
            Z.intersect_strip(C, phi, y, dict(E=np.zeros((2, 1))))


class Enclosing(unittest.TestCase):
    def test_enclose_points_by_hand(self):
        points = np.array([[0.0, 2.0, 2.0], [0.0, 2.0, 0.0]])
        Z = cora.Zonotope.enclose_points(points, np.eye(2))
        np.testing.assert_allclose(Z.center, [1, 1])
        np.testing.assert_allclose(Z.generators, np.eye(2))

    def test_both_methods_contain_the_cloud(self):
        rng = np.random.default_rng(14)
        for n in (2, 3, 4):
            points = rng.standard_normal((n, n)) @ rng.standard_normal((n, 50)) + 2
            for method in ("maiga", "stursberg"):
                Z = cora.Zonotope.enclose_points(points, method)
                self.assertTrue(Z.contains(points, tol=1e-9), (method, n))
        same = cora.Zonotope.enclose_points(np.array([[1.0, 1, 1], [2.0, 2, 2]]))
        np.testing.assert_allclose(same.center, [1, 2])
        self.assertEqual(np.abs(same.generators).max(), 0)

    def test_enclose_points_errors(self):
        with self.assertRaises(cora.errors.WrongValue):
            cora.Zonotope.enclose_points(np.zeros((2, 3)), "other")
        with self.assertRaises(cora.errors.WrongValue):
            cora.Zonotope.enclose_points(np.zeros((2, 0)))

    def test_enclose_many(self):
        A, B = zonotope([0.5, 0.5], 0.5 * np.eye(2)), zonotope([2.5, 0.5], 0.5 * np.eye(2))
        E, inverse = cora.Zonotope.enclose_many([A, B], np.array([1.0, 0.0]))
        np.testing.assert_allclose(E.center, [1.5, 0.5])
        np.testing.assert_allclose(E.generators, np.diag([1.5, 0.5]))
        np.testing.assert_allclose(inverse, np.eye(2))

    def test_enclose_many_encloses(self):
        rng = np.random.default_rng(15)
        Zs = [random_zonotope(rng, 3, 5) + np.array([k, -2.0 * k, 0.5]) for k in range(4)]
        E, inverse = cora.Zonotope.enclose_many(Zs, np.array([1.0, -2.0, 0.5]))
        for Z in Zs:
            for d in rng.standard_normal((30, 3)):
                self.assertGreaterEqual(support(E, d), support(Z, d) - 1e-9)
        G = inverse @ E.generators
        np.testing.assert_allclose(G, np.diag(np.diag(G)), atol=1e-12)
        with self.assertRaises(cora.errors.WrongValue):
            cora.Zonotope.enclose_many(Zs, np.zeros(3))


class MinkDiff(unittest.TestCase):
    def test_two_dimensions(self):
        square = lambda r, c=(0.0, 0.0): zonotope(c, r * np.eye(2))
        D = square(2, (1, 1)).mink_diff(square(1, (0, 0.5)))
        np.testing.assert_allclose(D.center, [1, 0.5])
        np.testing.assert_allclose(D.generators, np.eye(2))
        segment = zonotope([0, 0], [[0.5], [0.5]])
        for method in ("approx", "inner", "outer", "outer:coarse", "exact"):
            D = square(1).mink_diff(segment, method)
            np.testing.assert_allclose(D.generators, 0.5 * np.eye(2), atol=1e-9)
        np.testing.assert_allclose(square(1).mink_diff(np.array([0.25, 0.0])).center, [-0.25, 0])

    def test_three_dimensions(self):
        cube, small = zonotope(np.zeros(3), 2 * np.eye(3)), zonotope(np.zeros(3), 0.5 * np.eye(3))
        for method in ("approx", "inner", "outer", "outer:coarse", "exact"):
            np.testing.assert_allclose(cube.mink_diff(small, method).generators, 1.5 * np.eye(3), atol=1e-9)
        segment = zonotope(np.zeros(3), [[0.5], [0], [0]])
        np.testing.assert_allclose(cube.mink_diff(segment).generators, np.diag([1.5, 2, 2]), atol=1e-9)
        with self.assertRaises(cora.errors.WrongValue):
            cube.mink_diff(segment, "exact")

    def test_a_box_warns_and_is_enclosed(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            box = cora.Interval(np.array([-0.5, -0.5]), np.array([0.5, 0.5]))
            D = zonotope([0, 0], 2 * np.eye(2)).mink_diff(box)
        self.assertEqual(len(caught), 1)
        np.testing.assert_allclose(D.generators, 1.5 * np.eye(2), atol=1e-9)

    def test_inner_plus_the_subtrahend_is_inside(self):
        rng = np.random.default_rng(16)
        for _ in range(10):
            M = zonotope(np.zeros(2), 2 * rng.uniform(-1, 1, (2, 4)))
            S = zonotope(rng.uniform(-0.1, 0.1, 2), 0.1 * rng.uniform(-1, 1, (2, 2)))
            inner = M.mink_diff(S, "inner")
            for d in rng.standard_normal((40, 2)):
                self.assertLessEqual(support(inner, d) + support(S, d), support(M, d) + 1e-8)

    def test_errors(self):
        cube, small = zonotope(np.zeros(3), np.eye(3)), zonotope(np.zeros(3), 0.5 * np.eye(3))
        for method in ("inner:conZonotope", "inner:RaghuramanKoeln", "outer:scaling"):
            with self.assertRaises(cora.errors.NotSupported):
                cube.mink_diff(small, method)
        with self.assertRaises(cora.errors.WrongValue):
            cube.mink_diff(small, "other")
        with self.assertRaises(cora.errors.EmptySet):
            zonotope([0, 0], np.eye(2)).mink_diff(zonotope([0, 0], [[3], [0]]))


if __name__ == "__main__":
    unittest.main()
