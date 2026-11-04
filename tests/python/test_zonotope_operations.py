"""test_zonotope_operations - abs, box, compact, cart_prod, enclose, convex_hull, intersect, contains
and more of Zonotope in Python, on numpy and Eigen alone."""
# CORA: contSet/@zonotope/abs.m
# CORA: contSet/@zonotope/and_.m
# CORA: contSet/@zonotope/boundaryPoint.m
# CORA: contSet/@zonotope/box.m
# CORA: contSet/@zonotope/cartProd_.m
# CORA: contSet/@zonotope/center.m
# CORA: contSet/@zonotope/compact_.m
# CORA: contSet/@zonotope/constrSat.m
# CORA: contSet/@zonotope/contains_.m
# CORA: contSet/@zonotope/convHull_.m
# CORA: contSet/@zonotope/cubMap.m
# CORA: contSet/@zonotope/dH2box.m
# CORA: contSet/@zonotope/dominantDirections.m
# CORA: contSet/@zonotope/enclose.m
# PROPERTY: contSet/@zonotope/abs.m
# PROPERTY: contSet/@zonotope/and_.m
# PROPERTY: contSet/@zonotope/boundaryPoint.m
# PROPERTY: contSet/@zonotope/box.m
# PROPERTY: contSet/@zonotope/cartProd_.m
# PROPERTY: contSet/@zonotope/center.m
# PROPERTY: contSet/@zonotope/compact_.m
# PROPERTY: contSet/@zonotope/constrSat.m
# PROPERTY: contSet/@zonotope/contains_.m
# PROPERTY: contSet/@zonotope/convHull_.m
# PROPERTY: contSet/@zonotope/cubMap.m
# PROPERTY: contSet/@zonotope/dH2box.m
# PROPERTY: contSet/@zonotope/dominantDirections.m
# PROPERTY: contSet/@zonotope/enclose.m

import unittest

import cora
import numpy as np


def zonotope(c, G):
    return cora.Zonotope(np.array(c, dtype=float), np.array(G, dtype=float))


def diamond():
    """The set |x| + |y| <= 2."""
    return zonotope([0, 0], [[1, 1], [1, -1]])


def random_zonotope(rng, n=3, m=4):
    return zonotope(10 * rng.standard_normal(n), rng.uniform(-1, 1, (n, m)))


def support(Z, rng, count=40):
    """The support function along `count` random directions: (directions, values)."""
    d = rng.standard_normal((count, Z.dim))
    return d, np.array([Z.support_function(x) for x in d])


def within_tol(a, b, tol):
    """CORA's withinTol for finite numbers."""
    difference = np.abs(a - b)
    return (difference <= tol) | (difference <= tol * np.minimum(np.abs(a), abs(b)))


def matlab_delete_aligned(G, tol):
    """CORA's compact_(Z, 'all', tol), line by line (aux_deleteAligned)."""
    G = G[:, np.any(G, axis=0)]
    G = G[:, np.linalg.norm(G, axis=0) > tol]
    if G.shape[0] == 1:
        return np.abs(G).sum(axis=1, keepdims=True)
    length = np.linalg.norm(G, axis=0)
    order = np.argsort(length, kind="stable")
    unit, G = (G / length)[:, order], G[:, order].copy()
    keep = np.ones(G.shape[1], dtype=bool)
    for i in range(G.shape[1]):
        if keep[i]:
            dots = unit[:, i] @ unit
            parallel, anti = within_tol(dots, 1, tol), within_tol(dots, -1, tol)
            if parallel.sum() > 1 or anti.any():
                G[:, i] = G[:, parallel].sum(axis=1) - G[:, anti].sum(axis=1)
                parallel[i] = False
                keep = keep & ~parallel & ~anti
    return G[:, keep]


def matlab_cub_map(Zs, T, ind):
    """CORA's cubMap for one zonotope (aux_cubMapSingle), or for three (aux_cubMapMixed), with
    MATLAB's loops: the center and generators of the result as one matrix."""
    fulls = [np.hstack([Z.center.reshape(-1, 1), Z.generators]) for Z in Zs]
    rows = []
    for i in range(len(T)):
        if len(Zs) == 3:
            F1, F2, F3 = fulls
            Nq = F1.shape[1] * F2.shape[1]
            row = np.zeros(Nq * F3.shape[1])
            for k in ind[i]:
                vec = (F1.T @ T[i][k] @ F2).reshape(-1, order="F")
                for j in range(F3.shape[1]):
                    row[j * Nq:(j + 1) * Nq] += vec * F3[k, j]
            rows.append(row)
            continue
        F = fulls[0]
        N = F.shape[1]
        L = [np.zeros((N, N)) for _ in range(N)]
        for k in ind[i]:
            Q = F.T @ T[i][k] @ F
            lower = np.tril(Q, -1)
            Q = Q - lower + lower.T
            for j in range(N):
                L[j] = L[j] + Q * F[k, j]
        for k in range(1, N):
            for j in range(k):
                for h in range(j, N):
                    if h <= k:
                        L[j][h, k] += L[k][j, h]
                    else:
                        L[j][k, h] += L[k][j, h]
        L[0][0, 0] += 0.5 * np.diag(L[0])[1:].sum()
        for k in range(1, N):
            L[0][k, k] *= 0.5
        rows.append(np.concatenate([L[k][j, j:] for k in range(N) for j in range(k, N)]))
    return np.array(rows)


class Entrywise(unittest.TestCase):
    def test_abs_and_box_and_center(self):
        Z = zonotope([1, -2], [[1, -3, 0.5], [-2, 0.5, 0]])
        np.testing.assert_array_equal(Z.abs().center, [1, 2])
        np.testing.assert_array_equal(Z.abs().generators, [[1, 3, 0.5], [2, 0.5, 0]])
        np.testing.assert_array_equal(Z.box().center, Z.center)
        np.testing.assert_array_equal(Z.box().generators, [[4.5, 0], [0, 2.5]])
        np.testing.assert_array_equal(Z.center, [1, -2])

    def test_properties(self):
        rng = np.random.default_rng(4)
        Z = random_zonotope(rng)
        corner = np.abs(Z.center) + np.abs(Z.generators).sum(axis=1)
        points = Z.rand_point(100, cora.Rng(4))
        self.assertTrue(np.all(np.abs(points) <= corner[:, None] + 1e-12))
        np.testing.assert_array_equal(Z.abs().abs().generators, Z.abs().generators)
        np.testing.assert_allclose(Z.box().interval().sup, Z.interval().sup)
        d, values = support(Z, rng)
        self.assertTrue(np.all([Z.box().support_function(x) for x in d] >= values - 1e-12))
        mid = [(Z.support_function(x) - Z.support_function(-x)) / 2 for x in d]
        np.testing.assert_allclose(mid, d @ Z.center, atol=1e-9)


class Compact(unittest.TestCase):
    def test_zeros_and_tolerance(self):
        Z = zonotope([1, 2], [[1, 0, 1e-20, 0.5], [0, 0, 0, 0.5]])
        np.testing.assert_array_equal(Z.compact().generators, [[1, 0.5], [0, 0.5]])
        self.assertEqual(Z.compact("zeros", 0.0).generators.shape[1], 3)
        self.assertEqual(Z.compact("zeros", 0.8).generators.shape[1], 1)

    def test_all_merges_aligned_generators(self):
        Z = zonotope([0, 0], [[-0.5, 1, 2, 0], [0, 0, 0, 3]])
        np.testing.assert_allclose(Z.compact("all").generators, [[-3.5, 0], [0, 3]])
        np.testing.assert_allclose(zonotope([0], [[1, -2, 0]]).compact("all").generators, [[3]])
        np.testing.assert_array_equal(Z.compact("aligned").generators, Z.generators)

    def test_the_set_stays(self):
        rng = np.random.default_rng(4)
        Z = random_zonotope(rng, 3, 5)
        g = Z.generators
        R = zonotope(Z.center, np.hstack([g, -2 * g[:, :1], 0 * g[:, :1], 3 * g[:, 1:2]]))
        for method in ("zeros", "all"):
            C = R.compact(method, 1e-10)
            self.assertLess(C.generators.shape[1], R.generators.shape[1])
            d, values = support(R, rng)
            np.testing.assert_allclose([C.support_function(x) for x in d], values, rtol=1e-9)

    def test_all_agrees_with_the_matlab_loop(self):
        rng = np.random.default_rng(4)
        for n in (1, 2, 3):
            g = rng.uniform(-1, 1, (n, 4))
            G = np.hstack([g, -2 * g[:, :1], 3 * g[:, 1:2], 0 * g[:, :1], rng.uniform(-1, 1, (n, 2))])
            mine = zonotope(np.zeros(n), G).compact("all", 1e-10).generators
            np.testing.assert_allclose(mine, matlab_delete_aligned(G, 1e-10), atol=1e-12)

    def test_errors(self):
        Z = zonotope([0, 0], [[1, 0], [0, 0]])
        with self.assertRaises(ValueError):
            Z.compact("sorted")
        with self.assertRaises(ValueError):
            Z.compact("zeros", -1.0)


class Combining(unittest.TestCase):
    def test_cart_prod_of_each_kind(self):
        Z = zonotope([1, 2], [[1, 2], [3, 4]])
        P = Z.cart_prod(zonotope([5], [[6, 7, 8]]))
        np.testing.assert_array_equal(P.center, [1, 2, 5])
        np.testing.assert_array_equal(P.generators, [[1, 2, 0, 0, 0], [3, 4, 0, 0, 0], [0, 0, 6, 7, 8]])
        self.assertEqual(Z.cart_prod(np.array([9.0])).generators.shape, (3, 2))
        self.assertEqual(Z.cart_prod(cora.Interval(np.array([0.0, -1.0]), np.array([2.0, 1.0]))).dim, 4)
        self.assertEqual(cora.cart_prod(Z, Z).dim, 4)

    def test_cart_prod_support_function_adds(self):
        rng = np.random.default_rng(4)
        Z, Z2 = random_zonotope(rng, 2, 4), random_zonotope(rng, 3, 5)
        P = Z.cart_prod(Z2)
        for d in rng.standard_normal((20, 5)):
            split = Z.support_function(d[:2]) + Z2.support_function(d[2:])
            self.assertAlmostEqual(P.support_function(d), split)

    def test_enclose_and_convex_hull(self):
        Z = zonotope([0, 0], [[1, 0], [0, 1]])
        Z2 = zonotope([2, 0], [[1], [0]])
        E = Z.enclose(Z2)
        np.testing.assert_array_equal(E.center, [1, 0])
        np.testing.assert_array_equal(E.generators, [[1, -1, 0, 0], [0, 0, 0, 1]])
        rng = np.random.default_rng(4)
        A, B = random_zonotope(rng, 3, 5), random_zonotope(rng, 3, 3)
        box = cora.Interval(np.array([-1.0, 0.0, 0.5]), np.array([2.0, 1.0, 3.0]))
        d = rng.standard_normal((40, 3))
        for hull, other in ((A.convex_hull(B), B), (A.convex_hull(box), box)):
            bound = np.maximum([A.support_function(x) for x in d], [other.support_function(x) for x in d])
            self.assertTrue(np.all([hull.support_function(x) for x in d] >= bound - 1e-9))
        with_point = A.convex_hull(B.center)
        bound = np.maximum([A.support_function(x) for x in d], d @ B.center)
        self.assertTrue(np.all([with_point.support_function(x) for x in d] >= bound - 1e-9))
        self.assertIsInstance(cora.convex_hull(A, B), cora.Zonotope)

    def test_intersect_boxes_and_averaging(self):
        Z = zonotope([0, 0], [[1, 0], [0, 1]])
        R = Z.intersect(zonotope([1, 0], [[1, 0], [0, 0.5]]))
        np.testing.assert_array_equal(R.center, [0.5, 0])
        np.testing.assert_array_equal(R.generators, [[0.5, 0], [0, 0.5]])
        self.assertEqual(Z.intersect(cora.Interval(np.array([0.0, -2.0]), np.array([0.5, 2.0]))).dim, 2)
        A = zonotope([1, 0], [[1, 1], [1, -1]])
        B = zonotope([0, 1], [[2, 0], [0, 2]])
        M = A.intersect(B, "averaging")
        np.testing.assert_allclose(M.center, [2 / 3, 1 / 3])
        np.testing.assert_allclose(M.generators[0], [2 / 3, 2 / 3, 2 / 3, 0])
        with self.assertRaises(cora.errors.NotSupported):
            A.intersect(A)
        with self.assertRaises(cora.errors.EmptySet):
            Z.intersect(zonotope([5, 5], [[1, 0], [0, 1]]))

    def test_averaging_contains_the_intersection(self):
        rng = np.random.default_rng(4)
        shared = 0
        for _ in range(10):
            A = random_zonotope(rng, 2, 4)
            other = random_zonotope(rng, 2, 4).generators
            B = zonotope(A.center + [0.3, -0.2], 0.8 * A.generators + 0.4 * other)
            R = A.intersect(B, "averaging")
            for x in A.rand_point(40, cora.Rng(int(rng.integers(1000)))).T:
                if B.contains(x):
                    shared += 1
                    self.assertTrue(R.contains(x, tol=1e-9))
        self.assertGreater(shared, 5)


class Constraints(unittest.TestCase):
    def test_constraints_satisfied(self):
        Z = zonotope([0, 0], [[1, 0], [0, 1]])
        C = np.array([[1.0, 1.0]])
        self.assertTrue(Z.constraints_satisfied(C, np.array([3.0])))
        self.assertFalse(Z.constraints_satisfied(C, np.array([2.0])))
        self.assertFalse(Z.constraints_satisfied(C, np.array([1.9])))

    def test_true_is_sound(self):
        rng = np.random.default_rng(4)
        accepted = 0
        for _ in range(30):
            Z = random_zonotope(rng, 3, 4)
            C, d = rng.standard_normal((2, 3)), 8 * rng.standard_normal(2)
            if Z.constraints_satisfied(C, d):
                accepted += 1
                self.assertTrue(np.all(C @ Z.rand_point(100, cora.Rng(accepted)) < d[:, None]))
        self.assertGreater(accepted, 3)


class CubicMap(unittest.TestCase):
    def test_by_hand(self):
        Z = zonotope([1], [[0.5]])
        R = Z.cubic_map([[np.array([[2.0]])]])
        np.testing.assert_allclose(R.center, [2.75])
        np.testing.assert_allclose(R.generators, [[3.0, 0.75, 0.25]])
        M = Z.cubic_map(zonotope([2], [[1]]), zonotope([3], [[0.5]]), [[np.array([[1.0]])]])
        np.testing.assert_allclose(M.center, [6])
        np.testing.assert_allclose(M.generators, [[3, 3, 1.5, 1, 0.5, 0.5, 0.25]])

    def test_the_map_of_the_set_is_enclosed(self):
        rng = np.random.default_rng(4)
        n = 3
        Z = random_zonotope(rng, n, 4)
        T = [[rng.standard_normal((n, n)) for _ in range(n)] for _ in range(2)]
        R = Z.cubic_map(T)
        # m = 4: the 20 cubic, 10 quadratic and 4 linear terms of CORA's cubMap
        self.assertEqual(R.generators.shape, (2, 34))
        T0 = [[T[i][0], 0 * T[i][1], T[i][2]] for i in range(2)]
        np.testing.assert_allclose(Z.cubic_map(T, [[0, 2], [0, 2]]).generators, Z.cubic_map(T0).generators)
        for x in Z.rand_point(60, cora.Rng(4)).T:
            z = np.array([sum(x[k] * (x @ T[i][k] @ x) for k in range(n)) for i in range(2)])
            for d in rng.standard_normal((5, 2)):
                self.assertLessEqual(d @ z, R.support_function(d) + 1e-9)

    def test_agrees_with_the_matlab_loops(self):
        rng = np.random.default_rng(4)
        n, m = 3, 3
        Z, Z2, Z3 = (random_zonotope(rng, n, m + k) for k in range(3))
        T = [[rng.standard_normal((n, n)) for _ in range(n)] for _ in range(2)]
        ind = [[0, 1, 2], [2, 0]]
        for use_ind in (None, ind):
            chosen = use_ind or [[0, 1, 2]] * 2
            single = matlab_cub_map([Z], T, chosen)
            R = Z.cubic_map(T, use_ind)
            np.testing.assert_allclose(R.center, single[:, 0], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(R.generators, single[:, 1:], rtol=1e-12, atol=1e-12)
            mixed = matlab_cub_map([Z, Z2, Z3], T, chosen)
            M = Z.cubic_map(Z2, Z3, T, use_ind)
            np.testing.assert_allclose(M.center, mixed[:, 0], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(M.generators, mixed[:, 1:], rtol=1e-12, atol=1e-12)

    def test_errors(self):
        Z = zonotope([1, 0], [[0.5], [0.2]])
        with self.assertRaises(ValueError):
            Z.cubic_map([])
        with self.assertRaises(ValueError):
            Z.cubic_map([[np.eye(1)]])


class Measures(unittest.TestCase):
    def test_hausdorff_distance_to_box_methods(self):
        Z = zonotope([0, 0], [[1], [1]])
        self.assertAlmostEqual(Z.hausdorff_distance_to_box("naive"), np.sqrt(2))
        self.assertAlmostEqual(Z.hausdorff_distance_to_box("wgreedy"), 2.0)
        self.assertAlmostEqual(Z.hausdorff_distance_to_box("wopt"), np.sqrt(2))
        self.assertAlmostEqual(Z.hausdorff_distance_to_box("exact"), np.sqrt(2), places=3)
        self.assertEqual(Z.hausdorff_distance_to_box(), Z.hausdorff_distance_to_box("exact"))
        with self.assertRaises(ValueError):
            Z.hausdorff_distance_to_box("fast")
        with self.assertRaises(cora.errors.NotSupported):
            Z.hausdorff_distance_to_box("ell")

    def test_bounded_by_the_generator_lengths(self):
        rng = np.random.default_rng(4)
        for n in (2, 3):
            Z = random_zonotope(rng, n, 5)
            total = np.linalg.norm(Z.generators, axis=0).sum()
            for method in ("exact", "naive"):
                self.assertLessEqual(Z.hausdorff_distance_to_box(method), total + 1e-9)

    def test_dominant_directions(self):
        Z = zonotope([0, 0], [[1, 0, 0.1], [0, 2, 0.1]])
        np.testing.assert_array_equal(Z.dominant_directions(), [[0, 1], [2, 0]])
        np.testing.assert_array_equal(Z.dominant_directions(2, 1), [[0, 1], [2, 0]])
        rng = np.random.default_rng(4)
        Z = random_zonotope(rng, 3, 9)
        P = Z.dominant_directions()
        radius = np.abs(np.linalg.inv(P) @ Z.generators).sum(axis=1)
        for x in rng.standard_normal((30, 3)):
            self.assertGreaterEqual(x @ Z.center + np.abs(P.T @ x) @ radius, Z.support_function(x) - 1e-9)
        with self.assertRaises(ValueError):
            Z.dominant_directions(0, 1)


class Containment(unittest.TestCase):
    def test_points_in_a_diamond(self):
        Z = diamond()
        self.assertTrue(Z.contains(np.array([0.0, 0.0])))
        self.assertTrue(np.array([1.0, 1.0]) in Z)
        self.assertTrue(Z.contains(np.array([[0.5, 2.0, -1.0], [0.5, 0.0, 0.5]])))
        self.assertFalse(Z.contains(np.array([[0.5, 1.5], [0.5, 1.5]])))
        self.assertFalse(Z.contains(np.array([1.0 + 1e-10, 1.0])))
        self.assertTrue(Z.contains(np.array([1.0 + 1e-10, 1.0]), tol=1e-6))

    def test_sets_in_a_diamond(self):
        Z = diamond()
        small, large = 0.5 * Z, 1.2 * Z
        for method in ("exact", "exact:venum", "approx", "approx:st"):
            self.assertTrue(Z.contains(small, method))
            self.assertFalse(Z.contains(large, method))
        self.assertTrue(Z.contains(cora.Interval(np.array([-0.5, -0.5]), np.array([0.5, 0.5]))))
        self.assertFalse(Z.contains(cora.Interval(np.array([-1.5, -1.5]), np.array([1.5, 1.5]))))
        self.assertTrue(cora.contains(Z, small))
        with self.assertRaises(cora.errors.NotSupported):
            Z.contains(Z, "exact:polymax")
        with self.assertRaises(cora.errors.NoSpecificAlg):
            Z.contains(np.array([0.0, 0.0]), "simplex")

    def test_scaled_copies(self):
        rng = np.random.default_rng(4)
        for _ in range(4):
            Z = random_zonotope(rng, 3, 5)
            inside = zonotope(Z.center, 0.9 * Z.generators)
            outside = zonotope(Z.center, 1.1 * Z.generators)
            for method in ("exact", "approx"):
                self.assertTrue(Z.contains(inside, method))
                self.assertFalse(Z.contains(outside, method))

    def test_boundary_point(self):
        Z = diamond()
        np.testing.assert_allclose(Z.boundary_point(np.array([1.0, 0.0])), [2, 0], atol=1e-9)
        x = Z.boundary_point(np.array([1.0, 1.0]), np.array([1.0, 0.0]))
        np.testing.assert_allclose(x, [1.5, 0.5], atol=1e-9)
        with self.assertRaises(ValueError):
            Z.boundary_point(np.array([0.0, 0.0]))
        with self.assertRaises(ValueError):
            Z.boundary_point(np.array([1.0, 0.0]), np.array([5.0, 0.0]))

    def test_boundary_point_is_on_the_boundary(self):
        rng = np.random.default_rng(4)
        for _ in range(5):
            Z = random_zonotope(rng, 3, 5)
            d = rng.standard_normal(3)
            x = Z.boundary_point(d)
            self.assertTrue(Z.contains(x, tol=1e-6))
            self.assertFalse(Z.contains(Z.center + (x - Z.center) * 1.001))


if __name__ == "__main__":
    unittest.main()
