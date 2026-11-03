"""test_zonotope_maps_and_splits - origin, radius, rank, representsa, project_high_dim, rotate,
quad_map, split, split_first_gen, split_longest_gen, reduce_under_approx and unite of Zonotope, and
quad_map of Interval, in Python, on numpy and Eigen alone."""
# CORA: contSet/@zonotope/origin.m
# CORA: contSet/@zonotope/radius.m
# CORA: contSet/@zonotope/rank.m
# CORA: contSet/@zonotope/representsa_.m
# CORA: contSet/@contSet/representsa.m
# CORA: contSet/@zonotope/projectHighDim_.m
# CORA: contSet/@contSet/projectHighDim.m
# CORA: contSet/@zonotope/rotate.m
# CORA: contSet/@zonotope/quadMap.m
# CORA: contSet/@zonotope/split.m
# CORA: contSet/@zonotope/splitFirstGen.m
# CORA: contSet/@zonotope/splitLongestGen.m
# CORA: contSet/@zonotope/reduceUnderApprox.m
# CORA: contSet/@zonotope/or.m
# CORA: contSet/@interval/quadMap.m
# PROPERTY: contSet/@zonotope/radius.m
# PROPERTY: contSet/@zonotope/rank.m
# PROPERTY: contSet/@zonotope/rotate.m
# PROPERTY: contSet/@zonotope/quadMap.m
# PROPERTY: contSet/@zonotope/split.m
# PROPERTY: contSet/@zonotope/splitFirstGen.m
# PROPERTY: contSet/@zonotope/splitLongestGen.m
# PROPERTY: contSet/@zonotope/reduceUnderApprox.m
# PROPERTY: contSet/@zonotope/or.m
# PROPERTY: contSet/@interval/quadMap.m

import unittest

import numpy as np

import cora


def zonotope(c, G):
    return cora.Zonotope(np.array(c, dtype=float), np.array(G, dtype=float))


def random_zonotope(rng, n=3, m=4):
    return zonotope(10 * rng.standard_normal(n), rng.uniform(-1, 1, (n, m)))


def example():
    """CORA's example of split: the center (-4, 1) and three generators."""
    return zonotope([-4, 1], [[-3, -2, -1], [2, 3, 4]])


class Origin(unittest.TestCase):
    def test_origin(self):
        Z = cora.Zonotope.origin(3)
        np.testing.assert_array_equal(Z.center, np.zeros(3))
        self.assertEqual(Z.generators.shape, (3, 0))

    def test_errors(self):
        for n in (0, -1):
            with self.assertRaises(cora.errors.WrongValue):
                cora.Zonotope.origin(n)


class Measures(unittest.TestCase):
    def test_radius(self):
        Z = zonotope([0, 0], [[1, 3], [2, 1]])
        self.assertAlmostEqual(Z.radius(), 5)  # the box radius (4, 3)
        self.assertEqual(cora.Zonotope.origin(2).radius(), 0)

    def test_radius_bounds_the_set(self):
        rng = np.random.default_rng(31)
        Z = random_zonotope(rng, 4, 5)
        points = Z.rand_point(300, cora.Rng(31), "extreme")
        self.assertLessEqual(np.linalg.norm(points - Z.center[:, None], axis=0).max(), Z.radius() + 1e-12)
        self.assertLessEqual(Z.radius(), np.linalg.norm(Z.generators, axis=0).sum() + 1e-12)

    def test_rank(self):
        Z = zonotope([1, 5, -1], [[2, 0, 4], [6, 0, 0], [4, 0, 8]])
        self.assertEqual(Z.rank(), 2)
        self.assertEqual(cora.Zonotope.origin(2).rank(), 0)
        thin = zonotope([0, 0], [[1, 0], [0, 1e-9]])
        self.assertEqual((thin.rank(1e-12), thin.rank(tol=1e-8)), (2, 1))
        with self.assertRaises(cora.errors.WrongValue):
            thin.rank(-1)
        with self.assertRaises(cora.errors.NotSupported):
            cora.Zonotope.stack([thin, thin]).rank()

    def test_rank_of_products(self):
        rng = np.random.default_rng(41)
        for n, m, r in ((4, 6, 2), (5, 5, 5), (3, 7, 1)):
            G = rng.standard_normal((n, r)) @ rng.standard_normal((r, m))
            Z = zonotope(np.zeros(n), G)
            self.assertEqual(Z.rank(), r)
            self.assertEqual(Z.is_full_dim(), Z.rank() == n)

    def test_representsa(self):
        origin, seg = cora.Zonotope.origin(2), zonotope([0, 0], [[1], [0]])
        box, tilted = zonotope([0, 0], [[1, 0], [0, 2]]), zonotope([0, 0], [[1, 0], [1, 2]])
        hexagon = zonotope([0, 0], [[1, 0, 1], [0, 1, 1]])
        self.assertTrue(origin.representsa("origin", 0) and not seg.representsa("origin", 0))
        self.assertTrue(origin.representsa("point") and not seg.representsa("point"))
        self.assertTrue(seg.representsa("capsule") and not hexagon.representsa("capsule"))
        self.assertTrue(box.representsa("interval") and not tilted.representsa("interval"))
        self.assertTrue(tilted.representsa("parallelotope") and not hexagon.representsa("parallelotope"))
        for kind in ("zonotope", "conZonotope", "polytope", "polyZonotope", "zonoBundle", "convexSet"):
            self.assertTrue(hexagon.representsa(kind), kind)
        for kind in ("halfspace", "hyperplane", "probZonotope", "emptySet", "fullspace"):
            self.assertFalse(hexagon.representsa(kind), kind)

    def test_representsa_errors(self):
        Z = zonotope([0, 0], [[1], [0]])
        with self.assertRaises(cora.errors.WrongValue):
            Z.representsa("circle")
        for kind in ("conHyperplane", "ellipsoid", "polygon"):
            with self.assertRaises(cora.errors.NotSupported):
                Z.representsa(kind)


class Embedding(unittest.TestCase):
    def test_project_high_dim(self):
        Z = zonotope([1, 2], [[1, 0], [0, 3]])
        R = Z.project_high_dim(4, [3, 1])
        np.testing.assert_array_equal(R.center, [0, 2, 0, 1])
        np.testing.assert_array_equal(R.generators, [[0, 0], [0, 3], [0, 0], [1, 0]])
        with self.assertRaises(cora.errors.WrongValue):
            Z.project_high_dim(1, [0, 0])
        with self.assertRaises(cora.errors.WrongValue):
            Z.project_high_dim(3, [0, 3])

    def test_rotate(self):
        Z = zonotope([1, 0, 5], [[1, 0], [0, 2], [2, 0]])
        R = Z.rotate([0, 1], np.pi / 2)
        np.testing.assert_allclose(R.center, [0, 1, 5], atol=1e-15)
        np.testing.assert_allclose(R.generators, [[0, -2], [1, 0], [2, 0]], atol=1e-15)
        for dims in ([0], [0, 0], [0, 3], [-1, 1]):
            with self.assertRaises(cora.errors.WrongValue):
                Z.rotate(dims, 1.0)

    def test_rotation_is_orthogonal(self):
        rng = np.random.default_rng(81)
        Z, a = random_zonotope(rng, 4, 5), 1.1
        R = Z.rotate([1, 3], a)
        M = np.eye(4)
        M[np.ix_([1, 3], [1, 3])] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
        for d in rng.standard_normal((20, 4)):
            self.assertAlmostEqual(R.support_function(d), Z.support_function(M.T @ d), places=10)
        undone = R.rotate([1, 3], -a)
        np.testing.assert_allclose(undone.center, Z.center, atol=1e-12)
        np.testing.assert_allclose(undone.generators, Z.generators, atol=1e-12)


class QuadMap(unittest.TestCase):
    Q = [np.array([[1.0, -2.0], [-3.0, 4.0]]), np.array([[0.5, 0.0], [2.0, -1.0]])]

    def test_by_hand(self):
        Z1 = zonotope([-4, 1], [[-3, -2], [2, 3]])
        Z2 = zonotope([1, -3], [[4, 2], [2, -1]])
        R = Z1.quad_map(self.Q)
        np.testing.assert_allclose(R.center, [102.5, -16.25])
        np.testing.assert_allclose(R.generators, [[27.5, 35, 95, 110, 125], [-5.75, -9.5, -14, -26, -32]])
        M = Z1.quad_map(Z2, self.Q)
        np.testing.assert_allclose(M.center, [-43, 3])
        np.testing.assert_allclose(M.generators[:, :4], [[-51, -59, -4, -8], [8.5, 14, -2, 6]])

    def test_the_map_of_the_set_is_enclosed(self):
        rng = np.random.default_rng(61)
        Z, Z2 = random_zonotope(rng, 3, 3), random_zonotope(rng, 3, 2)
        Q = [rng.standard_normal((3, 3)) for _ in range(2)]
        R, M = Z.quad_map(Q), Z.quad_map(Z2, Q)
        x, y = Z.rand_point(30, cora.Rng(1), "extreme"), Z2.rand_point(30, cora.Rng(2))
        for k in range(30):
            self.assertTrue(R.contains(np.array([x[:, k] @ q @ x[:, k] for q in Q]), tol=1e-9))
            self.assertTrue(M.contains(np.array([x[:, k] @ q @ y[:, k] for q in Q]), tol=1e-9))

    def test_errors(self):
        Z = zonotope([1, 2], [[1], [0.5]])
        with self.assertRaises(cora.errors.DimensionMismatch):
            Z.quad_map([])
        with self.assertRaises(cora.errors.DimensionMismatch):
            Z.quad_map([np.ones((3, 3))])

    def test_interval(self):
        I = cora.Interval(np.array([-1.0]), np.array([1.0]))
        R = I.quad_map([np.array([[2.0]])])
        np.testing.assert_allclose((R.inf, R.sup), ([0.0], [2.0]))
        X = cora.Interval(np.array([1.0]), np.array([2.0]))
        Y = cora.Interval(np.array([-1.0]), np.array([3.0]))
        M = X.quad_map(Y, [np.array([[1.0]])])
        np.testing.assert_allclose((M.inf, M.sup), ([-3.0], [6.0]))
        with self.assertRaises(cora.errors.WrongValue):
            cora.Interval(np.array([0.0]), np.array([np.inf])).quad_map([np.array([[1.0]])])


class Splits(unittest.TestCase):
    def test_by_coordinate(self):
        pairs = example().split()
        self.assertEqual(len(pairs), 2)
        np.testing.assert_allclose(pairs[0][0].center, [-7, 1])
        np.testing.assert_allclose(pairs[0][1].center, [-1, 1])
        np.testing.assert_allclose(pairs[0][0].generators, [[3, 0], [0, 9]])
        lower, upper = example().split(1)
        np.testing.assert_allclose((lower.center, upper.center), ([-4, -3.5], [-4, 5.5]))
        np.testing.assert_allclose(upper.generators, [[6, 0], [0, 4.5]])

    def test_along_a_generator(self):
        Z = zonotope([0, 0], [[2, 0], [0, 4]])
        lower, upper = Z.split(1, 2)
        np.testing.assert_allclose((lower.center, upper.center), ([0, -2], [0, 2]))
        np.testing.assert_allclose(upper.generators, [[2, 0], [0, 2]])

    def test_by_a_direction(self):
        one, two = example().split(np.array([1.0, 1.0]))
        np.testing.assert_allclose((one.center, two.center), ([-5.25, -0.25], [-2.75, 2.25]))
        np.testing.assert_allclose(one.generators, [[1.25, -2.5, -2.5, -2.5], [1.25, 2.5, 2.5, 2.5]])
        p, q = example().split(np.array([1.0, 0.0]), np.array([1.0, 1.0]))
        r, s = example().split(np.array([0.0, 1.0]))
        np.testing.assert_allclose(p.center, r.center)
        np.testing.assert_allclose(q.generators, s.generators)

    def test_by_a_halfspace(self):
        pieces = example().split(np.array([1.0, -1.0]), -2.0)
        self.assertEqual(len(pieces), 2)
        np.testing.assert_allclose((pieces[0].center, pieces[1].center), ([-7, 4], [0.5, -3.5]))
        self.assertEqual(len(example().split(np.array([1.0, 0.0]), 100.0)), 1)

    def test_pieces_cover_the_set(self):
        rng = np.random.default_rng(91)
        Z = random_zonotope(rng, 3, 4)
        points = Z.rand_point(30, cora.Rng(91))
        for first, second in (Z.split(1, 4), Z.split(np.array([1.0, -2.0, 0.5])), Z.split_first_gen(),
                              Z.split_longest_gen()):
            for k in range(30):
                p = points[:, k]
                self.assertTrue(first.contains(p, tol=1e-9) or second.contains(p, tol=1e-9))

    def test_first_and_longest_generator(self):
        Z = zonotope([1, 1], [[1, 0.1, 0], [0, 0.1, 3]])
        lower, upper = Z.split_first_gen()
        np.testing.assert_allclose((lower.center, upper.center), ([1, -0.5], [1, 2.5]))
        np.testing.assert_allclose(upper.generators, [[1, 0.1, 0], [0, 0.1, 1.5]])
        first, second = Z.split_longest_gen()
        np.testing.assert_allclose((first.center, second.center), ([1, 2.5], [1, -0.5]))

    def test_errors(self):
        Z = example()
        parallel = (np.array([1.0, 2.0]), np.array([2.0, 4.0]))
        for call in (lambda: Z.split(2), lambda: Z.split(0, 0), lambda: Z.split(5, 1),
                     lambda: Z.split(np.zeros(2)), lambda: Z.split(*parallel)):
            with self.assertRaises(cora.errors.WrongValue):
                call()
        batch = cora.Zonotope.stack([Z, Z])
        for call in (lambda: batch.split(np.ones(2)), lambda: batch.split_first_gen(),
                     lambda: batch.split_longest_gen()):
            with self.assertRaises(cora.errors.NotSupported):
                call()


class ReduceUnderApprox(unittest.TestCase):
    Z = zonotope([1, 0], [[1, 3, -2, 3, -1, 0, 2, 3], [2, -1, 1, 0, 3, -2, 1, 2]])

    def test_by_hand(self):
        for name in ("kochdumper", "sum"):
            np.testing.assert_allclose(self.Z.reduce_under_approx(name, 1).generators, [[3, 12], [2, -2]])
        np.testing.assert_allclose(self.Z.reduce_under_approx("raghuraman", 1).generators, [[2, 5], [4, 0]])

    def test_the_result_is_inside(self):
        rng = np.random.default_rng(121)
        for name in ("yang", "raghuraman", "kochdumper"):
            for n, m, order in ((2, 8, 1.0), (3, 9, 2.0)):
                Z = random_zonotope(rng, n, m)
                R = Z.reduce_under_approx(name, order)
                self.assertLessEqual(R.generators.shape[1], order * n)
                self.assertTrue(Z.contains(R, tol=1e-10), name)

    def test_nothing_to_reduce(self):
        Z = zonotope([0, 0], [[1, 0, 2], [0, 0, 1]])
        np.testing.assert_array_equal(Z.reduce_under_approx("yang", 1).generators, [[1, 2], [0, 1]])

    def test_errors(self):
        for name in ("sadraddini", "scale", "boxLP", "boxCone", "boxWetzlinger", "nlp", "cluster"):
            with self.assertRaises(cora.errors.NotSupported):
                self.Z.reduce_under_approx(name, 1)
        for name, order in (("boxNLP", 1), ("yang", 0), ("kochdumper", 0.4)):
            with self.assertRaises(cora.errors.WrongValue):
                self.Z.reduce_under_approx(name, order)


class Unite(unittest.TestCase):
    @staticmethod
    def box(x0, x1, y0, y1):
        return zonotope([(x0 + x1) / 2, (y0 + y1) / 2], [[(x1 - x0) / 2, 0], [0, (y1 - y0) / 2]])

    def test_two_boxes(self):
        A, B = self.box(0, 1, 0, 1), self.box(2, 3, 0, 1)
        for method in ("linprog", "iterative", "parallelotope"):
            U = A.unite(B, method)
            hull = U.interval()
            np.testing.assert_allclose((hull.inf, hull.sup), ([0, 0], [3, 1]), atol=1e-9)
            self.assertTrue(U.contains(A, tol=1e-9) and U.contains(B, tol=1e-9), method)
        self.assertTrue((A | B).contains(B, tol=1e-9))

    def test_other_operands(self):
        A = self.box(0, 1, 0, 1)
        box = cora.Interval(np.array([2.0, 0.0]), np.array([3.0, 1.0]))
        np.testing.assert_allclose(A.unite(box).interval().sup, [3, 1], atol=1e-9)
        np.testing.assert_allclose(A.unite(np.array([5.0, 0.5])).interval().sup, [5, 1], atol=1e-9)
        U = A.unite([self.box(2, 3, 0, 1), self.box(-4, -3, 0, 1)], "iterative")
        hull = U.interval()  # the pairwise enclosure overshoots: [-5, 3] x [0, 1]
        np.testing.assert_allclose((hull.inf, hull.sup), ([-5, 0], [3, 1]), atol=1e-9)

    def test_the_union_contains_the_sets(self):
        rng = np.random.default_rng(131)
        Zs = [random_zonotope(rng, 3, 4) for _ in range(3)]
        for method in ("linprog", "iterative", "parallelotope"):
            U = Zs[0].unite(Zs[1:], method)
            for Z in Zs:
                self.assertTrue(U.contains(Z, tol=1e-8), method)

    def test_errors(self):
        A, B = self.box(0, 1, 0, 1), self.box(2, 3, 0, 1)
        for method in ("tedrake", "althoff"):
            with self.assertRaises(cora.errors.NotSupported):
                A.unite(B, method)
        with self.assertRaises(cora.errors.NotSupported):
            A.unite(B, "linprog", 2)
        with self.assertRaises(cora.errors.WrongValue):
            A.unite(B, "convex")
        with self.assertRaises(cora.errors.DimensionMismatch):
            A.unite(cora.Zonotope.origin(3))


if __name__ == "__main__":
    unittest.main()
