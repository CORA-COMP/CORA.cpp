"""test_cont_set_generic - the operations every set has (ContSet) on a box, a zonotope and a set
written in Python: center, dim, is_bounded, is_full_dim, reduce, linear_map, enlarge, minus,
minkowski_difference, isequal, eq, ne, decompose, project_high_dim, project_on_hyperplane,
generate_random and the unchecked and_, contains_ and isIntersecting_ forms."""
# CORA: contSet/@contSet/center.m
# CORA: contSet/@contSet/dim.m
# CORA: contSet/@contSet/isBounded.m
# CORA: contSet/@contSet/isFullDim.m
# CORA: contSet/@contSet/reduce.m
# CORA: contSet/@contSet/mtimes.m
# CORA: contSet/@contSet/enlarge.m
# CORA: contSet/@contSet/minus.m
# CORA: contSet/@contSet/minkDiff.m
# CORA: contSet/@contSet/isequal.m
# CORA: contSet/@contSet/eq.m
# CORA: contSet/@contSet/ne.m
# CORA: contSet/@contSet/decompose.m
# CORA: contSet/@contSet/projectHighDim.m
# CORA: contSet/@contSet/projectHighDim_.m
# CORA: contSet/@contSet/projectOnHyperplane.m
# CORA: contSet/@contSet/generateRandom.m
# CORA: contSet/@contSet/and_.m
# CORA: contSet/@contSet/contains_.m
# CORA: contSet/@contSet/isIntersecting_.m
# CORA: contSet/@contSet/supportFunc.m
# CORA: contSet/@contSet/supportFunc_.m
# CORA: contSet/@contSet/uminus.m
# CORA: contSet/@contSet/uplus.m
# CORA: contSet/@contSet/times.m
# CORA: contSet/@contSet/compact.m
# CORA: contSet/@contSet/compact_.m
# CORA: contSet/@contSet/cartProd_.m
# CORA: contSet/@contSet/convHull_.m
# PROPERTY: contSet/@contSet/projectOnHyperplane.m

import unittest

import numpy as np

import cora


def box(lo, hi):
    return cora.Interval(np.array(lo, dtype=float), np.array(hi, dtype=float))


def zonotope(c, G):
    return cora.Zonotope(np.array(c, dtype=float), np.array(G, dtype=float))


class Custom(cora.ContSet):
    """A set that implements only the layer the tests of the base class need."""

    def _dim(self):
        return 2

    def _center(self):
        return np.zeros(2)


class Generic(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("eigen"))
        self.I = box([-1.0, 0.0], [1.0, 2.0])
        self.Z = zonotope([1.0, 0.0], [[0.5, 0.0], [0.0, 0.5]])

    def test_center_dim_and_queries(self):
        self.assertEqual(cora.ContSet.center.fget(self.I).tolist(), [0.0, 1.0])
        self.assertEqual((self.I.dim, self.Z.dim), (2, 2))
        self.assertTrue(cora.ContSet.is_bounded(self.I) and cora.ContSet.is_bounded(self.Z))
        self.assertFalse(cora.Interval.unbounded(2).is_bounded())
        self.assertTrue(cora.ContSet.is_full_dim(self.Z))
        self.assertFalse(cora.ContSet.is_full_dim(zonotope([0.0, 0.0], [[1.0], [1.0]])))
        thin = box([0.0], [1e-4])
        self.assertTrue(thin.is_full_dim())
        self.assertFalse(thin.is_full_dim(tol=1e-3))
        with self.assertRaises(cora.errors.WrongValue):
            self.I.is_full_dim(tol=-1.0)

    def test_maps_and_shifts(self):
        M = np.array([[1.0, 1.0], [0.0, 1.0]])
        R = cora.ContSet.linear_map(self.I, M)
        self.assertEqual((R.inf.tolist(), R.sup.tolist()), ([-1.0, 0.0], [3.0, 2.0]))
        self.assertEqual(cora.ContSet.linear_map(self.Z, M).generators.tolist(), [[0.5, 0.5], [0.0, 0.5]])
        E = cora.ContSet.enlarge(self.Z, np.array([2.0, 3.0]))
        self.assertEqual(E.center.tolist(), [1.0, 0.0])
        self.assertEqual(E.generators.tolist(), [[1.0, 0.0], [0.0, 1.5]])
        self.assertEqual(cora.ContSet.enlarge(self.I, 2.0).sup.tolist(), [2.0, 3.0])
        m = cora.ContSet.minus(self.Z, np.array([1.0, 1.0]))
        self.assertEqual(m.center.tolist(), [0.0, -1.0])
        d = cora.ContSet.minkowski_difference(self.I, box([-0.5, 0.0], [0.5, 0.5]))
        self.assertEqual((d.inf.tolist(), d.sup.tolist()), ([-0.5, 0.0], [0.5, 1.5]))
        with self.assertRaises(cora.errors.DimensionMismatch):
            cora.ContSet.linear_map(self.I, np.ones((1, 3)))

    def test_reduce_decompose_and_embedding(self):
        Z = cora.Zonotope.generate_random(2, 7, cora.Rng(5))
        R = cora.ContSet.reduce(Z, 1)
        self.assertEqual(np.asarray(R.generators).shape[1], 2)
        self.assertEqual(cora.ContSet.reduce(self.I, 3).sup.tolist(), self.I.sup.tolist())
        with self.assertRaises(cora.errors.WrongValue):
            cora.ContSet.reduce(self.I, 0)
        I3 = box([-1.0, 0.0, 2.0], [1.0, 2.0, 5.0])
        parts = cora.ContSet.decompose(I3, [[0, 1], [2, 2]])
        self.assertEqual([p.dim for p in parts], [2, 1])
        self.assertEqual(parts[1].inf.tolist(), [2.0])
        with self.assertRaises(cora.errors.WrongValue):
            I3.decompose([[1, 3]])
        H = cora.ContSet.project_high_dim(self.Z, 4, [2, 0])
        self.assertEqual(H.center.tolist(), [0.0, 0.0, 1.0, 0.0])
        self.assertEqual(self.I.project_high_dim(4, [2, 0]).sup.tolist(), [2.0, 0.0, 1.0, 0.0])
        with self.assertRaises(cora.errors.WrongValue):
            cora.ContSet.project_high_dim(self.Z, 1, [0, 0])

    def test_project_on_hyperplane(self):
        Z = zonotope([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])
        R = cora.ContSet.project_on_hyperplane(Z, np.array([3.0, 3.0]), 6.0)
        self.assertAlmostEqual(R.support_function(np.array([1.0, 1.0])), 2.0)
        self.assertAlmostEqual(R.support_function(np.array([-1.0, -1.0])), -2.0)
        self.assertAlmostEqual(R.support_function(np.array([1.0, 0.0])), 2.0)
        self.assertAlmostEqual(R.support_function(np.array([-1.0, 0.0])), 0.0)
        with self.assertRaises(cora.errors.WrongValue):
            Z.project_on_hyperplane(np.zeros(2), 1.0)

    def test_equality(self):
        Zi = zonotope([0.0, 1.0], [[1.0, 0.0], [0.0, 1.0]])
        self.assertTrue(cora.isequal(Zi, self.I) and cora.isequal(self.I, Zi))
        self.assertTrue(self.I == Zi and Zi == self.I and self.I != self.Z)
        self.assertNotEqual(self.I, box([-1.0, 0.0], [1.0, 2.001]))
        self.assertTrue(cora.isequal(self.I, box([-1.0, 0.0], [1.0, 2.001]), tol=1e-2))
        self.assertFalse(self.I == 3)
        with self.assertRaises(cora.errors.DimensionMismatch):
            cora.isequal(self.I, box([0.0], [1.0]))
        with self.assertRaises(cora.errors.WrongValue):
            cora.isequal(self.I, self.I, tol=-1.0)

    def test_generate_random(self):
        rng = cora.Rng(3)
        S = cora.ContSet.generate_random(rng, 3, ["interval"])
        self.assertTrue(isinstance(S, cora.Interval) and S.dim == 3)
        Z = cora.ContSet.generate_random(rng, 4, ["zonotope"])
        self.assertEqual(np.asarray(Z.generators).shape, (4, 8))
        classes = {type(cora.ContSet.generate_random(rng)) for _ in range(40)}
        self.assertEqual(classes, {cora.Interval, cora.Zonotope})
        with self.assertRaises(cora.errors.NotSupported):
            cora.ContSet.generate_random(rng, 2, ["polytope"])

    def test_unchecked_forms_and_pairs(self):
        J = box([0.5, 1.0], [3.0, 3.0])
        X = cora.intersect(self.I, J)
        self.assertEqual((X.inf.tolist(), X.sup.tolist()), ([0.5, 1.0], [1.0, 2.0]))
        Z = zonotope([0.0, 1.0], [[0.5, 0.0], [0.0, 0.5]])
        self.assertTrue(cora.contains(Z, box([-0.25, 0.75], [0.25, 1.25])))
        self.assertFalse(cora.contains(Z, J))
        self.assertTrue(cora.is_intersecting(self.I, J))
        self.assertIn(("isequal", "Zonotope", "Interval"), cora.binary_pairs())

    def test_support_function(self):
        d = np.array([1.0, -1.0])
        self.assertEqual(cora.ContSet.support_function(self.I, d), 1.0)
        self.assertEqual(cora.ContSet.support_function(self.Z, d), 2.0)
        with self.assertRaises(cora.errors.DimensionMismatch):
            cora.ContSet.support_function(self.I, np.ones(3))

    def test_uminus_uplus_and_times(self):
        N = cora.ContSet.__neg__(self.Z)
        self.assertEqual(N.center.tolist(), [-1.0, 0.0])
        self.assertEqual(N.generators.tolist(), [[-0.5, 0.0], [0.0, -0.5]])
        self.assertEqual((-self.I).sup.tolist(), [1.0, 0.0])
        P = +self.Z
        self.assertEqual(P.center.tolist(), self.Z.center.tolist())
        self.assertEqual(P.generators.tolist(), self.Z.generators.tolist())
        T = cora.ContSet.times(self.Z, np.array([2.0, -1.0]))
        self.assertEqual(T.center.tolist(), [2.0, 0.0])
        self.assertEqual(T.generators.tolist(), [[1.0, 0.0], [0.0, -0.5]])
        self.assertEqual(cora.ContSet.times(self.Z, 3.0).center.tolist(), [3.0, 0.0])
        for E in (cora.EmptySet(2), cora.Fullspace(2)):
            self.assertEqual(type(-E), type(E))
            self.assertEqual(type(+E), type(E))
            self.assertEqual(type(cora.ContSet.times(E, np.array([1.0, 2.0]))), type(E))
        with self.assertRaises(cora.errors.DimensionMismatch):
            cora.ContSet.times(self.I, np.ones(3))
        with self.assertRaises(cora.errors.Noops):
            -Custom()
        with cora.checks(False):
            self.assertEqual(cora.ContSet.times(self.I, np.ones(2)).sup.tolist(), [1.0, 2.0])

    def test_compact(self):
        Z = zonotope([0.0, 0.0], [[1.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        self.assertEqual(np.asarray(cora.ContSet.compact(Z).generators).shape, (2, 3))
        self.assertEqual(np.asarray(cora.ContSet.compact(Z, "all").generators).shape, (2, 2))
        self.assertEqual(np.asarray(cora.ContSet.compact(Z, "aligned").generators).shape, (2, 4))
        self.assertEqual(np.asarray(cora.ContSet.compact(Z, "zeros", 1.5).generators).shape, (2, 1))
        self.assertEqual(cora.ContSet.compact(self.I, "anything", -1.0).sup.tolist(), [1.0, 2.0])
        self.assertIsInstance(cora.ContSet.compact(cora.EmptySet(2)), cora.EmptySet)
        self.assertIsInstance(cora.ContSet.compact(cora.Fullspace(2)), cora.Fullspace)
        with self.assertRaises(cora.errors.WrongValue):
            cora.ContSet.compact(Z, "nonsense")
        with self.assertRaises(cora.errors.WrongValue):
            cora.ContSet.compact(Z, "zeros", -1.0)
        with self.assertRaises(cora.errors.Noops):
            Custom().compact()

    def test_a_pair_without_an_entry_is_noops(self):
        for operation in (cora.cart_prod, cora.convex_hull):
            with self.assertRaises(cora.errors.Noops):
                operation(self.Z, Custom())

    def test_a_python_set_has_no_operation(self):
        C = Custom()
        for call in (C.is_bounded, C.is_full_dim, lambda: C.reduce(1), lambda: C.enlarge(2.0)):
            with self.assertRaises(cora.errors.Noops):
                call()
        with self.assertRaises(cora.errors.Noops):
            C == self.I
        with cora.checks(False):
            self.assertTrue(self.I.is_full_dim(tol=-1.0))


if __name__ == "__main__":
    unittest.main()
