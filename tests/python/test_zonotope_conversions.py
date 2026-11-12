"""test_zonotope_conversions - the zonotope filter, simplex, polytope, vertices and matrix zonotope.

CORA authors: Matthias Althoff, Niklas Kochdumper, Tobias Ladner
Authors:       Claude Sonnet 5.5
Written:       06-November-2026
Last update:   ---
Reviewed:      ---
"""
# CORA: contSet/@zonotope/dim.m
# CORA: contSet/@zonotope/filterOut.m
# CORA: contSet/@zonotope/generators.m
# CORA: contSet/@zonotope/matZonotope.m
# CORA: contSet/@zonotope/polytope.m
# CORA: contSet/@zonotope/simplex.m
# CORA: contSet/@zonotope/vertices_.m
# CORA: matrixSet/@intervalMatrix/matZonotope.m
# CORA: matrixSet/@matZonotope/expmIndMixed.m
# CORA: matrixSet/@matZonotope/numgens.m
# CORA: matrixSet/@matZonotope/reduce.m
# CORA: matrixSet/@matZonotope/vertices.m
# CORA: matrixSet/@matZonotope/volume.m

import unittest

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# ----------------------------------------  BEGIN CODE  ---------------------------------------- #

# ----------------------------------------  AUXILIARY  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

def zonotope(c, G):
    return cora.Zonotope(np.array(c, dtype=float), np.array(G, dtype=float))


# ===========================================  MAIN  =========================================== #


# ============================================  MAIN  ============================================ #

class ZonotopeConversions(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("eigen"))

    def test_filter_out(self):
        # a cell inside another is dropped, equal cells drop each other
        big = zonotope([0.0, 0.0], 2.0 * np.eye(2))
        small = zonotope([0.5, 0.0], np.eye(2))
        away = zonotope([10.0, 0.0], np.eye(2))
        self.assertEqual(cora.Zonotope.filter_out([small, big, away]), [1, 2])
        self.assertEqual(cora.Zonotope.filter_out([big, big]), [])
        with self.assertRaises(cora.errors.WrongValue):
            cora.Zonotope.filter_out([])

    def test_dim_and_generators(self):
        # the dimension and the generators as given
        Z = zonotope([1.0, 2.0, 3.0], [[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
        self.assertEqual(Z.dim, 3)
        np.testing.assert_array_equal(Z.generators, [[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
        self.assertEqual(cora.MatZonotope(np.zeros((2, 2))).num_generators, 0)

    def test_polytope_and_vertices(self):
        # the rectangle [0, 2] x [-1, 5]: four facets and four corners, counter-clockwise
        Z = zonotope([1.0, 2.0], [[1.0, 0.0], [0.0, 3.0]])
        Q = Z.polytope()
        self.assertEqual(Q.A.shape, (4, 2))
        np.testing.assert_allclose(sorted(Q.b.ravel()), [0.0, 1.0, 2.0, 5.0])
        corners = Z.vertices()[0]
        self.assertEqual(corners.shape, (4, 2))
        self.assertTrue(Q.contains(corners.T))
        np.testing.assert_allclose(sorted(map(tuple, corners)), [(0, -1), (0, 5), (2, -1), (2, 5)])

    def test_simplex_and_mat_zonotope(self):
        # the simplex has n + 1 facets and contains the zonotope
        rng = cora.Rng(97)
        Z = cora.Zonotope.generate_random(rng, 3)
        P = Z.simplex()
        self.assertEqual(P.A.shape, (4, 3))
        self.assertTrue(P.contains(Z.rand_point(20, rng, "extreme")))
        M = zonotope([1.0, 2.0], [[1.0, 0.0, 3.0], [0.0, 2.0, 4.0]]).mat_zonotope()
        self.assertEqual((M.rows, M.cols, M.num_generators), (2, 1, 3))
        np.testing.assert_array_equal(M.zonotope().generators, [[1.0, 0.0, 3.0], [0.0, 2.0, 4.0]])


class MatrixSets(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("eigen"))

    def test_interval_matrix_to_mat_zonotope(self):
        # one generator per entry, the same box
        A = cora.IntervalMatrix(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[0.5, 0.0], [0.0, 1.0]]))
        M = A.mat_zonotope()
        self.assertEqual(M.num_generators, 4)
        np.testing.assert_array_equal(M.center, [[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(M.interval_matrix().infimum, A.infimum)

    def test_mat_zonotope_vertices_volume_and_reduce(self):
        # the corner matrices in order, the volume 8 and the shape kept by reduce
        M = cora.MatZonotope(np.array([[1.0, 0.0], [0.0, 1.0]]),
                             np.array([[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 2.0]]]))
        V = M.vertices()[0]
        self.assertEqual(V.shape, (4, 2, 2))
        np.testing.assert_array_equal(V[0], [[0.0, 0.0], [0.0, -1.0]])
        np.testing.assert_array_equal(V[3], [[2.0, 0.0], [0.0, 3.0]])
        flat = cora.MatZonotope(np.zeros((1, 2)), np.array([[[1.0, 0.0]], [[0.0, 2.0]]]))
        self.assertAlmostEqual(float(flat.volume()), 8.0)
        rng = np.random.default_rng(98)
        big = cora.MatZonotope(rng.standard_normal((2, 2)), rng.uniform(-1, 1, (9, 2, 2)))
        reduced = big.reduce(1)
        self.assertEqual((reduced.rows, reduced.cols), (2, 2))
        self.assertLessEqual(reduced.num_generators, 4)
        with self.assertRaises(cora.errors.NotSupported):
            big.reduce(1, "pca")

    def test_expm_ind_mixed(self):
        # the lists of powers of both parts
        rng = np.random.default_rng(99)
        A = cora.MatZonotope(0.5 * rng.standard_normal((3, 3)), rng.uniform(-0.3, 0.3, (2, 3, 3)))
        R = A.expm_ind_mixed(2, 8)
        self.assertEqual((len(R.powers), len(R.powers2)), (2, 6))
        with self.assertRaises(cora.errors.WrongValue):
            A.expm_ind_mixed(0, 8)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------  END OF CODE  ---------------------------------------- #

# ----------------------------------------  END OF CODE  ----------------------------------------- #
