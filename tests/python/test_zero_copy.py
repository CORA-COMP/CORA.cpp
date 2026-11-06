"""test_zero_copy - numpy arrays cross into C++ and back without a copy

A C-contiguous float64 (or float32) array becomes an Eigen tensor over its own memory, and a tensor
goes back as a numpy array over the tensor's memory: a set built from G hands G's memory back. Only
an array that is not C-contiguous, or of another dtype, is copied. The benchmark times the round
trip of a 1000 x 1000 array against one copy of it.

Authors:       Tobias Ladner
Written:       03-November-2026
Last update:   ---
Reviewed:      ---
"""
import time
import unittest

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

try:
    import torch
except ImportError:
    torch = None


# ============================================  MAIN  ============================================ #

class ZeroCopy(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("eigen"))
        rng = np.random.default_rng(3)
        self.c = rng.standard_normal(1000)
        self.G = rng.standard_normal((1000, 1000))

    def test_what_comes_in_comes_back(self):
        Z = cora.Zonotope(self.c, self.G)
        self.assertTrue(np.shares_memory(Z.generators, self.G))
        self.assertTrue(np.shares_memory(Z.center, self.c))
        # the set's memory is not the caller's to change
        self.assertFalse(Z.generators.flags.writeable)

    def test_a_result_is_handed_over_without_a_copy(self):
        Z = cora.Zonotope(self.c[:3], self.G[:3, :4].copy())
        W = Z + Z
        self.assertTrue(np.shares_memory(W.generators, W.generators))
        self.assertFalse(np.shares_memory(W.generators, Z.generators))
        # a fresh result that nothing else holds is the caller's
        points = Z.rand_point(5, cora.Rng(1))
        self.assertTrue(points.flags.writeable)

    def test_only_a_strided_or_foreign_array_is_copied(self):
        Gt = self.G[:5, :5].T  # Fortran order: copied once, with the same values
        Z = cora.Zonotope(self.c[:5], Gt)
        self.assertFalse(np.shares_memory(Z.generators, Gt))
        np.testing.assert_array_equal(Z.generators, Gt)
        Z = cora.Zonotope(self.c[:2].astype(np.float32), self.G[:2, :2].astype(np.float32))
        self.assertEqual(Z.generators.dtype, np.float32)
        Zi = cora.Zonotope(np.array([1, 2]), np.array([[1, 0], [0, 1]]))
        self.assertEqual(Zi.generators.dtype, np.float64)

    def test_benchmark_no_copy_per_call(self):
        calls = 200
        start = time.perf_counter()
        for _ in range(calls):
            G = cora.Zonotope(self.c, self.G).generators
        per_call = (time.perf_counter() - start) / calls
        start = time.perf_counter()
        for _ in range(calls):
            copy = self.G.copy()
        per_copy = (time.perf_counter() - start) / calls
        print(f"\n  1000 x 1000 in and out of a Zonotope: {per_call * 1e6:.1f} us per call; "
              f"one copy of the array: {per_copy * 1e6:.1f} us")
        self.assertTrue(np.shares_memory(G, self.G) and copy.shape == self.G.shape)
        # the call is cheaper than a copy
        self.assertLess(per_call, per_copy / 5)

    @unittest.skipUnless(torch is not None and cora.has_torch, "needs torch")
    def test_a_torch_tensor_stays_itself(self):
        G = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        Z = cora.Zonotope(torch.zeros(3, dtype=torch.float64), G)
        self.assertEqual(Z.generators.data_ptr(), G.data_ptr())
        Z.generators.sum().backward()
        self.assertTrue(torch.equal(G.grad, torch.ones(3, 4, dtype=torch.float64)))


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
