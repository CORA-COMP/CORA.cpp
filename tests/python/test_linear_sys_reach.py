"""test_linear_sys_reach - LinearSys.reach in Python: the same call for one set or a batch, on torch

or numpy, with gradients and the options checked.

CORA authors: Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger
Authors:       Tobias Ladner
Written:       02-November-2026
Last update:   03-November-2026
Reviewed:      ---
"""
# CORA: contDynamics/@linearSys/linearSys.m
# CORA: contDynamics/@linearSys/reach.m

import unittest

import numpy as np


try:
    import torch
except ImportError:  # numpy-only build: test_numpy.py covers it
    raise unittest.SkipTest("needs torch") from None

import cora


if not cora.has_torch:  # torch is installed, but this build has no torch backend
    raise unittest.SkipTest("needs a build with torch")

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

F64 = dict(dtype=torch.float64)
ALGORITHMS = ["standard", "wrapping_free"]


def system(n=3, m=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, generator=g, **F64) * 0.5 - 0.3 * torch.eye(n, **F64)
    return A, torch.randn(n, generator=g, **F64), torch.randn(n, m, generator=g, **F64)


# ============================================  MAIN  ============================================ #

class ReachSet(unittest.TestCase):
    def test_the_result_is_two_lists_of_zonotopes(self):
        A, c, G = system()
        R = cora.LinearSys(A).reach(cora.Zonotope(c, G), time_step=0.1, t_final=1.0)
        self.assertEqual((len(R.time_interval), len(R.time_point)), (10, 11))
        self.assertIsInstance(R.time_interval[0], cora.Zonotope)
        self.assertEqual(tuple(R.time_interval[0].center.shape), (3,))
        self.assertEqual(R.time_interval[0].generators.shape[-1],
                         3 * 4 + 3 + 1)  # 2m+1 from linear_combination, m+n from F

    def test_numpy_runs_on_eigen_and_agrees_with_torch(self):
        A, c, G = system()
        for lin_alg in ALGORITHMS:
            t = cora.LinearSys(A).reach(cora.Zonotope(c, G), time_step=0.1, t_final=1.0, alg=lin_alg)
            n = cora.LinearSys(A.numpy()).reach(cora.Zonotope(c.numpy(), G.numpy()), time_step=0.1,
                                                t_final=1.0, alg=lin_alg)
            self.assertIsInstance(n.time_interval[0].generators, np.ndarray)
            for zt, zn in zip(t.time_interval + t.time_point, n.time_interval + n.time_point, strict=True):
                np.testing.assert_allclose(zt.center.numpy(), zn.center, atol=1e-10)
                np.testing.assert_allclose(zt.generators.numpy(), zn.generators, atol=1e-10)

    def test_a_batch_of_sets_equals_a_loop(self):
        # The batch is in the object: the call is the one of a single set.
        A, _, _ = system()
        c, G = torch.randn(5, 3, **F64), torch.randn(5, 3, 4, **F64)
        sys = cora.LinearSys(A)
        for lin_alg in ALGORITHMS:
            both = sys.reach(cora.Zonotope(c, G), time_step=0.1, t_final=1.0, alg=lin_alg)
            for b in range(5):
                one = sys.reach(cora.Zonotope(c[b], G[b]), time_step=0.1, t_final=1.0, alg=lin_alg)
                for zb, z1 in zip(both.time_interval + both.time_point, one.time_interval + one.time_point,
                                  strict=True):
                    torch.testing.assert_close(zb.center[b], z1.center)
                    torch.testing.assert_close(zb.generators[b], z1.generators)

    def test_a_batch_of_systems_and_of_sets_together(self):
        As = torch.stack([system(seed=s)[0] for s in range(3)])
        c, G = torch.randn(3, 3, **F64), torch.randn(3, 3, 2, **F64)
        both = cora.LinearSys(As).reach(cora.Zonotope(c, G), time_step=0.1, t_final=0.5)
        systems = cora.LinearSys(As).reach(cora.Zonotope(c[0], G[0]), time_step=0.1, t_final=0.5)
        for b in range(3):
            one = cora.LinearSys(As[b]).reach(cora.Zonotope(c[b], G[b]), time_step=0.1, t_final=0.5)
            torch.testing.assert_close(both.time_interval[-1].generators[b], one.time_interval[-1].generators)
            single = cora.LinearSys(As[b]).reach(cora.Zonotope(c[0], G[0]), time_step=0.1, t_final=0.5)
            # the batch equals the single runs
            torch.testing.assert_close(systems.time_interval[-1].generators[b],
                                       single.time_interval[-1].generators)

    def test_the_algorithms_share_their_time_points(self):
        A, c, G = system()
        Z = cora.Zonotope(c, G)
        s = cora.LinearSys(A).reach(Z, time_step=0.1, t_final=1.0, alg="standard")
        w = cora.LinearSys(A).reach(Z, time_step=0.1, t_final=1.0, alg="wrapping_free")
        for a, b in zip(s.time_point, w.time_point, strict=True):
            torch.testing.assert_close(a.generators, b.generators, atol=1e-9, rtol=1e-9)

    def test_the_first_enclosure_matches_matlab_cora(self):
        # Reference: MATLAB CORA R2024b, x' = [-0.2 1; -1 -0.2] x, zonotope([1; 0.5], [0.1 0; 0 0.2]).
        sys = cora.LinearSys(torch.tensor([[-0.2, 1.0], [-1.0, -0.2]], **F64))
        X0 = cora.Zonotope(torch.tensor([1.0, 0.5], **F64), torch.tensor([[0.1, 0.0], [0.0, 0.2]], **F64))
        R = sys.reach(X0, time_step=0.1, t_final=0.5, taylor_terms=8)
        for d, want in ((torch.tensor([1.0, 0.0], **F64), 1.14551093304521),
                        (torch.tensor([0.0, 1.0], **F64), 0.710682871515058),
                        (torch.tensor([1.0, -2.0], **F64), 0.765085217909499)):
            self.assertAlmostEqual(float(R.time_interval[0].support_function(d)), want, places=12)

    def test_an_unknown_lin_alg_is_described(self):
        A, c, G = system()
        with self.assertRaises(ValueError) as caught:
            cora.LinearSys(A).reach(cora.Zonotope(c, G), time_step=0.1, t_final=1.0, alg="nope")
        self.assertIn("'standard'", str(caught.exception))
        self.assertIn("'wrapping_free'", str(caught.exception))

    @unittest.skipUnless(torch.cuda.is_available(), "no CUDA device")
    def test_gpu_agrees_with_cpu(self):
        A, c, G = system()
        cpu = cora.LinearSys(A).reach(cora.Zonotope(c, G), time_step=0.1, t_final=1.0)
        gpu = cora.LinearSys(A.cuda()).reach(cora.Zonotope(c.cuda(), G.cuda()), time_step=0.1, t_final=1.0)
        self.assertEqual(gpu.time_interval[0].generators.device.type, "cuda")
        torch.testing.assert_close(cpu.time_interval[-1].generators, gpu.time_interval[-1].generators.cpu())


# gradients through the objects
class Gradients(unittest.TestCase):
    def test_gradcheck_through_the_objects(self):
        for lin_alg in ALGORITHMS:
            for custom in (False, True):
                A, c, G = (t.requires_grad_() for t in system(n=2, m=2))

                def f(A, c, G, custom=custom, lin_alg=lin_alg):
                    sys = cora.LinearSys(A, custom_backward=custom)
                    R = sys.reach(cora.Zonotope(c, G), time_step=0.2, t_final=0.6, taylor_terms=6,
                                  alg=lin_alg)
                    return sum(Z.generators.abs().sum() + Z.center.sum() for Z in R.time_interval)

                self.assertTrue(torch.autograd.gradcheck(f, (A, c, G), atol=1e-5, rtol=1e-4))

    def test_custom_backward_matches_autograd(self):
        grads = []
        for custom in (False, True):
            A, c, G = (t.requires_grad_() for t in system(seed=4))
            R = cora.LinearSys(A, custom_backward=custom).reach(cora.Zonotope(c, G), time_step=0.1,
                                                                t_final=1.0)
            sum(Z.generators.abs().sum() for Z in R.time_interval).backward()
            grads.append([A.grad, c.grad, G.grad])
        for a, b in zip(*grads, strict=True):
            # the gradients agree
            torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-9)


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
