"""test_linearSys_reach - LinearSys.reach in Python: the same call for one set or a batch, on torch
or numpy, with gradients and the options checked."""
import unittest

import numpy as np
import torch

import cora

F64 = dict(dtype=torch.float64)
ALGORITHMS = ["standard", "wrapping-free"]


def system(n=3, m=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, generator=g, **F64) * 0.5 - 0.3 * torch.eye(n, **F64)
    return A, torch.randn(n, generator=g, **F64), torch.randn(n, m, generator=g, **F64)


class Reach(unittest.TestCase):
    def test_the_result_is_two_lists_of_zonotopes(self):
        A, c, G = system()
        R = cora.LinearSys(A).reach(cora.Zonotope(c, G), timeStep=0.1, tFinal=1.0)
        self.assertEqual((len(R.timeInt), len(R.timePoint)), (10, 11))
        self.assertIsInstance(R.timeInt[0], cora.Zonotope)
        self.assertEqual(tuple(R.timeInt[0].c.shape), (3,))
        self.assertEqual(R.timeInt[0].G.shape[-1], 3 * 4 + 3 + 1)  # 2m+1 from linComb, m+n from F

    def test_numpy_runs_on_eigen_and_agrees_with_torch(self):
        A, c, G = system()
        for linAlg in ALGORITHMS:
            t = cora.LinearSys(A).reach(cora.Zonotope(c, G), 0.1, 1.0, linAlg=linAlg)
            n = cora.LinearSys(A.numpy()).reach(cora.Zonotope(c.numpy(), G.numpy()), 0.1, 1.0, linAlg=linAlg)
            self.assertIsInstance(n.timeInt[0].G, np.ndarray)
            for zt, zn in zip(t.timeInt + t.timePoint, n.timeInt + n.timePoint):
                np.testing.assert_allclose(zt.c.numpy(), zn.c, atol=1e-10)
                np.testing.assert_allclose(zt.G.numpy(), zn.G, atol=1e-10)

    def test_a_batch_of_sets_equals_a_loop(self):
        # The batch is in the object: the call is the one of a single set.
        A, _, _ = system()
        c, G = torch.randn(5, 3, **F64), torch.randn(5, 3, 4, **F64)
        sys = cora.LinearSys(A)
        for linAlg in ALGORITHMS:
            both = sys.reach(cora.Zonotope(c, G), 0.1, 1.0, linAlg=linAlg)
            for b in range(5):
                one = sys.reach(cora.Zonotope(c[b], G[b]), 0.1, 1.0, linAlg=linAlg)
                for zb, z1 in zip(both.timeInt + both.timePoint, one.timeInt + one.timePoint):
                    torch.testing.assert_close(zb.c[b], z1.c)
                    torch.testing.assert_close(zb.G[b], z1.G)

    def test_a_batch_of_systems_and_of_sets_together(self):
        As = torch.stack([system(seed=s)[0] for s in range(3)])
        c, G = torch.randn(3, 3, **F64), torch.randn(3, 3, 2, **F64)
        both = cora.LinearSys(As).reach(cora.Zonotope(c, G), 0.1, 0.5)
        systems = cora.LinearSys(As).reach(cora.Zonotope(c[0], G[0]), 0.1, 0.5)
        for b in range(3):
            one = cora.LinearSys(As[b]).reach(cora.Zonotope(c[b], G[b]), 0.1, 0.5)
            torch.testing.assert_close(both.timeInt[-1].G[b], one.timeInt[-1].G)
            single = cora.LinearSys(As[b]).reach(cora.Zonotope(c[0], G[0]), 0.1, 0.5)
            torch.testing.assert_close(systems.timeInt[-1].G[b], single.timeInt[-1].G)

    def test_the_algorithms_share_their_time_points(self):
        A, c, G = system()
        Z = cora.Zonotope(c, G)
        s = cora.LinearSys(A).reach(Z, 0.1, 1.0, linAlg="standard")
        w = cora.LinearSys(A).reach(Z, 0.1, 1.0, linAlg="wrapping-free")
        for a, b in zip(s.timePoint, w.timePoint):
            torch.testing.assert_close(a.G, b.G, atol=1e-9, rtol=1e-9)

    def test_the_first_enclosure_matches_matlab_cora(self):
        # Reference: MATLAB CORA R2024b, x' = [-0.2 1; -1 -0.2] x, zonotope([1; 0.5], [0.1 0; 0 0.2]).
        sys = cora.LinearSys(torch.tensor([[-0.2, 1.0], [-1.0, -0.2]], **F64))
        X0 = cora.Zonotope(torch.tensor([1.0, 0.5], **F64), torch.tensor([[0.1, 0.0], [0.0, 0.2]], **F64))
        R = sys.reach(X0, timeStep=0.1, tFinal=0.5, taylorTerms=8)
        for d, want in ((torch.tensor([1.0, 0.0], **F64), 1.14551093304521),
                        (torch.tensor([0.0, 1.0], **F64), 0.710682871515058),
                        (torch.tensor([1.0, -2.0], **F64), 0.765085217909499)):
            self.assertAlmostEqual(float(R.timeInt[0].supportFunc(d)), want, places=12)

    def test_an_unknown_linAlg_is_described(self):
        A, c, G = system()
        with self.assertRaises(ValueError) as caught:
            cora.LinearSys(A).reach(cora.Zonotope(c, G), 0.1, 1.0, linAlg="nope")
        self.assertIn("'standard'", str(caught.exception))
        self.assertIn("'wrapping-free'", str(caught.exception))

    @unittest.skipUnless(torch.cuda.is_available(), "no CUDA device")
    def test_gpu_agrees_with_cpu(self):
        A, c, G = system()
        cpu = cora.LinearSys(A).reach(cora.Zonotope(c, G), 0.1, 1.0)
        gpu = cora.LinearSys(A.cuda()).reach(cora.Zonotope(c.cuda(), G.cuda()), 0.1, 1.0)
        self.assertEqual(gpu.timeInt[0].G.device.type, "cuda")
        torch.testing.assert_close(cpu.timeInt[-1].G, gpu.timeInt[-1].G.cpu())


class Gradients(unittest.TestCase):
    def test_gradcheck_through_the_objects(self):
        for linAlg in ALGORITHMS:
            for custom in (False, True):
                A, c, G = (t.requires_grad_() for t in system(n=2, m=2))

                def f(A, c, G):
                    sys = cora.LinearSys(A, customBackward=custom)
                    R = sys.reach(cora.Zonotope(c, G), 0.2, 0.6, 6, linAlg=linAlg)
                    return sum(Z.G.abs().sum() + Z.c.sum() for Z in R.timeInt)

                self.assertTrue(torch.autograd.gradcheck(f, (A, c, G), atol=1e-5, rtol=1e-4))

    def test_custom_backward_matches_autograd(self):
        grads = []
        for custom in (False, True):
            A, c, G = (t.requires_grad_() for t in system(seed=4))
            R = cora.LinearSys(A, customBackward=custom).reach(cora.Zonotope(c, G), 0.1, 1.0)
            sum(Z.G.abs().sum() for Z in R.timeInt).backward()
            grads.append([A.grad, c.grad, G.grad])
        for a, b in zip(*grads):
            torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-9)


if __name__ == "__main__":
    unittest.main()
