"""The Python module: backend choice by input type, batching, autograd through the binding.

    make python TORCH=...  &&  PYTHONPATH=build python -m unittest tests/test_python.py
"""
import unittest

import numpy as np
import torch

import coracpp

ALGORITHMS = ["standard", "wrapping-free"]
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def system(n=3, m=4, batch=(), seed=0, dtype=torch.float64, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    make = lambda *shape: torch.randn(*shape, generator=g, dtype=dtype).to(device)
    return make(*batch, n, n) * 0.5, make(*batch, n), make(*batch, n, m)


class Reach(unittest.TestCase):
    def test_numpy_runs_on_eigen_and_agrees_with_torch(self):
        A, c, G = system()
        for algorithm in ALGORITHMS:
            t = coracpp.reach(A, c, G, 0.1, 1.0, linAlg=algorithm)
            n = coracpp.reach(A.numpy(), c.numpy(), G.numpy(), 0.1, 1.0, linAlg=algorithm)
            self.assertIsInstance(n.timeInt_G, np.ndarray)
            for a, b in [(t.timeInt_c, n.timeInt_c), (t.timeInt_G, n.timeInt_G),
                         (t.timePoint_c, n.timePoint_c), (t.timePoint_G, n.timePoint_G)]:
                np.testing.assert_allclose(a.numpy(), b, atol=1e-10)

    def test_shapes(self):
        A, c, G = system(n=3, m=4)
        r = coracpp.reach(A, c, G, 0.1, 1.0)
        self.assertEqual(r.timeInt_c.shape, (10, 3))
        self.assertEqual(r.timeInt_G.shape, (10, 3, 3 * 4 + 3 + 1))  # 2m+1 from linComb, m+n from F
        self.assertEqual(r.timePoint_c.shape, (11, 3))

    def test_batching_equals_a_loop(self):
        # `vmap`: the call is written for one set and one system, and batches by itself.
        for algorithm in ALGORITHMS:
            A, c, G = system(batch=(5,))
            both = coracpp.reach(A, c, G, 0.1, 1.0, linAlg=algorithm)
            for b in range(5):
                one = coracpp.reach(A[b], c[b], G[b], 0.1, 1.0, linAlg=algorithm)
                torch.testing.assert_close(both.timeInt_G[:, b], one.timeInt_G)
                torch.testing.assert_close(both.timePoint_c[:, b], one.timePoint_c)

    def test_one_system_many_sets_and_many_systems_one_set(self):
        A, c, G = system(batch=(4,))
        sets = coracpp.reach(A[0], c, G, 0.1, 1.0)
        systems = coracpp.reach(A, c[0], G[0], 0.1, 1.0)
        for b in range(4):
            torch.testing.assert_close(sets.timeInt_G[:, b],
                                       coracpp.reach(A[0], c[b], G[b], 0.1, 1.0).timeInt_G)
            torch.testing.assert_close(systems.timeInt_G[:, b],
                                       coracpp.reach(A[b], c[0], G[0], 0.1, 1.0).timeInt_G)

    def test_gradients_through_the_binding(self):
        for algorithm in ALGORITHMS:
            for custom in (False, True):
                A, c, G = (t.requires_grad_() for t in system(n=2, m=2))

                def f(A, c, G):
                    r = coracpp.reach(A, c, G, 0.2, 0.6, taylorTerms=6, linAlg=algorithm,
                                      customBackward=custom)
                    return r.timeInt_G.abs().sum() + r.timeInt_c.sum()

                self.assertTrue(torch.autograd.gradcheck(f, (A, c, G), atol=1e-5, rtol=1e-4))

    def test_customBackward_matches_autograd(self):
        grads = []
        for custom in (False, True):
            A, c, G = (t.requires_grad_() for t in system(seed=4))
            r = coracpp.reach(A, c, G, 0.1, 1.0, customBackward=custom)
            (r.timeInt_G.abs().sum() + r.timePoint_c.pow(2).sum()).backward()
            grads.append([A.grad, c.grad, G.grad])
        for a, b in zip(*grads):
            torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-9)

    @unittest.skipUnless("cuda" in DEVICES, "no CUDA device")
    def test_gpu_agrees_with_cpu(self):
        A, c, G = system(batch=(3,))
        cpu = coracpp.reach(A, c, G, 0.1, 1.0)
        gpu = coracpp.reach(A.cuda(), c.cuda(), G.cuda(), 0.1, 1.0)
        self.assertEqual(gpu.timeInt_G.device.type, "cuda")
        torch.testing.assert_close(cpu.timeInt_G, gpu.timeInt_G.cpu())

    def test_bad_algorithm(self):
        A, c, G = system()
        with self.assertRaises(ValueError):
            coracpp.reach(A, c, G, 0.1, 1.0, linAlg="nope")


if __name__ == "__main__":
    unittest.main()
