"""coracpp.simulate and coracpp.rand_point: exact trajectories, random points, batching,
gradients, and simulations that stay inside the reachable sets."""
import itertools
import unittest

import numpy as np
import torch

import coracpp

DTYPE = torch.float64


def oscillator(n=3, m=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, generator=g, dtype=DTYPE) * 0.5 - 0.3 * torch.eye(n, dtype=DTYPE)
    return A, torch.randn(n, generator=g, dtype=DTYPE), torch.randn(n, m, generator=g, dtype=DTYPE)


def support(c, G, d):
    """max over the zonotope (c, G) along d, for tensors with any leading dimensions."""
    return (d * c).sum(-1) + (d.unsqueeze(-2) @ G).abs().sum((-2, -1))


class RandPoint(unittest.TestCase):
    def test_points_lie_in_the_set(self):
        _, c, G = oscillator()
        P = coracpp.rand_point(c, G, 2000, seed=3)
        self.assertEqual(tuple(P.shape), (3, 2000))
        for _ in range(20):
            d = torch.randn(3, dtype=DTYPE)
            self.assertLessEqual(float((d @ P).max()), float(support(c, G, d)) + 1e-9)
        self.assertLess(float((P.mean(-1) - c).abs().max()), 0.15)

    def test_seed(self):
        _, c, G = oscillator()
        self.assertTrue(torch.equal(coracpp.rand_point(c, G, 10, seed=1), coracpp.rand_point(c, G, 10, seed=1)))
        self.assertFalse(torch.equal(coracpp.rand_point(c, G, 10, seed=1), coracpp.rand_point(c, G, 10, seed=2)))

    def test_extreme_points_are_corners(self):
        _, c, G = oscillator(m=3)
        P = coracpp.rand_point(c, G, 30, seed=0, extreme=True)
        corners = torch.stack([c + G @ torch.tensor(b, dtype=DTYPE) for b in itertools.product([-1.0, 1.0], repeat=3)])
        nearest = (P.T[:, None, :] - corners[None]).norm(dim=-1).min(dim=1).values
        self.assertLess(float(nearest.max()), 1e-10)

    def test_batches_and_numpy(self):
        _, c, G = oscillator()
        cs, Gs = torch.stack([c, 2 * c]), torch.stack([G, G])
        self.assertEqual(tuple(coracpp.rand_point(cs, Gs, 7).shape), (2, 3, 7))
        P = coracpp.rand_point(c.numpy(), G.numpy(), 7, seed=4)
        self.assertIsInstance(P, np.ndarray)
        self.assertEqual(P.shape, (3, 7))


class Simulate(unittest.TestCase):
    def test_follows_the_matrix_exponential(self):
        A, c, G = oscillator()
        x0 = coracpp.rand_point(c, G, 5, seed=1)
        x = coracpp.simulate(A, x0, 0.1, 1.0)
        self.assertEqual(tuple(x.shape), (11, 3, 5))
        for k in range(11):
            torch.testing.assert_close(x[k], torch.matrix_exp(A * 0.1 * k) @ x0, atol=1e-12, rtol=1e-12)

    def test_numpy_agrees_with_torch(self):
        A, c, G = oscillator()
        x0 = coracpp.rand_point(c, G, 5, seed=1)
        on_eigen = coracpp.simulate(A.numpy(), x0.numpy(), 0.1, 1.0)
        self.assertIsInstance(on_eigen, np.ndarray)
        np.testing.assert_allclose(on_eigen, coracpp.simulate(A, x0, 0.1, 1.0).numpy(), atol=1e-12)

    def test_batches(self):
        A, c, G = oscillator()
        As = torch.stack([A, 0.5 * A, A - 0.1 * torch.eye(3, dtype=DTYPE)])
        x0 = coracpp.rand_point(c, G, 4, seed=2)
        x = coracpp.simulate(As, x0, 0.1, 0.5)
        self.assertEqual(tuple(x.shape), (6, 3, 3, 4))
        for b in range(3):
            torch.testing.assert_close(x[:, b], coracpp.simulate(As[b], x0, 0.1, 0.5))

    def test_gradient(self):
        A, c, G = oscillator(n=2, m=2)
        x0 = coracpp.rand_point(c, G, 3, seed=1)
        self.assertTrue(torch.autograd.gradcheck(
            lambda M: coracpp.simulate(M, x0, 0.2, 0.6).pow(2).sum(), (A.clone().requires_grad_(),)))

    def test_simulations_stay_in_the_reachable_set(self):
        # The check every reachability tool is measured against, between time points too.
        A, c, G = oscillator(seed=5)
        refine, dt = 5, 0.2
        for algorithm in ("standard", "wrapping-free"):
            r = coracpp.reach(A, c, G, time_step=dt, t_final=2.0, taylor_terms=8, algorithm=algorithm)
            for extreme in (False, True):
                x = coracpp.simulate(A, coracpp.rand_point(c, G, 50, seed=7, extreme=extreme), dt / refine, 2.0)
                d = torch.randn(30, 3, dtype=DTYPE)
                for j in range(x.shape[0]):
                    k = min(j // refine, r.time_int_c.shape[0] - 1)
                    bound = support(r.time_int_c[k], r.time_int_G[k], d)          # (30,)
                    best = (d @ x[j]).max(-1).values                              # (30,)
                    self.assertGreaterEqual(float((bound - best).min()), -1e-9, (algorithm, extreme, j))

    @unittest.skipUnless(torch.cuda.is_available(), "no CUDA device")
    def test_gpu(self):
        A, c, G = oscillator()
        x0 = coracpp.rand_point(c.cuda(), G.cuda(), 5, seed=1)
        x = coracpp.simulate(A.cuda(), x0, 0.1, 1.0)
        self.assertEqual(x.device.type, "cuda")
        torch.testing.assert_close(x.cpu(), coracpp.simulate(A, x0.cpu(), 0.1, 1.0))


if __name__ == "__main__":
    unittest.main()
