"""test_linear_sys_simulate - LinearSys.simulate and simulate_random in Python.

CORA authors: Matthias Althoff, Mark Wetzlinger
Authors:       Tobias Ladner
Written:       02-November-2026
Last update:   ---
Reviewed:      ---
"""
# CORA: contDynamics/@contDynamics/simulateRandom.m
# CORA: contDynamics/@linearSys/simulate.m

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


def system(n=3, m=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, generator=g, **F64) * 0.5 - 0.3 * torch.eye(n, **F64)
    return A, torch.randn(n, generator=g, **F64), torch.randn(n, m, generator=g, **F64)


# ============================================  MAIN  ============================================ #

class Simulate(unittest.TestCase):
    def test_follows_the_matrix_exponential(self):
        A, c, G = system()
        x0 = cora.Zonotope(c, G).rand_point(5, cora.Rng(1))
        x = cora.LinearSys(A).simulate(x0, time_step=0.1, t_final=1.0)
        self.assertEqual(tuple(x.shape), (11, 3, 5))
        for k in range(11):
            torch.testing.assert_close(x[k], torch.matrix_exp(A * 0.1 * k) @ x0, atol=1e-12, rtol=1e-12)

    def test_numpy_agrees_with_torch(self):
        A, c, G = system()
        x0 = cora.Zonotope(c, G).rand_point(5, cora.Rng(1))
        on_eigen = cora.LinearSys(A.numpy()).simulate(x0.numpy(), time_step=0.1, t_final=1.0)
        self.assertIsInstance(on_eigen, np.ndarray)
        np.testing.assert_allclose(on_eigen, cora.LinearSys(A).simulate(x0, time_step=0.1,
                                                                        t_final=1.0).numpy(), atol=1e-12)

    def test_a_batch_of_systems(self):
        A, c, G = system()
        As = torch.stack([A, 0.5 * A, A - 0.1 * torch.eye(3, **F64)])
        x0 = cora.Zonotope(c, G).rand_point(4, cora.Rng(2))
        x = cora.LinearSys(As).simulate(x0, time_step=0.1, t_final=0.5)
        self.assertEqual(tuple(x.shape), (6, 3, 3, 4))
        for b in range(3):
            torch.testing.assert_close(x[:, b], cora.LinearSys(As[b]).simulate(x0, time_step=0.1,
                                                                               t_final=0.5))

    def test_a_batch_of_sets(self):
        A, _, _ = system()
        c, G = torch.randn(4, 3, **F64), torch.randn(4, 3, 2, **F64)
        Z = cora.Zonotope(c, G)
        x = cora.LinearSys(A).simulate(Z.rand_point(6, cora.Rng(0)), time_step=0.1, t_final=0.5)
        self.assertEqual(tuple(x.shape), (6, 4, 3, 6))

    def test_gradient(self):
        A, c, G = system(n=2, m=2)
        x0 = cora.Zonotope(c, G).rand_point(3, cora.Rng(1))
        self.assertTrue(torch.autograd.gradcheck(
            lambda M: cora.LinearSys(M).simulate(x0, time_step=0.2,
                                                 t_final=0.6).pow(2).sum(), (A.clone().requires_grad_(),)))

    def test_simulations_stay_in_the_reachable_set(self):
        # The check every reachability tool is measured against, between time points too.
        A, c, G = system(seed=5)
        Z, sys = cora.Zonotope(c, G), cora.LinearSys(A)
        refine, dt = 5, 0.2
        for lin_alg in ("standard", "wrapping_free"):
            R = sys.reach(Z, time_step=dt, t_final=2.0, taylor_terms=8, alg=lin_alg)
            for kind in ("standard", "extreme"):
                x = sys.simulate(Z.rand_point(50, cora.Rng(7), type=kind), time_step=dt / refine, t_final=2.0)
                d = torch.randn(30, 3, **F64)
                for j in range(x.shape[0]):
                    step = min(j // refine, len(R.time_interval) - 1)
                    bound = R.time_interval[step].support_function(d)  # (30,)
                    best = (d @ x[j]).max(-1).values  # (30,)
                    self.assertGreaterEqual(float((bound - best).min()), -1e-9, (lin_alg, kind, j))


class SimulateRandom(unittest.TestCase):
    def test_a_seeded_draw_repeats(self):
        A, c, G = system()
        Z, sys = cora.Zonotope(c, G), cora.LinearSys(A)
        a = sys.simulate_random(Z, points=8, time_step=0.1, t_final=1.0, rng=cora.Rng(5))
        b = sys.simulate_random(Z, points=8, time_step=0.1, t_final=1.0, rng=cora.Rng(5))
        self.assertEqual(tuple(a.shape), (11, 3, 8))
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, sys.simulate_random(Z, points=8, time_step=0.1, t_final=1.0,
                                                            rng=cora.Rng(6))))

    def test_any_set_is_a_start(self):
        A, _, _ = system()
        I = cora.Interval(torch.tensor([0.9, -0.1, -0.1], **F64), torch.tensor([1.1, 0.1, 0.1], **F64))
        x = cora.LinearSys(A).simulate_random(I, points=5, time_step=0.1, t_final=0.5, rng=cora.Rng(1))
        self.assertEqual(tuple(x.shape), (6, 3, 5))
        self.assertTrue(bool(I.contains(x[0][:, 0])))

    @unittest.skipUnless(torch.cuda.is_available(), "no CUDA device")
    def test_gpu(self):
        A, c, G = system()
        x0 = cora.Zonotope(c.cuda(), G.cuda()).rand_point(5, cora.Rng(1))
        x = cora.LinearSys(A.cuda()).simulate(x0, time_step=0.1, t_final=1.0)
        self.assertEqual(x.device.type, "cuda")
        torch.testing.assert_close(x.cpu(), cora.LinearSys(A).simulate(x0.cpu(), time_step=0.1, t_final=1.0))


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
