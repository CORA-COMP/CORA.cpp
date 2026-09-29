"""coracpp.Specification: halfspaces checked against reachable sets, from torch and numpy."""
import unittest

import numpy as np
import torch

import coracpp

A = torch.tensor([[-0.1, 1.0], [-1.0, -0.1]], dtype=torch.float64)
C = torch.tensor([1.0, 0.0], dtype=torch.float64)
G = 0.1 * torch.eye(2, dtype=torch.float64)
OPTIONS = dict(timeStep=0.1, tFinal=6.0, taylorTerms=8)


def x1(value):
    return torch.tensor([value, 0.0], dtype=torch.float64)


class Specification(unittest.TestCase):
    def setUp(self):
        self.reach = coracpp.reach(A, C, G, **OPTIONS)

    def test_safeSet(self):
        # The oscillator swings right to about 1.1 at the start and about 0.9 later.
        self.assertTrue(coracpp.Specification.safeSet(x1(1.0), 1.5).check(self.reach))
        self.assertFalse(coracpp.Specification.safeSet(x1(1.0), 1.0).check(self.reach))

    def test_firstViolation_is_a_step(self):
        step = coracpp.Specification.safeSet(x1(1.0), 1.0).firstViolation(self.reach)
        self.assertEqual(step, 0)
        self.assertEqual(coracpp.Specification.safeSet(x1(1.0), 1.5).firstViolation(self.reach), -1)

        # The trajectory swings left of -0.5 later on: the first enclosure to touch is a late one.
        wall = coracpp.Specification.unsafeSet(x1(1.0), -0.5)
        late = wall.firstViolation(self.reach)
        self.assertGreater(late, 5)
        self.assertLess(late, self.reach.timeInt_c.shape[0])

    def test_unsafeSet(self):
        self.assertTrue(coracpp.Specification.unsafeSet(x1(1.0), -3.0).check(self.reach))
        self.assertFalse(coracpp.Specification.unsafeSet(x1(1.0), 0.5).check(self.reach))

    def test_numpy_runs_on_eigen(self):
        r = coracpp.reach(A.numpy(), C.numpy(), G.numpy(), **OPTIONS)
        safe = coracpp.Specification.safeSet(np.array([1.0, 0.0]), 1.5)
        unsafe = coracpp.Specification.safeSet(np.array([1.0, 0.0]), 1.0)
        self.assertTrue(safe.check(r))
        self.assertFalse(unsafe.check(r))
        self.assertEqual(unsafe.firstViolation(r), 0)

    def test_the_two_backends_agree(self):
        r = coracpp.reach(A.numpy(), C.numpy(), G.numpy(), **OPTIONS)
        for bound in (0.8, 1.0, 1.2, 1.5):
            torch_spec = coracpp.Specification.safeSet(x1(1.0), bound)
            numpy_spec = coracpp.Specification.safeSet(np.array([1.0, 0.0]), bound)
            self.assertEqual(torch_spec.firstViolation(self.reach), numpy_spec.firstViolation(r))

    def test_backends_do_not_mix(self):
        with self.assertRaises(ValueError):
            coracpp.Specification.safeSet(np.array([1.0, 0.0]), 1.5).check(self.reach)

    @unittest.skipUnless(torch.cuda.is_available(), "no CUDA device")
    def test_gpu(self):
        r = coracpp.reach(A.cuda(), C.cuda(), G.cuda(), **OPTIONS)
        spec = coracpp.Specification.safeSet(x1(1.0).cuda(), 1.0)
        self.assertEqual(spec.firstViolation(r), 0)


if __name__ == "__main__":
    unittest.main()
