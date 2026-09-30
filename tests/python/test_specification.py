"""test_specification - the Specification class in Python: halfspaces against any set."""
import unittest

import numpy as np
try:
    import torch
except ImportError:  # numpy-only build: test_numpy.py covers it
    raise unittest.SkipTest("needs torch")

import cora

F64 = dict(dtype=torch.float64)
A = torch.tensor([[-0.1, 1.0], [-1.0, -0.1]], **F64)
X0 = cora.Zonotope(torch.tensor([1.0, 0.0], **F64), 0.1 * torch.eye(2, **F64))
OPTIONS = dict(timeStep=0.1, tFinal=6.0, taylorTerms=8)


def x1(value):
    return torch.tensor([value, 0.0], **F64)


class Specification(unittest.TestCase):
    def setUp(self):
        self.R = cora.LinearSys(A).reach(X0, **OPTIONS)

    def test_safeSet(self):
        # The oscillator swings right to about 1.1 at the start and about 0.9 later.
        self.assertTrue(cora.Specification.safeSet(x1(1.0), 1.5).check(self.R.timeInt))
        self.assertFalse(cora.Specification.safeSet(x1(1.0), 1.0).check(self.R.timeInt))

    def test_firstViolation_is_a_step(self):
        self.assertEqual(cora.Specification.safeSet(x1(1.0), 1.0).firstViolation(self.R.timeInt), 0)
        self.assertEqual(cora.Specification.safeSet(x1(1.0), 1.5).firstViolation(self.R.timeInt), -1)
        # The trajectory swings left of -0.5 later on: the first enclosure to touch is a late one.
        late = cora.Specification.unsafeSet(x1(1.0), -0.5).firstViolation(self.R.timeInt)
        self.assertGreater(late, 5)
        self.assertLess(late, len(self.R.timeInt))

    def test_unsafeSet(self):
        self.assertTrue(cora.Specification.unsafeSet(x1(1.0), -3.0).check(self.R.timeInt))
        self.assertFalse(cora.Specification.unsafeSet(x1(1.0), 0.5).check(self.R.timeInt))

    def test_any_set_can_be_checked(self):
        spec = cora.Specification.safeSet(x1(1.0), 1.2)
        self.assertTrue(spec.check(X0))
        self.assertTrue(spec.check(cora.Interval(x1(0.9), x1(1.1))))
        self.assertFalse(spec.check(cora.Interval(x1(0.9), x1(1.3))))

    def test_holds_answers_for_each_member_of_a_batch(self):
        c = torch.tensor([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], **F64)
        batch = cora.Zonotope(c, torch.eye(2, **F64).expand(3, 2, 2).contiguous())
        self.assertEqual(cora.Specification.safeSet(x1(1.0), 3.0).holds(batch), [True, True, False])
        self.assertEqual(cora.Specification.unsafeSet(x1(1.0), 1.5).holds(batch), [False, False, True])

    def test_type_and_halfspaces(self):
        spec = cora.Specification.safeSet(x1(1.0), 1.5)
        self.assertEqual(spec.type, "safeSet")
        ((a, b),) = spec.halfspaces
        np.testing.assert_allclose(a, [1.0, 0.0])
        self.assertEqual(b, 1.5)
        self.assertEqual(cora.Specification.unsafeSet(x1(1.0), 0.0).type, "unsafeSet")

    def test_numpy_runs_on_eigen(self):
        R = cora.LinearSys(A.numpy()).reach(cora.Zonotope(X0.c.numpy(), X0.G.numpy()), **OPTIONS)
        spec = cora.Specification.safeSet(np.array([1.0, 0.0]), 1.0)
        self.assertFalse(spec.check(R.timeInt))
        self.assertEqual(spec.firstViolation(R.timeInt), 0)

    def test_the_two_backends_agree(self):
        R = cora.LinearSys(A.numpy()).reach(cora.Zonotope(X0.c.numpy(), X0.G.numpy()), **OPTIONS)
        for bound in (0.8, 1.0, 1.2, 1.5):
            on_torch = cora.Specification.safeSet(x1(1.0), bound)
            on_eigen = cora.Specification.safeSet(np.array([1.0, 0.0]), bound)
            self.assertEqual(on_torch.firstViolation(self.R.timeInt), on_eigen.firstViolation(R.timeInt))

    def test_backends_do_not_mix(self):
        with self.assertRaises(ValueError):
            cora.Specification.safeSet(np.array([1.0, 0.0]), 1.5).check(self.R.timeInt)


if __name__ == "__main__":
    unittest.main()
