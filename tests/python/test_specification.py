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
OPTIONS = dict(time_step=0.1, t_final=6.0, taylor_terms=8)


def x1(value):
    return torch.tensor([value, 0.0], **F64)


class Specification(unittest.TestCase):
    def setUp(self):
        self.R = cora.LinearSys(A).reach(X0, **OPTIONS)

    def test_safe_set(self):
        # The oscillator swings right to about 1.1 at the start and about 0.9 later.
        self.assertTrue(cora.Specification.safe_set(x1(1.0), 1.5).check(self.R.time_interval))
        self.assertFalse(cora.Specification.safe_set(x1(1.0), 1.0).check(self.R.time_interval))

    def test_first_violation_is_a_step(self):
        self.assertEqual(cora.Specification.safe_set(x1(1.0), 1.0).first_violation(self.R.time_interval), 0)
        self.assertEqual(cora.Specification.safe_set(x1(1.0), 1.5).first_violation(self.R.time_interval), -1)
        # The trajectory swings left of -0.5 later on: the first enclosure to touch is a late one.
        late = cora.Specification.unsafe_set(x1(1.0), -0.5).first_violation(self.R.time_interval)
        self.assertGreater(late, 5)
        self.assertLess(late, len(self.R.time_interval))

    def test_unsafe_set(self):
        self.assertTrue(cora.Specification.unsafe_set(x1(1.0), -3.0).check(self.R.time_interval))
        self.assertFalse(cora.Specification.unsafe_set(x1(1.0), 0.5).check(self.R.time_interval))

    def test_any_set_can_be_checked(self):
        spec = cora.Specification.safe_set(x1(1.0), 1.2)
        self.assertTrue(spec.check(X0))
        self.assertTrue(spec.check(cora.Interval(x1(0.9), x1(1.1))))
        self.assertFalse(spec.check(cora.Interval(x1(0.9), x1(1.3))))

    def test_holds_answers_for_each_member_of_a_batch(self):
        c = torch.tensor([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], **F64)
        batch = cora.Zonotope(c, torch.eye(2, **F64).expand(3, 2, 2).contiguous())
        self.assertEqual(cora.Specification.safe_set(x1(1.0), 3.0).holds(batch), [True, True, False])
        self.assertEqual(cora.Specification.unsafe_set(x1(1.0), 1.5).holds(batch), [False, False, True])

    def test_type_and_halfspaces(self):
        spec = cora.Specification.safe_set(x1(1.0), 1.5)
        self.assertEqual(spec.type, "safe_set")
        ((a, b),) = spec.halfspaces
        np.testing.assert_allclose(a, [1.0, 0.0])
        self.assertEqual(b, 1.5)
        self.assertEqual(cora.Specification.unsafe_set(x1(1.0), 0.0).type, "unsafe_set")

    def test_numpy_runs_on_eigen(self):
        R = cora.LinearSys(A.numpy()).reach(cora.Zonotope(X0.center.numpy(), X0.generators.numpy()),
                                            **OPTIONS)
        spec = cora.Specification.safe_set(np.array([1.0, 0.0]), 1.0)
        self.assertFalse(spec.check(R.time_interval))
        self.assertEqual(spec.first_violation(R.time_interval), 0)

    def test_the_two_backends_agree(self):
        R = cora.LinearSys(A.numpy()).reach(cora.Zonotope(X0.center.numpy(), X0.generators.numpy()),
                                            **OPTIONS)
        for bound in (0.8, 1.0, 1.2, 1.5):
            on_torch = cora.Specification.safe_set(x1(1.0), bound)
            on_eigen = cora.Specification.safe_set(np.array([1.0, 0.0]), bound)
            self.assertEqual(on_torch.first_violation(self.R.time_interval),
                             on_eigen.first_violation(R.time_interval))

    def test_backends_do_not_mix(self):
        with self.assertRaises(ValueError):
            cora.Specification.safe_set(np.array([1.0, 0.0]), 1.5).check(self.R.time_interval)


if __name__ == "__main__":
    unittest.main()
