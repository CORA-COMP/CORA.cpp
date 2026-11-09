"""test_specification - the Specification class in Python: a polytope against any set.

CORA authors: Niklas Kochdumper, Tobias Ladner, Mark Wetzlinger
Authors:       Claude Sonnet 5.5, Tobias Ladner
Written:       29-September-2026
Last update:   03-November-2026
Reviewed:      ---
"""
# CORA: specification/@specification/check.m
# CORA: specification/@specification/specification.m

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
A = torch.tensor([[-0.1, 1.0], [-1.0, -0.1]], **F64)
X0 = cora.Zonotope(torch.tensor([1.0, 0.0], **F64), 0.1 * torch.eye(2, **F64))
OPTIONS = dict(time_step=0.1, t_final=6.0, taylor_terms=8)


def x1(value):
    return torch.tensor([value, 0.0], **F64)


def halfspace(offset):
    """The halfspace x1 <= offset as a polytope on the torch backend."""
    return cora.Polytope(x1(1.0)[None, :], torch.tensor([offset], **F64))


def halfspace_np(offset):
    """The same on the numpy (Eigen) backend."""
    return cora.Polytope(np.array([[1.0, 0.0]]), np.array([offset]))


# ============================================  MAIN  ============================================ #

class Specification(unittest.TestCase):
    def setUp(self):
        self.R = cora.LinearSys(A).reach(X0, **OPTIONS)

    def test_safe_set(self):
        # The oscillator swings right to about 1.1 at the start and about 0.9 later.
        self.assertTrue(cora.Specification.safe_set(halfspace(1.5)).check(self.R.time_interval))
        self.assertFalse(cora.Specification.safe_set(halfspace(1.0)).check(self.R.time_interval))

    def test_first_violation_is_a_step(self):
        R = self.R.time_interval
        self.assertEqual(cora.Specification.safe_set(halfspace(1.0)).first_violation(R), 0)
        self.assertEqual(cora.Specification.safe_set(halfspace(1.5)).first_violation(R), -1)
        # The trajectory swings left of -0.5 later on: the first enclosure to touch is a late one.
        late = cora.Specification.unsafe_set(halfspace(-0.5)).first_violation(self.R.time_interval)
        self.assertGreater(late, 5)
        self.assertLess(late, len(self.R.time_interval))

    def test_unsafe_set(self):
        self.assertTrue(cora.Specification.unsafe_set(halfspace(-3.0)).check(self.R.time_interval))
        self.assertFalse(cora.Specification.unsafe_set(halfspace(0.5)).check(self.R.time_interval))

    def test_any_set_can_be_checked(self):
        spec = cora.Specification.safe_set(halfspace(1.2))
        self.assertTrue(spec.check(X0))
        self.assertTrue(spec.check(cora.Interval(x1(0.9), x1(1.1))))
        self.assertFalse(spec.check(cora.Interval(x1(0.9), x1(1.3))))

    def test_a_batch_of_sets_is_refused(self):
        c = torch.tensor([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], **F64)
        batch = cora.Zonotope(c, torch.eye(2, **F64).expand(3, 2, 2).contiguous())
        with self.assertRaises(cora.errors.NotSupported):
            cora.Specification.safe_set(halfspace(3.0)).check(batch)

    def test_type_and_set(self):
        spec = cora.Specification.safe_set(halfspace(1.5))
        self.assertEqual(spec.type, "safe_set")
        # the polytope of the specification, with its rows
        np.testing.assert_allclose(spec.set.A, [[1.0, 0.0]])
        np.testing.assert_allclose(spec.set.b, [1.5])
        self.assertEqual(cora.Specification.unsafe_set(halfspace(0.0)).type, "unsafe_set")

    def test_numpy_runs_on_eigen(self):
        R = cora.LinearSys(A.numpy()).reach(cora.Zonotope(X0.center.numpy(), X0.generators.numpy()),
                                            **OPTIONS)
        spec = cora.Specification.safe_set(halfspace_np(1.0))
        self.assertFalse(spec.check(R.time_interval))
        self.assertEqual(spec.first_violation(R.time_interval), 0)

    def test_the_two_backends_agree(self):
        R = cora.LinearSys(A.numpy()).reach(cora.Zonotope(X0.center.numpy(), X0.generators.numpy()),
                                            **OPTIONS)
        for bound in (0.8, 1.0, 1.2, 1.5):
            on_torch = cora.Specification.safe_set(halfspace(bound))
            on_eigen = cora.Specification.safe_set(halfspace_np(bound))
            self.assertEqual(on_torch.first_violation(self.R.time_interval),
                             on_eigen.first_violation(R.time_interval))

    def test_backends_do_not_mix(self):
        with self.assertRaises(ValueError):
            cora.Specification.safe_set(halfspace_np(1.5)).check(self.R.time_interval)


# run the tests
if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
