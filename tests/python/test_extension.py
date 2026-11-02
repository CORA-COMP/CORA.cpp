"""test_extension - prototyping in Python: a ContSet subclass, a registered pair and a registered
algorithm run through the checked entries, the registry, plotting and reach like the built-in ones;
errors, warnings and the scoped settings."""
# CORA: contSet/@contSet/contSet.m
# CORA: contSet/@contSet/plus.m
# CORA: contSet/@contSet/project.m
# CORA: contSet/@contSet/vertices.m

import unittest
import warnings

import numpy as np

import cora


class Box(cora.ContSet):
    """A box c + r [-1, 1]^n written in Python: only the _impl layer."""

    def __init__(self, c, r):
        super().__init__()
        self.c, self.r = np.asarray(c, dtype=float), np.asarray(r, dtype=float)

    def _dim(self):
        return len(self.c)

    def _center(self):
        return self.c

    def _support_function(self, d):
        return float(d @ self.c + np.abs(d) @ self.r)

    def _interval(self):
        return cora.Interval(self.c - self.r, self.c + self.r)

    def _rand_point(self, N, rng):
        return self.c[:, None] + self.r[:, None] * (2 * np.random.default_rng(0).random((len(self.c), N)) - 1)

    def _project(self, dims):
        return Box(self.c[dims], self.r[dims])

    def _vertices(self):
        (x, y), (a, b) = self.c, self.r
        return [np.array([[x - a, y - b], [x + a, y - b], [x + a, y + b], [x - a, y + b]])]


class Subclass(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("eigen"))

    def test_the_checked_entries_check_a_python_set(self):
        B = Box([1.0, 2.0], [0.5, 0.25])
        self.assertAlmostEqual(B.support_function(np.array([1.0, 0.0])), 1.5)
        self.assertEqual(B.dim, 2)
        self.assertEqual(B.class_name, "Box")
        with self.assertRaises(cora.errors.DimensionMismatch):
            B.support_function(np.array([1.0, 0.0, 0.0]))
        with self.assertRaises(ValueError):  # the CORA classes are builtin errors too
            B.project([])
        with cora.checks(False):
            self.assertAlmostEqual(B.support_function(np.array([0.0, 1.0])), 2.25)

    def test_a_python_set_projects_and_has_vertices(self):
        B = Box([1.0, 2.0, 3.0], [0.5, 0.25, 1.0])
        self.assertIsInstance(B.project([2, 0]), Box)
        self.assertEqual(Box([0.0, 0.0], [1.0, 1.0]).vertices()[0].shape, (4, 2))

    def test_a_registered_pair_is_used_both_ways(self):
        def plus_box_zonotope(box, Z):
            return cora.Zonotope(box.c + Z.center, np.hstack([np.diag(box.r), Z.generators]))

        cora.register_binary("plus", Box, cora.Zonotope, plus_box_zonotope)
        self.assertIn(("plus", "Box", "Zonotope"), cora.binary_pairs())
        Z = cora.Zonotope(np.array([1.0, 0.0]), np.eye(2))
        B = Box([0.0, 1.0], [0.5, 0.5])
        for S in (B + Z, Z + B, cora.plus(B, Z)):
            self.assertIsInstance(S, cora.Zonotope)
            np.testing.assert_allclose(S.center, [1.0, 1.0])
        with self.assertRaises(cora.errors.Noops):
            B & Z

    def test_plotting_takes_a_python_set(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _, ax = plt.subplots()
        cora.plot(Box([0.0, 0.0], [1.0, 2.0]), ax=ax)
        plt.close("all")


@cora.register_algorithm(cora.LinearSys, "euler_box")
def euler_box(sys, problem, time_step=0.1):
    """A crude prototype: steps of x + dt A x on the zonotope, enlarged by a box."""
    A = np.asarray(sys.A)
    X, sets = problem.initial_set, [problem.initial_set]
    for _ in range(int(round(problem.t_final / time_step))):
        X = (np.eye(len(A)) + time_step * A) @ X
        sets.append(X)
    return cora.ReachSet(time_interval=sets[1:], time_point=sets)


class RegisteredAlgorithm(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("eigen"))
        self.sys = cora.LinearSys(np.array([[-1.0, 0.0], [0.0, -2.0]]))
        self.X0 = cora.Zonotope(np.array([1.0, 1.0]), 0.1 * np.eye(2))

    def test_reach_runs_it_by_name(self):
        R = self.sys.reach(self.X0, t_final=1.0, alg="euler_box", time_step=0.25)
        self.assertIsInstance(R, cora.ReachSet)
        self.assertEqual(len(R.time_point), 5)

    def test_the_problem_is_checked_as_for_a_built_in_algorithm(self):
        with self.assertRaises(cora.errors.WrongValue):
            self.sys.reach(self.X0, t_final=-1.0, alg="euler_box")
        with self.assertRaises(cora.errors.DimensionMismatch):
            self.sys.reach(cora.Zonotope(np.zeros(3), np.eye(3)), t_final=1.0, alg="euler_box")

    def test_an_unknown_algorithm_names_the_registered_ones(self):
        with self.assertRaisesRegex(ValueError, "euler_box"):
            self.sys.reach(self.X0, t_final=1.0, alg="euler")


class ErrorsAndSettings(unittest.TestCase):
    def test_every_error_class_has_its_identifier(self):
        self.assertTrue(issubclass(cora.errors.WrongValue, cora.CORAError))
        self.assertTrue(issubclass(cora.errors.WrongValue, ValueError))
        self.assertEqual(cora.errors.WrongValue.identifier, "CORA:wrongValue")

    def test_an_option_of_another_algorithm_is_named_so(self):
        sys = cora.LinearSys(np.array([[-1.0]]))
        X0 = cora.Zonotope(np.array([1.0]), np.array([[0.1]]))
        with self.assertRaisesRegex(TypeError, "not an option of the algorithm 'wrapping_free'"):
            sys.reach(X0, t_final=1.0, alg="wrapping_free", time_step=0.1, zonotope_order=5)

    def test_an_unknown_keyword_is_described(self):
        sys = cora.LinearSys(np.array([[-1.0]]))
        X0 = cora.Zonotope(np.array([1.0]), np.array([[0.1]]))
        with self.assertRaisesRegex(TypeError, "did you mean 'time_step'"):
            sys.reach(X0, t_final=1.0, time_stp=0.1)

    def test_warnings_reach_python(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cora._cora._warn("CORA:deprecated", "a test warning")
        self.assertTrue(any(issubclass(w.category, cora.CORAWarning) for w in caught))

    def test_scopes_nest_and_restore(self):
        default = cora.current_backend()
        with cora.backend("eigen"):
            self.assertEqual(cora.current_backend(), "eigen")
        self.assertEqual(cora.current_backend(), default)
        self.assertTrue(cora.checks_enabled())
        with cora.checks(False):
            self.assertFalse(cora.checks_enabled())
            with cora.checks(True):
                self.assertTrue(cora.checks_enabled())
            self.assertFalse(cora.checks_enabled())
        self.assertTrue(cora.checks_enabled())


if __name__ == "__main__":
    unittest.main()
