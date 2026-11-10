"""test_hybrid_automaton - the hybrid automaton classes in Python: a bouncing ball through the bindings.

CORA authors: Mark Wetzlinger, Matthias Althoff, Niklas Kochdumper, Stefan Liu, Viktor Kotsev
Authors:       Claude Opus 5.5
Written:       09-November-2026
Last update:   ---
Reviewed:      ---
"""
# CORA: hybridDynamics/@linearReset/linearReset.m
# CORA: hybridDynamics/@linearReset/evaluate.m
# CORA: hybridDynamics/@transition/transition.m
# CORA: hybridDynamics/@location/location.m
# CORA: hybridDynamics/@hybridAutomaton/hybridAutomaton.m
# CORA: hybridDynamics/@hybridAutomaton/reach.m
# CORA: hybridDynamics/@location/reach.m
# CORA: hybridDynamics/@location/potInt.m
# CORA: hybridDynamics/@location/guardIntersect.m
# CORA: hybridDynamics/@location/guardIntersect_zonoGirard.m
# CORA: hybridDynamics/@location/guardIntersect_nondetGuard.m
# CORA: hybridDynamics/@location/guardIntersect_conZonotope.m
# CORA: hybridDynamics/@location/calcBasis.m
# CORA: hybridDynamics/@location/checkFlow.m
# CORA: contSet/@polytope/and_.m

import math
import unittest

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

G = 9.81


def ball():
    """x' = v, v' = -g on x >= 0; at x = 0 with v <= 0 the ball bounces, v := -0.75 v."""
    floor = cora.Polytope(np.array([[-1.0, 0.0]]), np.array([0.0]))
    guard = cora.Polytope(np.array([[0.0, 1.0]]), np.array([0.0]), np.array([[1.0, 0.0]]), np.array([0.0]))
    bounce = cora.LinearReset(np.array([[1.0, 0.0], [0.0, -0.75]]))
    flow = cora.LinearSys(np.array([[0.0, 1.0], [0.0, 0.0]]), np.zeros((2, 1)))
    fall = cora.Location("fall", floor, [cora.Transition(guard, bounce, 0)], flow, np.array([0.0, -G]))
    return cora.HybridAutomaton("ball", [fall])


DROP = cora.Interval(np.array([1.0, 0.0]), np.array([1.2, 0.0])).zonotope()


# ============================================  MAIN  ============================================ #

class HybridAutomaton(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("eigen"))

    def test_reset_maps_a_box_and_points(self):
        reset = cora.LinearReset(np.array([[1.0, 0.0], [0.0, -0.5]]), c=np.array([1.0, 0.0]))
        box = cora.Interval(np.array([0.0, 2.0]), np.array([1.0, 4.0])).zonotope()
        image = reset.evaluate(box).interval()
        np.testing.assert_allclose(image.inf, [1.0, -2.0])
        np.testing.assert_allclose(image.sup, [2.0, -1.0])
        np.testing.assert_allclose(reset.evaluate(np.array([[0.0], [2.0]])), [[1.0], [-1.0]])

    def test_the_ball_bounces_with_every_guard_intersection(self):
        HA = ball()
        # each method lets the ball bounce, the first jump between the fastest and slowest drop
        for method, enclose, order in [("zono_girard", ["box"], 0), ("nondet_guard", ["pca"], 0),
                                       ("con_zonotope", ["box", "pca"], 5)]:
            R = HA.reach(DROP, t_final=1.5, time_step=0.01, taylor_terms=6, zonotope_order=20,
                         guard_intersect=method, enclose=enclose, guard_order=order)
            self.assertTrue(R.verified)
            self.assertGreaterEqual(len(R.branches), 3, method)
            lo, hi = R.branches[1].time_point_times[0]
            self.assertLessEqual(lo, math.sqrt(2 / G))
            self.assertGreaterEqual(hi, math.sqrt(2 * 1.2 / G))

    def test_the_cuts_are_boxed_by_passes_or_by_programs(self):
        HA = ball()
        # a cut is boxed by Fourier-Motzkin passes (fm_passes) or, as CORA, by linear programs
        for options in [{"fm_passes": 1}, {"fm_passes": 3}, {"exact_cut": True}]:
            R = HA.reach(DROP, t_final=1.5, time_step=0.01, taylor_terms=6, zonotope_order=20,
                         guard_intersect="nondet_guard", enclose=["pca"], **options)
            self.assertTrue(R.verified)
            self.assertGreaterEqual(len(R.branches), 3, options)

    def test_specifications_stop_reach(self):
        low = cora.Specification.safe_set(cora.Polytope(np.array([[1.0, 0.0]]), np.array([1.1])))
        R = ball().reach(DROP, t_final=1.5, time_step=0.01, specs=[(low, [0])])
        self.assertFalse(R.verified)
        # a custom predicate is asked with the sets themselves
        seen = []
        custom = cora.Specification.custom(lambda S: seen.append(S.dim) or True)
        self.assertEqual(custom.type, "custom")
        self.assertTrue(ball().reach(DROP, t_final=0.2, time_step=0.01, specs=[custom]).verified)
        self.assertEqual(set(seen), {2})

    def test_wrong_input_is_refused(self):
        with self.assertRaises(cora.errors.WrongValue):
            ball().reach(DROP, t_final=1.0, time_step=[0.1, 0.1])
        with self.assertRaises(cora.errors.WrongValue):
            ball().reach(DROP, t_final=1.0, time_step=0.1, guard_intersect="levelSet")


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
