"""test_hybrid_level_set - a location with a level set invariant and guard through the bindings.

CORA authors: Niklas Kochdumper, Mark Wetzlinger
Authors:       Claude Sonnet 5.5
Written:       11-November-2026
Last update:   ---
Reviewed:      ---
"""
# CORA: hybridDynamics/@location/location.m
# CORA: hybridDynamics/@transition/transition.m
# CORA: hybridDynamics/@location/reach.m

import unittest

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

def disc():
    """The unit disc x^2 + y^2 <= 1 as a level set."""
    x, y = cora.Expr.var(0), cora.Expr.var(1)
    return cora.LevelSet(x * x + y * y - 1.0, 2, "<=")


def particle():
    """A particle that moves right at speed 1 inside the disc, with a circle as a guard."""
    x, y = cora.Expr.var(0), cora.Expr.var(1)
    circle = cora.LevelSet(x * x + y * y - 1.0, 2, "==")
    flow = cora.NonlinearSys(lambda s: [cora.Expr(1.0), cora.Expr(0.0)], 2)
    jump = cora.Transition(circle, cora.LinearReset(np.eye(2)), 0)
    return cora.Location("fly", disc(), [jump], flow)


# ============================================  MAIN  ============================================ #

class LevelSetLocation(unittest.TestCase):
    # the invariant and the guard are level sets
    def test_parts(self):
        loc = particle()
        self.assertEqual(loc.dim, 2)
        self.assertEqual(loc.invariant.class_name, "LevelSet")
        self.assertEqual(loc.transitions[0].guard.class_name, "LevelSet")

    # the reach stops where the sets have left the disc
    def test_invariant_ends_reach(self):
        flow = cora.NonlinearSys(lambda s: [cora.Expr(1.0), cora.Expr(0.0)], 2)
        HA = cora.HybridAutomaton("disc", [cora.Location("fly", disc(), [], flow)])
        X0 = cora.Zonotope(np.array([-0.5, 0.0]), 0.05 * np.eye(2))
        R = HA.reach(X0, t_final=3.0, time_step=0.1)
        steps = len(R.branches[0].time_interval)
        self.assertTrue(15 <= steps <= 19)

    # a level set guard that is reached cannot be enclosed yet
    def test_guard_not_ported(self):
        HA = cora.HybridAutomaton("disc", [particle()])
        X0 = cora.Zonotope(np.array([-0.5, 0.0]), 0.05 * np.eye(2))
        with self.assertRaises(cora.errors.NotSupported):
            HA.reach(X0, t_final=3.0, time_step=0.1)


if __name__ == "__main__":
    unittest.main()


# ----------------------------------------  END OF CODE  ----------------------------------------- #
