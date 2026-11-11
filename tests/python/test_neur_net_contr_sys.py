"""test_neur_net_contr_sys - NeurNetContrSys and the activations of NeuralNetwork in Python.

CORA authors: Niklas Kochdumper, Tobias Ladner
Authors:       Claude Sonnet 5.5
Written:       11-November-2026
Last update:   ---
Reviewed:      ---
"""
# CORA: contDynamics/@neurNetContrSys/neurNetContrSys.m
# CORA: contDynamics/@neurNetContrSys/reach.m
# CORA: contDynamics/@neurNetContrSys/simulate.m
# CORA: contDynamics/@neurNetContrSys/verify.m

import unittest

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

def plant():
    """x' = (x2 + u_ext, (1 - x1^2) x2 - x1 + u_nn): the network drives the second input."""
    return cora.NonlinearSys(lambda x, u: [x[1] + u[1], (1 - x[0] ** 2) * x[1] - x[0] + u[0]], 2, 2)


def network():
    layers = [
        (np.array([[0.8, -0.5], [0.3, 0.9], [-0.6, 0.4]]), np.array([0.1, -0.2, 0.0])),
        (np.array([[0.7, -0.4, 0.5]]), np.array([0.05])),
    ]
    return cora.NeuralNetwork(layers, ["tanh", "tanh"])


def start():
    return cora.Zonotope(np.array([1.4, 2.3]), np.diag([0.05, 0.04]))


def disturbance():
    return cora.Zonotope(np.array([0.0]), np.array([[0.01]]))


# ============================================  MAIN  ============================================ #

class Activations(unittest.TestCase):
    def test_points_follow_the_activations(self):
        # one neuron with the weight 1
        nn = cora.NeuralNetwork([(np.array([[1.0]]), np.array([0.0]))], ["sigmoid"])
        self.assertAlmostEqual(float(nn.evaluate(np.array([0.5]))[0]), 1 / (1 + np.exp(-0.5)), places=12)
        # and the same for tanh
        nn = cora.NeuralNetwork([(np.array([[1.0]]), np.array([0.0]))], ["tanh"])
        self.assertAlmostEqual(float(nn.evaluate(np.array([0.5]))[0]), np.tanh(0.5), places=12)

    def test_an_unknown_activation_is_refused(self):
        with self.assertRaises(cora.errors.WrongValue):
            cora.NeuralNetwork([(np.array([[1.0]]), np.array([0.0]))], ["swish"])

    def test_the_pairs_of_a_set_have_the_dimension_of_input_and_output(self):
        # two inputs and one output
        pairs = network().evaluate_with_input(start())
        self.assertEqual(pairs.dim, 3)


class ClosedLoop(unittest.TestCase):
    def test_the_sets_hold_a_run(self):
        sys = cora.NeurNetContrSys(plant(), network(), 0.05)
        R = sys.reach(start(), 0.5, time_step=0.025, input_set=disturbance(), zonotope_order=20)
        self.assertTrue(R["completed"])
        times, x = sys.simulate(np.array([[1.4], [2.3]]), 0.5, 0.025, u=np.array([[0.005]]))
        # the run lies in the box of the set at every sample
        for k in range(1, len(times)):
            box = R["time_point"][k].interval()
            lo, hi = np.asarray(box.inf), np.asarray(box.sup)
            self.assertTrue(np.all(x[k, :, 0] >= lo - 1e-9) and np.all(x[k, :, 0] <= hi + 1e-9))

    def test_the_polynomial_zonotope_start_completes(self):
        sys = cora.NeurNetContrSys(plant(), network(), 0.05)
        R = sys.reach(cora.PolyZonotope(start()), 0.5, time_step=0.025, input_set=disturbance())
        self.assertTrue(R["completed"] and len(R["time_interval"]) == 20)

    def test_verify_decides(self):
        sys = cora.NeurNetContrSys(plant(), network(), 0.05)
        rng = cora.Rng(1)
        loose = cora.Specification.safe_set(cora.Polytope(np.array([[1.0, 0.0]]), np.array([10.0])))
        verdict = sys.verify(start(), 0.5, 10, rng, time_step=0.025, input_set=disturbance(), spec=loose)
        self.assertEqual(verdict, "VERIFIED")
        never = cora.Specification.safe_set(cora.Polytope(np.array([[1.0, 0.0]]), np.array([1.0])))
        verdict = sys.verify(start(), 0.5, 10, rng, time_step=0.025, input_set=disturbance(), spec=never)
        self.assertEqual(verdict, "FALSIFIED")

    def test_wrong_inputs_are_refused(self):
        sys = cora.NeurNetContrSys(plant(), network(), 0.05)
        with self.assertRaises(cora.errors.WrongValue):
            sys.reach(start(), 0.5, time_step=0.025)  # the external input set is missing


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
