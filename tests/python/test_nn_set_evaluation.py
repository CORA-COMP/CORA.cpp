"""test_nn_set_evaluation - the network on Taylor models and constrained zonotopes, and as a system.

CORA authors: Niklas Kochdumper, Tobias Ladner, Laura Luetzow
Authors:       Claude Sonnet 5.5
Written:       23-November-2026
Last update:   ---
Reviewed:      ---
"""
# CORA: nn/layers/nonlinear/@nnActivationLayer/evaluateTaylm.m
# CORA: nn/layers/nonlinear/@nnActivationLayer/evaluateConZonotope.m
# CORA: nn/@neuralNetwork/nonlinearSysDT.m
# CORA: nn/@neuralNetwork/explain.m
# CORA: nn/@neuralNetwork/computeFGSMAttack.m

import unittest

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

def network(activation):
    """Two inputs, a layer of three, an output of two."""
    return cora.NeuralNetwork(
        [
            ([[1.0, -2.0], [0.5, 1.5], [-1.0, 1.0]], [0.1, -0.2, 0.3]),
            ([[1.0, 1.0, -1.0], [0.5, -1.0, 2.0]], [0.0, 0.2]),
        ],
        [activation, "identity"],
    )


# the three operations
class SetEvaluation(unittest.TestCase):
    def test_taylm_contains_the_outputs(self):
        # the box of the inputs as models, and the models of the outputs
        nn = network("tanh")
        x = cora.Taylm.from_interval(cora.Interval(np.array([-1.0, -1.0]), np.array([1.0, 1.0])))
        y = nn.evaluate_taylm(x, poly_method="regression", orders=[[2], None])
        for a in np.linspace(-1, 1, 5):
            for b in np.linspace(-1, 1, 5):
                out = nn.evaluate(np.array([a, b]))
                for k in range(2):
                    lo, hi = y[k].bounds()
                    self.assertTrue(lo - 1e-12 <= out[k] <= hi + 1e-12)

    # the outputs of points of a constrained set lie in the output set
    def test_con_zonotope_contains_the_outputs(self):
        nn = network("relu")
        c = np.array([0.2, -0.1])
        G = np.array([[1.0, 0.5], [-0.5, 1.0], [0.3, 0.2]])
        X = cora.ConZonotope(c, G, np.array([[1.0, 1.0, 0.0]]), np.array([0.5]))
        for approx in (True, False):
            Y = nn.evaluate(X, bound_approx=approx)
            for x1 in np.linspace(-0.5, 1.0, 7):
                p = c + np.array([x1, 0.5 - x1, 0.4]) @ G
                self.assertTrue(Y.contains(nn.evaluate(p)))

    # the system holds the network and its Jacobians
    def test_system(self):
        nn = network("tanh")
        sys = nn.nonlinear_sys_dt()
        x = np.array([0.3, -0.4])
        self.assertEqual((sys.num_states, sys.num_inputs, sys.num_outputs), (2, 5, 2))
        np.testing.assert_allclose(sys.output(x, np.zeros(5)), nn.evaluate(x), atol=1e-12)
        np.testing.assert_array_equal(sys.next_state(x, np.zeros(5)), x)
        np.testing.assert_allclose(sys.output_jacobian_x(x), nn.sensitivity(x))
        self.assertEqual(np.asarray(sys.output_jacobian_u(x, np.zeros(5))).shape, (2, 5))

    # the explanation of a linear classifier, and the attack that moves each input by epsilon
    def test_explain(self):
        nn = cora.NeuralNetwork(
            [([[3.0, 0.5, 0.1], [0.0, 0.0, 0.0]], [0.0, 0.0])], ["identity"]
        )
        x = np.array([1.0, 1.0, 1.0])
        freed, order, times = nn.explain(x, 0, 1.2)
        self.assertEqual((list(freed), list(order)), ([2, 1], [2, 1, 0]))
        self.assertEqual(len(times), 3)
        freed, _, _ = nn.explain(x, 0, 1.2, method="abstract+refine", refine_steps=[1.0])
        self.assertEqual(list(freed), [2, 1])
        z = nn.compute_fgsm_attack([[1.0, 1.0, 1.0]], [[1.0, 0.0]], 0.25, mask=[1.0, 0.0, 1.0])
        np.testing.assert_allclose(z, [[0.75, 1.0, 0.75]])
        with self.assertRaisesRegex(Exception, "class below"):
            nn.explain(x, 5, 1.2)


if __name__ == "__main__":
    unittest.main()


# ----------------------------------------  END OF CODE  ----------------------------------------- #
