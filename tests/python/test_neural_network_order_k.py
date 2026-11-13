"""test_neural_network_order_k - polynomials of order k on polynomial zonotopes, and refine, in Python.

A neuron of order k is enclosed by a polynomial of that order and an error, evaluated on the
powers of its set. The tests check a hand-computed case, that order 1 through the neuron-wise
path is the vectorised line, that the sets contain the outputs of sampled points for every
poly_method, that a limit on the generators keeps the enclosure, and that refine raises the orders
its heuristic names.

CORA authors: Tobias Ladner, Niklas Kochdumper
Authors:       Claude Sonnet 5.5
Written:       13-November-2026
Last update:   ---
Reviewed:      ---
"""
# CORA: nn/layers/nonlinear/@nnActivationLayer/evaluatePolyZonotopeNeuron.m
# CORA: nn/layers/nonlinear/@nnActivationLayer/computeApproxPoly.m
# CORA: nn/layers/nonlinear/@nnActivationLayer/nnActivationLayer.m
# CORA: nn/@neuralNetwork/refine.m
# CORA: nn/@neuralNetwork/getRefinableLayers.m

import unittest

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #


def neuron(activation):
    """A network of one neuron a(x)."""
    return cora.NeuralNetwork([(np.array([[1.0]]), np.array([0.0]))], [activation])


def interval_set(lo, hi):
    """x = c + g alpha over alpha in [-1, 1]."""
    return cora.PolyZonotope(np.array([(lo + hi) / 2]), np.array([[(hi - lo) / 2]]), None,
                             np.array([[1]]), [1])


def outside(net, X, Y, n=300):
    """How many outputs of sampled points of X leave the box of Y."""
    box = Y.zonotope().interval()
    y = net.evaluate(np.asarray(X.rand_point(n, cora.Rng(3))))
    return int(((y < np.asarray(box.inf)[:, None] - 1e-9) | (y > np.asarray(box.sup)[:, None] + 1e-9)).sum())


# ============================================  MAIN  ============================================ #

class OrderK(unittest.TestCase):
    def test_a_relu_of_order_2_is_the_hand_computed_polynomial(self):
        # 3/16 (x + 1)^2 + 7/96 +- 25/96 over [-1, 3] by Singh's polynomial
        Y = neuron("relu").evaluate(interval_set(-1.0, 3.0), poly_method="singh", orders=[2])
        # the centre, the exponents, the generators and the error
        np.testing.assert_allclose(Y.c, [79.0 / 96.0])
        by_exponent = {int(e): g for e, g in zip(Y.E[0], Y.G[0], strict=True)}
        self.assertEqual(sorted(by_exponent), [1, 2])
        np.testing.assert_allclose([by_exponent[1], by_exponent[2]], [1.5, 0.75])
        np.testing.assert_allclose(Y.GI, [[25.0 / 96.0]])

    def test_order_1_is_the_line_bit_for_bit(self):
        # a network of two layers on a zonotope as polynomial zonotope
        rng = np.random.default_rng(1)
        net = cora.NeuralNetwork([(rng.normal(size=(4, 2)), rng.normal(size=4)),
                                  (rng.normal(size=(1, 4)), rng.normal(size=1))],
                                 ["tanh", "sigmoid"])
        X = cora.PolyZonotope(cora.Zonotope(np.array([0.1, -0.2]), 0.3 * np.eye(2)))
        for in_gi in (True, False):
            line = net.evaluate(X, add_approx_error_to_gi=in_gi)
            poly = net.evaluate(X, add_approx_error_to_gi=in_gi, num_generators=1000000)
            np.testing.assert_array_equal(line.c, poly.c)
            # the order of the generators is the constructor's: compare the columns sorted
            for a, b in ((line.GI, poly.GI), (np.vstack([line.G, line.E]), np.vstack([poly.G, poly.E]))):
                np.testing.assert_array_equal(a[:, np.lexsort(a[::-1])], b[:, np.lexsort(b[::-1])])

    def test_every_method_encloses_the_activation(self):
        for activation in ("sigmoid", "tanh"):
            for method in ("regression", "ridgeregression", "bernstein", "taylor", "throw-catch"):
                for order in (1, 2, 3):
                    X = interval_set(-1.5, 2.0)
                    Y = neuron(activation).evaluate(X, poly_method=method, orders=[order])
                    self.assertEqual(outside(neuron(activation), X, Y), 0, (activation, method, order))

    def test_sets_with_even_exponents_and_independent_generators_hold(self):
        # x = 0.2 + a^2 + 0.5 a b - 0.4 b^3 + 0.8 t: CORA's square of such a set misses its top
        X = cora.PolyZonotope(np.array([0.2]), np.array([[1.0, 0.5, -0.4]]), np.array([[0.8]]),
                              np.array([[2, 1, 0], [0, 1, 3]]), [1, 2])
        # every activation at the orders 2 to 4
        for activation in ("sigmoid", "tanh", "relu"):
            for order in (2, 3, 4):
                net = neuron(activation)
                Y = net.evaluate(X, orders=[order], poly_method="regression")
                self.assertEqual(outside(net, X, Y), 0)

    def test_a_limit_on_the_generators_keeps_the_enclosure(self):
        # a network of two layers on a set with one independent generator
        rng = np.random.default_rng(2)
        net = cora.NeuralNetwork([(rng.normal(size=(5, 3)), rng.normal(size=5)),
                                  (rng.normal(size=(2, 5)), rng.normal(size=2))],
                                 ["tanh", "identity"])
        X = cora.PolyZonotope(np.zeros(3), 0.3 * np.eye(3), 0.05 * np.ones((3, 1)),
                              np.eye(3, dtype=int), [1, 2, 3])
        # the same order with and without the limit
        full = net.evaluate(X, orders=[2], poly_method="regression")
        cut = net.evaluate(X, orders=[2], num_generators=6, poly_method="regression")
        self.assertEqual(outside(net, X, cut), 0)
        self.assertLess(cut.G.shape[1] + cut.GI.shape[1], full.G.shape[1] + full.GI.shape[1])


class Refine(unittest.TestCase):
    def network(self):
        return cora.NeuralNetwork(
            [(np.array([[1.0, -1.0], [2.0, 1.0]]), np.array([0.0, 0.1])),
             (np.array([[1.0, 0.5], [-0.5, 1.0]]), np.array([0.0, 0.0])),
             (np.array([[1.0, 1.0]]), np.array([0.0]))], ["relu", "tanh", "identity"])

    def test_refinable_layers_and_errors(self):
        net = self.network()
        self.assertEqual(list(net.refinable_layers()), [0, 1])
        X = cora.PolyZonotope(np.zeros(2), np.eye(2), None, np.eye(2, dtype=int), [1, 2])
        errors = net.approx_errors(X)
        self.assertEqual([len(e) for e in errors], [2, 2, 0])

    def test_refine_raises_what_the_heuristic_names(self):
        net = self.network()
        # layer: the layer of the largest norm
        orders = net.refine([], [[1.0, 0.1], [0.5, 0.4], []])
        self.assertEqual([list(o) for o in orders], [[2, 2], [1, 1], [1]])
        # neuron: the neurons above gamma times the largest, in its layer
        orders = net.refine([], [[1.0, 0.05], [0.5, 0.4], []], type="neuron")
        self.assertEqual([list(o) for o in orders], [[2, 1], [1, 1], [1]])
        # all stops at max_order
        orders = net.refine([], [[1.0, 1.0], [1.0, 1.0], []], type="all", max_order=2)
        orders = net.refine(orders, [[1.0, 1.0], [1.0, 1.0], []], type="all", max_order=2)
        self.assertEqual([list(o) for o in orders], [[2, 2], [2, 2], [1]])

    def test_wrong_arguments_are_described(self):
        net = neuron("sigmoid")
        with self.assertRaises(ValueError):
            net.evaluate(interval_set(-1.0, 1.0), orders=[0])
        with self.assertRaises(ValueError):
            net.evaluate(interval_set(-1.0, 1.0), poly_method="foo")
        with self.assertRaises(cora.CORAError):
            net.evaluate(interval_set(-1.0, 1.0), orders=[2])
        with self.assertRaises(ValueError):
            net.refine([], [[1.0]], type="foo")


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
