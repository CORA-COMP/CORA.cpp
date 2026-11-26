"""test_joint_network - several networks as one joint network on a box of new inputs

The network field of VNN-COMP 2026 (a literal list of pairs, or one path). InputMap: equalities of
inputs (isomorphic_acasxu's X_g == X_f) leave the inputs of f, exactly; an inequality
(monotonic_acasxu's X_g[0] <= X_f[0], X_g[0] >= lo) becomes a new input of the slack and a clamp,
and every point of the box maps to feasible inputs while every feasible input has a point (the
box covers the feasible set); an infeasible or unbounded set fails. joint_layers: the joint
network equals the networks side by side on the mapped inputs, with and without the clamp layer;
networks of different depths are not supported.

Authors:       Claude Opus 5.5
Written:       25-November-2026
Last update:   ---
Reviewed:      ---
"""

import unittest


try:
    import numpy as np
except ImportError:
    raise unittest.SkipTest("needs numpy") from None

from cora.joint_network import InputMap, joint_layers, parse_network_field


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

class NetworkField(unittest.TestCase):
    def test_pairs_and_path(self):
        field = "[('f', 'onnx/a.onnx'), ('g', 'onnx/b.onnx')]"
        self.assertEqual(parse_network_field(field), [("f", "onnx/a.onnx"), ("g", "onnx/b.onnx")])
        self.assertEqual(parse_network_field("onnx/a.onnx"), [(None, "onnx/a.onnx")])
        with self.assertRaises(ValueError):
            parse_network_field("[('f',)]")
        with self.assertRaises(ValueError):
            parse_network_field("[1 +")


class Inputs(unittest.TestCase):
    def test_equalities(self):
        # X_g[i] == X_f[i] for 3 inputs: the box of X_f, X_g a copy
        lo, hi = np.array([0.0, -1, 2] + [-np.inf] * 3), np.array([1.0, 1, 2] + [np.inf] * 3)
        imap = InputMap(lo, hi, EQUAL3)
        np.testing.assert_array_equal(imap.lo, [0, -1, 2])
        np.testing.assert_array_equal(imap.hi, [1, 1, 2])
        self.assertIsNone(imap.W0)
        w = np.array([[0.25, -0.5, 2.0]])
        np.testing.assert_array_equal(imap.inputs(w), [[0.25, -0.5, 2, 0.25, -0.5, 2]])
        self.assertTrue(imap.feasible(imap.inputs(w)[0]))
        self.assertFalse(imap.feasible([0.25, -0.5, 2, 0.3, -0.5, 2]))

    def test_inequality_and_clamp(self):
        # x0 in [-1, 1], x1 = x2 <= x0, x1 >= -0.5: the slack x1 - x0 and a clamp at -0.5
        lo, hi = np.array([-1.0, -0.5, -np.inf]), np.array([1.0, np.inf, np.inf])
        relations = [({1: 1.0, 0: -1.0}, 0.0), ({1: 1.0, 2: -1.0}, 0.0), ({1: -1.0, 2: 1.0}, 0.0)]
        imap = InputMap(lo, hi, relations)
        self.assertEqual(imap.dim, 2)
        self.assertIsNotNone(imap.W0)
        rng = np.random.default_rng(0)
        w = imap.lo + rng.random((500, imap.dim)) * (imap.hi - imap.lo)
        corners = np.array([[a, b] for a in (imap.lo[0], imap.hi[0]) for b in (imap.lo[1], imap.hi[1])])
        for v in imap.inputs(np.vstack([w, corners])):
            self.assertTrue(imap.feasible(v), v)
        # every feasible input has a point in the box: its slack and its x0
        x0 = rng.uniform(-0.5, 1.0, 300)
        x1 = -0.5 + rng.random(300) * (x0 + 0.5)
        z = np.stack([x0, x1], 1)
        point = z @ np.linalg.inv(imap.Minv).T
        self.assertTrue(np.all(point >= imap.lo - 1e-12) and np.all(point <= imap.hi + 1e-12))
        np.testing.assert_allclose(imap.inputs(point), np.stack([x0, x1, x1], 1), atol=1e-12)

    def test_errors(self):
        with self.assertRaisesRegex(ValueError, "unbounded"):
            InputMap(np.array([0.0, -np.inf]), np.array([1.0, np.inf]), [({0: 1.0, 1: -1.0}, 0.0)])
        with self.assertRaisesRegex(ValueError, "infeasible|empty"):
            InputMap(np.array([0.0, 2.0]), np.array([1.0, 3.0]), [({1: 1.0, 0: -1.0}, 0.0)])


class Layers(unittest.TestCase):
    def test_side_by_side(self):
        # two ReLU networks of 3 inputs on the mapped inputs of an equality and of an inequality
        rng = np.random.default_rng(1)
        nets = [random_net(rng, [3, 8, 6, 2]) for _ in range(2)]
        # X_g == X_f; X_g[0] <= X_f[0] (a clamp at -0.5) with X_g[1:] == X_f[1:]
        cases = [
            (np.array([0.0, -1, 2] + [-np.inf] * 3), np.array([1.0, 1, 2] + [np.inf] * 3), EQUAL3),
            (np.array([-1.0, -1, 0, -0.5, -np.inf, -np.inf]), np.array([1.0, 1, 1] + [np.inf] * 3),
             [({3: 1.0, 0: -1.0}, 0.0)] + EQUAL3[2:]),
        ]
        for lo, hi, relations in cases:
            # the joint network on points of the box against the two on the mapped inputs
            imap = InputMap(lo, hi, relations)
            layers, acts = joint_layers(imap, nets, [slice(0, 3), slice(3, 6)])
            w = imap.lo + rng.random((50, imap.dim)) * (imap.hi - imap.lo)
            v = imap.inputs(w)
            expected = np.hstack([forward(nets[0], v[:, :3]), forward(nets[1], v[:, 3:])])
            np.testing.assert_allclose(forward((layers, acts), w), expected, atol=1e-12)

    def test_depths_differ(self):
        rng = np.random.default_rng(2)
        imap = InputMap(np.array([0.0, -np.inf]), np.array([1.0, np.inf]),
                        [({0: 1.0, 1: -1.0}, 0.0), ({0: -1.0, 1: 1.0}, 0.0)])
        with self.assertRaises(NotImplementedError):
            joint_layers(imap, [random_net(rng, [1, 4, 1]), random_net(rng, [1, 4, 4, 1])],
                         [slice(0, 1), slice(1, 2)])


# -----------------------------------------  AUXILIARY  ------------------------------------------ #

EQUAL3 = [r for i in range(3) for r in (({i: 1.0, i + 3: -1.0}, 0.0), ({i: -1.0, i + 3: 1.0}, 0.0))]


def random_net(rng, sizes):
    """A ReLU network ([(W, b)], activations) of the layer sizes, the last layer the identity."""
    layers = [(rng.standard_normal((b, a)), rng.standard_normal(b)) for a, b in
              zip(sizes[:-1], sizes[1:], strict=True)]
    return layers, ["relu"] * (len(layers) - 1) + ["identity"]


def forward(net, x):
    """The network on the rows x in float64."""
    for (W, b), act in zip(*net, strict=True):
        x = x @ np.asarray(W).T + b
        x = np.maximum(x, 0.0) if act == "relu" else x
    return x

# ----------------------------------------  END OF CODE  ----------------------------------------- #
