"""test_onnx_network - ONNX networks on torch against onnxruntime, and their cora.NeuralNetwork

Small graphs of every operator the loader knows, built with onnx.helper: the batched torch run
equals onnxruntime's (float32) to float32 accuracy; a chain of affine maps and activations lowers
to dense layers whose network equals the graph in float64, and a plain Gemm keeps its weights bit
for bit; a residual connection and a ResNet block lower to layers with residuals (the batch norms
folded into the convolutions) and max pools to pairs of b + relu(a - b), equal to the graph; a
padded max pool and a quotient that broadcasts its dividend do not lower and say why; a batch norm
of features and a slice of the input concatenated with a branch lower; a product of two values is a
product layer, and a ReduceSum, a difference of branches and scalings lower; a Gemm or a MatMul
of a row and a constant matrix takes the weights themselves, far below the identity's memory; the
verifier's backend is torch for a convolution, eigen for a dense network; a binarized network
lowers to sign layers, its output Softmax dropped for a property that only compares outputs;
vit_2023's, nn4sys' and ml4acopf's shapes (quotients, sin, cos) lower and equal the graph;
ReluSplitter's pairs relu(t), relu(-t) merge into the next layer (the base network's weights), a
layer with an unpaired neuron stays.

Authors:       Claude Opus 5.5
Written:       21-November-2026
Last update:   24-November-2026
Reviewed:      ---
"""

import os
import subprocess
import sys
import tempfile
import unittest


try:
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from onnx import TensorProto, helper, numpy_helper
except ImportError:  # the VNN-COMP front end needs onnx, onnxruntime and torch
    raise unittest.SkipTest("needs onnx, onnxruntime and torch") from None

import cora


if not cora.has_torch:
    raise unittest.SkipTest("needs a build with torch") from None

from cora.onnx_network import OnnxNetwork, _constant_matrix


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

class TorchRun(unittest.TestCase):
    def test_every_operator_agrees_with_onnxruntime(self):
        # each graph on 7 rows at once, against onnxruntime one sample at a time
        for m in (mlp(), conv_net(), residual_net(), resnet_block(), maxpool_net(), lyapunov_net()):
            net = OnnxNetwork(m, dtype=torch.float32)
            x = RNG.uniform(-1, 1, (7, net.num_inputs)).astype(np.float32)
            expected = onnxruntime_rows(m, x)
            np.testing.assert_allclose(net(torch.from_numpy(x)).numpy(), expected, rtol=1e-5, atol=1e-5)
            # the same graph in float64 is as close
            y64 = net.cast(torch.float64)(torch.from_numpy(x).double()).numpy()
            np.testing.assert_allclose(y64, expected, rtol=1e-5, atol=1e-5)

    def test_gradients_flow(self):
        # autograd through the batched run, as PGD uses it
        net = OnnxNetwork(conv_net(), dtype=torch.float32)
        x = torch.zeros(3, net.num_inputs, requires_grad=True)
        (grad,) = torch.autograd.grad(net(x).sum(), x)
        self.assertEqual(tuple(grad.shape), (3, 16))
        self.assertTrue(bool(torch.isfinite(grad).all()))


class Lowering(unittest.TestCase):
    def test_chains_lower_to_the_same_function(self):
        # the dense layers in float64 equal the graph in float64
        for m in (mlp(), conv_net()):
            net = OnnxNetwork(m, dtype=torch.float64)
            layers, why = net.dense_layers()
            self.assertIsNone(why)
            x = RNG.uniform(-1, 1, (9, net.num_inputs))
            np.testing.assert_allclose(
                dense_rows(layers, x), net(torch.from_numpy(x)).numpy(), rtol=1e-12, atol=1e-12
            )

    def test_activations_and_weights(self):
        # the chain of the MLP: relu, sigmoid, tanh, identity; the Gemm's weights exactly
        m = mlp()
        layers, _ = OnnxNetwork(m).dense_layers()
        self.assertEqual(layers[1], ["relu", "sigmoid", "tanh", "identity"])
        W2 = numpy_helper.to_array(m.graph.initializer[3])
        self.assertTrue(np.array_equal(layers[0][1][0].numpy(), W2.astype(np.float64)))

    def test_neural_network_evaluates_like_the_graph(self):
        # the cora.NeuralNetwork of the chain at points
        net = OnnxNetwork(mlp(), dtype=torch.float64)
        nn, why = net.to_neural_network()
        self.assertIsNone(why)
        x = torch.as_tensor(RNG.uniform(-1, 1, (5, 5)))
        np.testing.assert_allclose(np.asarray(nn.evaluate(x)), net(x).numpy(), rtol=1e-12, atol=1e-12)

    def test_what_does_not_lower(self):
        # a padded max pool and a quotient that broadcasts its dividend, each with the reason
        for m, reason in ((maxpool_net(pads=1), "MaxPool"),
                          (quotient_net(broadcast=True), "broadcasts the dividend")):
            nn, why = OnnxNetwork(m).to_neural_network()
            self.assertIsNone(nn)
            self.assertIn(reason, why)

    def test_convolutions_lower_layer_by_layer(self):
        # with a Conv the layers are conv2d and gather dicts, never a matrix of the convolution
        net = OnnxNetwork(conv_net(), dtype=torch.float64)
        lowered, why = net.layers()
        self.assertIsNone(why)
        kinds = [next(iter(layer)) if isinstance(layer, dict) else "dense" for layer in lowered[0]]
        self.assertIn("conv2d", kinds)
        self.assertIn("gather", kinds)
        x = RNG.uniform(-1, 1, (9, net.num_inputs))
        np.testing.assert_allclose(
            structured_rows(*lowered, x), net(torch.from_numpy(x)).numpy(), rtol=1e-12, atol=1e-12
        )

    def test_residual_graphs(self):
        # a dense residual and a ResNet block: layers with residuals, the network equals the graph
        for m, residuals in ((residual_net(), 1), (resnet_block(), 1)):
            net = OnnxNetwork(m, dtype=torch.float64)
            nn, why = net.to_neural_network()
            self.assertIsNone(why)
            (layers, _), _ = net.layers()
            self.assertEqual(sum(1 for la in layers if la.get("residuals")), residuals)
            x = torch.as_tensor(RNG.uniform(-1, 1, (6, net.num_inputs)))
            np.testing.assert_allclose(
                np.asarray(nn.evaluate(x)), net(x).numpy(), rtol=1e-12, atol=1e-12
            )
        # the block's batch norms are folded into its convolutions: conv, conv + shortcut, gemm
        (layers, acts), _ = OnnxNetwork(resnet_block()).layers()
        self.assertEqual(acts, ["relu", "relu", "identity"])
        self.assertTrue(all("conv2d" in la for la in layers[:2]))
        self.assertIn("conv2d", layers[1]["residuals"][0]["map"])

    def test_products_and_differences(self):
        # products, a quadratic form, differences, scalings: the network equals the graph
        for m in (product_net(), lyapunov_net()):
            net = OnnxNetwork(m, dtype=torch.float64)
            nn, why = net.to_neural_network()
            self.assertIsNone(why)
            x = torch.as_tensor(RNG.uniform(-1, 1, (6, net.num_inputs)))
            np.testing.assert_allclose(
                np.asarray(nn.evaluate(x)), net(x).numpy(), rtol=1e-12, atol=1e-12
            )
        (layers, acts), _ = OnnxNetwork(product_net()).layers()
        self.assertIn("product", acts)

    def test_features_and_branches(self):
        # a batch norm of features (CNN-7), and a slice of the input concatenated with a branch
        for m in (cnn7_net(), slice_concat_net()):
            net = OnnxNetwork(m, dtype=torch.float64)
            nn, why = net.to_neural_network()
            self.assertIsNone(why)
            x = torch.as_tensor(RNG.uniform(-1, 1, (6, net.num_inputs)))
            np.testing.assert_allclose(
                np.asarray(nn.evaluate(x)), net(x).numpy(), rtol=1e-12, atol=1e-12
            )
        # the batch norm folds into the Gemm before it: conv, gemm, gemm
        (layers, acts), _ = OnnxNetwork(cnn7_net()).layers()
        self.assertEqual(acts, ["relu", "relu", "identity"])

    def test_max_pools(self):
        # the pairwise max pools equal the graph (2 x 2; an odd 3 x 3 of stride 2, two channels)
        for m in (maxpool_net(), maxpool_net(kernel=3, stride=2, channels=2)):
            net = OnnxNetwork(m, dtype=torch.float64)
            nn, why = net.to_neural_network()
            self.assertIsNone(why)
            x = torch.as_tensor(RNG.uniform(-1, 1, (6, net.num_inputs)))
            np.testing.assert_allclose(
                np.asarray(nn.evaluate(x)), net(x).numpy(), rtol=1e-12, atol=1e-12
            )
        # verify on the pool's enclosure: a sampled violation is never unsat
        net = OnnxNetwork(maxpool_net(kernel=3, stride=2, channels=2), dtype=torch.float64)
        nn, _ = net.to_neural_network()
        x = RNG.uniform(-1, 1, (4000, net.num_inputs))
        y = net(torch.as_tensor(x)).numpy()
        A = np.zeros((1, 1, net.num_outputs))
        A[0, 0, 0], A[0, 0, 1] = 1.0, -1.0
        low = float((y @ A[0, 0]).min())
        lo, hi = -np.ones((1, net.num_inputs)), np.ones((1, net.num_inputs))
        out = nn.verify(lo, hi, [0], A, np.array([[low + 0.01]]), 20, cora.vnncomp_options("x"))
        self.assertEqual(out["status"], "sat")

    def test_wide_gemm_in_chunks(self):
        # a Gemm after a wide convolution: the weights bit for bit, far below the identity's 1.6 GB
        m = wide_conv_net()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "wide.onnx")
            onnx.save(m, path)
            script = (
                "import resource, sys; from cora.onnx_network import OnnxNetwork as N; "
                "net = N.load(sys.argv[1]); peak = lambda: resource.getrusage("
                "resource.RUSAGE_SELF).ru_maxrss / 1024; before = peak(); "
                "(layers, _), _ = net.layers(); import numpy as np; "
                "np.save(sys.argv[2], layers[-1]['W'].numpy()); print(peak() - before)"
            )
            out = os.path.join(tmp, "W.npy")
            env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path), OMP_NUM_THREADS="2")
            grown = subprocess.run([sys.executable, "-c", script, path, out], env=env,
                                   capture_output=True, text=True, check=True).stdout
            self.assertLess(float(grown), 400.0)  # MiB
            W = numpy_helper.to_array(m.graph.initializer[2])
            self.assertTrue(np.array_equal(np.load(out), W.astype(np.float64)))

    def test_constant_matrices_without_unit_vectors(self):
        # a Gemm (alpha, B as is) and a MatMul after a convolution: their weights, bit for bit
        m = conv_gemm_matmul_net()
        net = OnnxNetwork(m, dtype=torch.float64)
        (layers, acts), why = net.layers()
        self.assertIsNone(why)
        W1, W2 = (numpy_helper.to_array(m.graph.initializer[k]) for k in (2, 4))
        self.assertTrue(np.array_equal(layers[-2]["W"].numpy(), 0.5 * W1.astype(np.float64).T))
        self.assertTrue(np.array_equal(layers[-1]["W"].numpy(), W2.astype(np.float64).T))
        x = torch.as_tensor(RNG.uniform(-1, 1, (4, net.num_inputs)))
        np.testing.assert_allclose(np.asarray(cora.NeuralNetwork(layers, acts).evaluate(x)),
                                   net(x).numpy(), rtol=1e-12, atol=1e-12)
        # a transA, a non-finite weight and a batch of rows keep the unit vectors
        B = torch.ones(3, 2, dtype=torch.float64)
        self.assertIsNone(_constant_matrix("Gemm", [None, B], {"transA": 1}, (1, 3)))
        self.assertIsNone(_constant_matrix("MatMul", [None, B * float("inf")], {}, (1, 3)))
        self.assertIsNone(_constant_matrix("MatMul", [None, B], {}, (1, 4, 3)))

    def test_split_fold_and_wide_scaling(self):
        # a Split of sizes as an input, then a product with a fixed input of 12000 entries
        n = 12000
        m = model(
            [helper.make_node("Split", ["x", "sizes"], ["a", "b"], axis=1),
             helper.make_node("Relu", ["a"], ["r"]),
             helper.make_node("Mul", ["r", "b"], ["y"])],
            [("x", [1, 2 * n])], [("y", [1, n])],
            {"sizes": np.array([n, n], dtype=np.int64)},
        )
        net = OnnxNetwork(m, dtype=torch.float64)
        self.assertEqual(net.op_types(), {"Slice", "Relu", "Mul"})
        (_, why) = net.layers()
        self.assertIn("of two values", why)
        # folded: a scaling, a gather with factors (no diagonal), equal to the graph
        lo, hi = -np.ones(2 * n), np.ones(2 * n)
        lo[n:] = hi[n:] = RNG.uniform(-2, 2, n)
        folded = net.fold_fixed_inputs(lo, hi)
        self.assertEqual(folded.op_types(), {"Slice", "Relu", "Mul"})
        (layers, acts), why = folded.layers()
        self.assertIsNone(why)
        self.assertTrue(any(isinstance(la, dict) and "scale" in la for la in layers))
        x = torch.as_tensor(np.concatenate([RNG.uniform(-1, 1, (3, n)), np.tile(lo[n:], (3, 1))], 1))
        np.testing.assert_allclose(np.asarray(cora.NeuralNetwork(layers, acts).evaluate(x)),
                                   net(x).numpy(), rtol=1e-12, atol=1e-12)

    def test_relu_pairs_merge(self):
        # ReluSplitter's split layer (relu(t), relu(-t) read as c and -c): merged into the next
        # layer, the base network's weights bit for bit, equal to the graph and onnxruntime
        m, base = split_net()
        net = OnnxNetwork(m, dtype=torch.float64)
        (layers, acts), why = net.layers()
        self.assertIsNone(why)
        self.assertEqual(net.merged_relu_pairs, 3)
        self.assertEqual(acts, ["relu", "identity"])
        (base_layers, _), _ = OnnxNetwork(base, dtype=torch.float64).layers()
        self.assertTrue(torch.equal(layers[0][0], base_layers[0][0]))
        np.testing.assert_allclose(layers[0][1].numpy(), base_layers[0][1].numpy(), atol=1e-6)
        x = RNG.uniform(-3, 3, (200, 4))
        np.testing.assert_allclose(dense_rows((layers, acts), x),
                                   net(torch.as_tensor(x)).numpy(), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(dense_rows((layers, acts), x),
                                   onnxruntime_rows(m, x.astype(np.float32)), rtol=1e-5, atol=1e-5)
        # a pair read through columns that are not opposite, or a neuron without a partner: kept
        for broken in ("column", "row"):
            m, _ = split_net(broken)
            net = OnnxNetwork(m, dtype=torch.float64)
            (layers, acts), _ = net.layers()
            self.assertEqual(net.merged_relu_pairs, 0, broken)
            self.assertEqual(len(layers), 3, broken)
            np.testing.assert_allclose(dense_rows((layers, acts), x),
                                       net(torch.as_tensor(x)).numpy(), rtol=1e-12, atol=1e-12)

    def test_floor_and_pow(self):
        # ml4acopf's x - 2 pi floor(x / (2 pi)) and Pow(x, 2): floor and square layers
        tau = np.float32(2 * np.pi)
        nodes = [
            helper.make_node("Gemm", ["x", "W", "b"], ["g"], transB=1),
            helper.make_node("Div", ["g", "tau"], ["d"]),
            helper.make_node("Floor", ["d"], ["f"]),
            helper.make_node("Mul", ["f", "tau"], ["t"]),
            helper.make_node("Sub", ["g", "t"], ["w"]),
            helper.make_node("Pow", ["w", "two"], ["y"]),
        ]
        init = {"W": weight(3, 2) * 4, "b": weight(3), "tau": np.array(tau), "two": np.array(2.0, np.float32)}
        m = model(nodes, [("x", [1, 2])], [("y", [1, 3])], init)
        net = OnnxNetwork(m, dtype=torch.float64)
        (layers, acts), why = net.layers()
        self.assertIsNone(why)
        self.assertIn("floor", acts)
        self.assertIn("square", acts)
        x = torch.as_tensor(RNG.uniform(-3, 3, (20, 2)))
        np.testing.assert_allclose(np.asarray(cora.NeuralNetwork(layers, acts).evaluate(x)),
                                   net(x).numpy(), rtol=1e-9, atol=1e-9)
        # Pow(x, 3) a cube layer; Pow of another exponent runs, but does not lower
        init["two"] = np.array(3.0, np.float32)
        cube = OnnxNetwork(model(nodes, [("x", [1, 2])], [("y", [1, 3])], init), dtype=torch.float64)
        (layers, acts), _ = cube.layers()
        self.assertIn("cube", acts)
        np.testing.assert_allclose(np.asarray(cora.NeuralNetwork(layers, acts).evaluate(x)),
                                   net(x).numpy() ** 1.5, rtol=1e-9, atol=1e-9)
        init["two"] = np.array(1.5, np.float32)
        other = OnnxNetwork(model(nodes, [("x", [1, 2])], [("y", [1, 3])], init), dtype=torch.float64)
        np.testing.assert_allclose(other(x).numpy(), net(x).numpy() ** 0.75, rtol=1e-9)
        self.assertIsNone(other.layers()[0])

    def test_signs_and_an_output_softmax(self):
        # a binarized network: sign layers; its output Softmax dropped, the logits equal the graph's
        m = sign_net()
        net = OnnxNetwork(m, dtype=torch.float64)
        (whole, kept), _ = net.layers()  # the Softmax kept: exp and 1 / sum, like the graph
        self.assertEqual(kept[-2:], ["exp", "reciprocal"])
        x = torch.as_tensor(RNG.uniform(-1, 1, (6, net.num_inputs)))
        np.testing.assert_allclose(np.asarray(cora.NeuralNetwork(whole, kept).evaluate(x)),
                                   net(x).numpy(), rtol=1e-12, atol=1e-12)
        (layers, acts), why = net.layers(drop_output_softmax=True)
        self.assertIsNone(why)
        self.assertTrue(net.dropped_softmax)
        self.assertEqual(acts.count("sign"), 2)
        nn = cora.NeuralNetwork(layers, acts)
        x = torch.as_tensor(RNG.uniform(-1, 1, (6, net.num_inputs)))
        logits = torch.softmax(torch.as_tensor(np.asarray(nn.evaluate(x))), -1)
        np.testing.assert_allclose(logits.numpy(), net(x).numpy(), rtol=1e-12, atol=1e-12)
        # a property of comparisons only keeps the order of the outputs; one with a constant not
        from cora.nn_verify import _compares_outputs
        from cora.vnnlib import parse_vnnlib

        box = "".join(f"(declare-const X_{i} Real)\n(assert (<= X_{i} 1))\n(assert (>= X_{i} 0))\n"
                      for i in range(3))
        outs = "(declare-const Y_0 Real)\n(declare-const Y_1 Real)\n"
        self.assertTrue(_compares_outputs(parse_vnnlib(box + outs + "(assert (>= Y_0 Y_1))\n", 3, 2)))
        self.assertFalse(_compares_outputs(parse_vnnlib(box + outs + "(assert (>= Y_0 0.5))\n", 3, 2)))

    def test_attention_and_softmax(self):
        # SAGAN's self-attention: two products of values and a softmax inside, equal to the graph
        net = OnnxNetwork(attention_net(), dtype=torch.float64)
        nn, why = net.to_neural_network()
        self.assertIsNone(why)
        x = torch.as_tensor(RNG.uniform(-1, 1, (6, net.num_inputs)))
        np.testing.assert_allclose(
            np.asarray(nn.evaluate(x)), net(x).numpy(), rtol=1e-10, atol=1e-12
        )
        (_, acts), _ = OnnxNetwork(attention_net()).layers()
        self.assertEqual([a for a in acts if a != "identity"],
                         ["product", "exp", "reciprocal", "product"])
        # verify on its enclosure: a sampled violation is never unsat
        x = RNG.uniform(-1, 1, (4000, net.num_inputs))
        y = net(torch.as_tensor(x)).numpy()
        A = np.zeros((1, 1, net.num_outputs))
        A[0, 0, 0], A[0, 0, 1] = 1.0, -1.0
        low = float((y @ A[0, 0]).min())
        lo, hi = -np.ones((1, net.num_inputs)), np.ones((1, net.num_inputs))
        out = nn.verify(lo, hi, [0], A, np.array([[low + 0.01]]), 20, cora.vnncomp_options("x"))
        self.assertEqual(out["status"], "sat")

    def test_vision_transformer(self):
        # vit_2023's graph (vit_net): the torch run as onnxruntime, then the layers as the graph
        m = vit_net()
        net = OnnxNetwork(m, dtype=torch.float64)
        x = RNG.uniform(-1, 1, (5, net.num_inputs))
        expected = onnxruntime_rows(m, x.astype(np.float32))
        np.testing.assert_allclose(net(torch.as_tensor(x)).numpy(), expected, rtol=1e-5, atol=1e-5)
        nn, why = net.to_neural_network()
        self.assertIsNone(why)
        np.testing.assert_allclose(
            np.asarray(nn.evaluate(torch.as_tensor(x))), net(torch.as_tensor(x)).numpy(), rtol=1e-10,
            atol=1e-12,
        )

    def test_gather_from_the_end(self):
        # nn4sys' pensieve: the last entry of a row by a Gather of index -1, onnxruntime's value
        m = pensieve_net()
        net = OnnxNetwork(m, dtype=torch.float64)
        x = RNG.uniform(-1, 1, (5, net.num_inputs))
        expected = onnxruntime_rows(m, x.astype(np.float32))
        np.testing.assert_allclose(net(torch.as_tensor(x)).numpy(), expected, rtol=1e-5, atol=1e-5)
        nn, why = net.to_neural_network()
        self.assertIsNone(why)
        np.testing.assert_allclose(np.asarray(nn.evaluate(torch.as_tensor(x))), expected, rtol=1e-5,
                                   atol=1e-5)

    def test_quotients_and_an_input_without_batch(self):
        # pensieve_parallel's input without a batch axis and relu^3 / sum relu^3; x / (relu + 1)
        for m in (pensieve_parallel_net(), quotient_net()):
            net = OnnxNetwork(m, dtype=torch.float64)
            (layers, acts), why = net.layers()
            self.assertIsNone(why)
            self.assertIn("inverse", acts)
            x = RNG.uniform(0.1, 1, (6, net.num_inputs))
            expected = onnxruntime_rows(m, x.astype(np.float32))
            got = np.asarray(cora.NeuralNetwork(layers, acts).evaluate(torch.as_tensor(x)))
            np.testing.assert_allclose(got, net(torch.as_tensor(x)).numpy(), rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
        self.assertIn("cube", OnnxNetwork(pensieve_parallel_net()).layers()[0][1])
        # an input without a batch axis read by another operator does not lower
        bad = pensieve_parallel_net(reshape=False)
        self.assertIn("without a batch axis", OnnxNetwork(bad).layers()[1])

    def test_sin_cos_and_constant_shapes(self):
        # ml4acopf's power flows: v_i v_j cos(t_i - t_j), sin of the same; Where, Expand on constants
        m = trig_net()
        net = OnnxNetwork(m, dtype=torch.float64)
        (layers, acts), why = net.layers()
        self.assertIsNone(why)
        self.assertTrue({"sin", "cos", "product"} <= set(acts))
        x = RNG.uniform(-1, 1, (6, net.num_inputs))
        got = np.asarray(cora.NeuralNetwork(layers, acts).evaluate(torch.as_tensor(x)))
        np.testing.assert_allclose(got, net(torch.as_tensor(x)).numpy(), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(got, onnxruntime_rows(m, x.astype(np.float32)), rtol=1e-5,
                                   atol=1e-5)

    def test_columns_and_thresholds(self):
        # ml4acopf: a column C x^T, a ReLU sum over thresholds; a column into a join refuses
        m = acopf_net()
        net = OnnxNetwork(m, dtype=torch.float64)
        x = RNG.uniform(-1, 1, (5, net.num_inputs))
        expected = onnxruntime_rows(m, x.astype(np.float32))
        np.testing.assert_allclose(net(torch.as_tensor(x)).numpy(), expected, rtol=1e-5, atol=1e-5)
        nn, why = net.to_neural_network()
        self.assertIsNone(why)
        np.testing.assert_allclose(np.asarray(nn.evaluate(torch.as_tensor(x))), expected,
                                   rtol=1e-5, atol=1e-5)
        _, why = OnnxNetwork(acopf_net(join_column=True)).layers()
        self.assertIn("column", why)

    def test_backend_by_network(self):
        # torch for a convolution (or a GPU), eigen for a dense network (cora.nn_verify.load)
        from cora.nn_verify import choose_backend

        conv, _ = OnnxNetwork(conv_net()).layers()
        dense, _ = OnnxNetwork(mlp()).layers()
        self.assertEqual(choose_backend(conv[0]), "torch")
        self.assertEqual(choose_backend(dense[0]), "eigen")
        self.assertEqual(choose_backend(dense[0], "cuda:0"), "torch")
        self.assertEqual(choose_backend(dense[0], "cpu", ["relu", "product"]), "torch")

    def test_size_cap(self):
        # max_elements refuses unit-vector batches beyond it
        _, why = OnnxNetwork(conv_net()).dense_layers(max_elements=100)
        self.assertIn("max_elements", why)


class Loading(unittest.TestCase):
    def test_shapes_and_operators(self):
        net = OnnxNetwork(conv_net())
        self.assertEqual((net.input_shape, net.num_inputs, net.num_outputs), ([1, 1, 4, 4], 16, 4))
        self.assertIn("Conv", net.op_types())
        self.assertGreater(net.activation_elements(), 16)

    def test_unknown_operator(self):
        m = model([helper.make_node("Erf", ["x"], ["y"])], [("x", [1, 2])], [("y", [1, 2])], {})
        with self.assertRaisesRegex(NotImplementedError, "Erf"):
            OnnxNetwork(m)(torch.zeros(1, 2))


# -----------------------------------------  AUXILIARY  ------------------------------------------ #

RNG = np.random.default_rng(3)


def weight(*shape):
    """A random float32 array."""
    return RNG.standard_normal(shape).astype(np.float32)


def model(nodes, inputs, outputs, initializers, opset=13):
    """A model of the nodes with float32 inputs and outputs (name, shape), of the opset."""
    graph = helper.make_graph(
        nodes,
        "g",
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, s) for n, s in inputs],
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, s) for n, s in outputs],
        [numpy_helper.from_array(a, name) for name, a in initializers.items()],
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m


def onnxruntime_rows(m, x):
    """onnxruntime's outputs of the rows x, one sample of the declared shape at a time."""
    sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
    return np.stack([sess.run(None, {inp.name: xi.reshape(shape)})[0].reshape(-1) for xi in x])


def dense_rows(layers, x):
    """The lowered layers on the rows x in float64."""
    h = torch.as_tensor(x, dtype=torch.float64)
    act = {"relu": torch.relu, "sigmoid": torch.sigmoid, "tanh": torch.tanh, "identity": lambda t: t}
    for (W, b), a in zip(*layers, strict=True):
        h = act[a](h @ W.T + b)
    return h.numpy()


def split_net(broken=""):
    """ReluSplitter's split of the first layer of relu(W0 x + b0) -> W3 (x of 4, 3 neurons): a
    layer of the pairs relu(s W0_i x + c_i), relu(-s W0_i x - c_i) in shuffled order, read as s and
    -s by neuron i with the bias b0_i - s c_i; and the base network. broken: a column of the second
    neuron not opposite ("column"), or one row unlike its partner ("row")."""
    W0, b0, W3, b3 = weight(3, 4), weight(3), weight(2, 3), weight(2)
    s = np.array([1, -1, 1], np.float32)
    c = np.array([5.0, -2.5, 0.75], np.float32)
    order = [3, 0, 5, 1, 4, 2]  # pair i is (i, i + 3) before the shuffle
    W1 = np.concatenate([s[:, None] * W0, -s[:, None] * W0])[order]
    b1 = np.concatenate([c, -c])[order]
    W2 = np.concatenate([np.diag(s), -np.diag(s)], 1)[:, order]
    b2 = (b0 - s * c).astype(np.float32)
    if broken == "column":
        W2[1, order.index(4)] *= 0.5
    if broken == "row":
        W1[order.index(5), 0] += 1.0
    nodes = [
        helper.make_node("Gemm", ["x", "W1", "b1"], ["g1"], transB=1),
        helper.make_node("Relu", ["g1"], ["r1"]),
        helper.make_node("Gemm", ["r1", "W2", "b2"], ["g2"], transB=1),
        helper.make_node("Relu", ["g2"], ["r2"]),
        helper.make_node("Gemm", ["r2", "W3", "b3"], ["y"], transB=1),
    ]
    init = {"W1": W1.astype(np.float32), "b1": b1, "W2": W2.astype(np.float32), "b2": b2,
            "W3": W3, "b3": b3}
    split = model(nodes, [("x", [1, 4])], [("y", [1, 2])], init)
    base = model([helper.make_node("Gemm", ["x", "W0", "b0"], ["g2"], transB=1)] + nodes[3:],
                 [("x", [1, 4])], [("y", [1, 2])], {"W0": W0, "b0": b0, "W3": W3, "b3": b3})
    return split, base


def mlp():
    """Sub, Flatten, MatMul + Add, Relu, Gemm, Sigmoid, Mul, Div, Tanh: acasxu-like."""
    nodes = [
        helper.make_node("Sub", ["x", "mean"], ["s"]),
        helper.make_node("Flatten", ["s"], ["f"]),
        helper.make_node("MatMul", ["f", "W1"], ["m"]),
        helper.make_node("Add", ["m", "b1"], ["a"]),
        helper.make_node("Relu", ["a"], ["r"]),
        helper.make_node("Gemm", ["r", "W2", "b2"], ["g"], transB=1),
        helper.make_node("Sigmoid", ["g"], ["sg"]),
        helper.make_node("Mul", ["sg", "k"], ["mu"]),
        helper.make_node("Div", ["mu", "k2"], ["d"]),
        helper.make_node("Tanh", ["d"], ["t"]),
        helper.make_node("Gemm", ["t", "W3", "b3"], ["y"]),
    ]
    # the constants of the graph
    init = {
        "mean": weight(1, 1, 1, 5),
        "W1": weight(5, 8),
        "b1": weight(8),
        "W2": weight(6, 8),
        "b2": weight(6),
        "k": weight(6),
        "k2": np.full(6, 2.0, np.float32),
        "W3": weight(6, 3),
        "b3": weight(3),
    }
    return model(nodes, [("x", [1, 1, 1, 5])], [("y", [1, 3])], init)


def structured_rows(layers, activations, x):
    """The layers of OnnxNetwork.layers on the rows x, by torch (the reference)."""
    import torch.nn.functional as F

    h = torch.as_tensor(x, dtype=torch.float64)
    acts = {"relu": torch.relu, "sigmoid": torch.sigmoid, "tanh": torch.tanh}
    for layer, act in zip(layers, activations, strict=True):
        # a dense layer, a gather, a ConvTranspose or a convolution
        if not isinstance(layer, dict):
            h = h @ layer[0].T + layer[1]
        elif "W" in layer:
            h = h @ layer["W"].T + layer["bias"]
        elif "gather" in layer:
            padded = torch.cat([h, torch.zeros(h.shape[0], 1, dtype=h.dtype)], 1)
            index = [i if i >= 0 else h.shape[1] for i in layer["gather"]]
            h = padded[:, index] + layer["bias"]
        elif "conv_transpose2d" in layer:
            c, hh, ww = layer["output"]
            # the transpose of the convolution on the output: by autograd of the convolution
            t, left, b, r = layer["pads"]
            z0 = torch.zeros(h.shape[0], c, hh, ww, dtype=h.dtype, requires_grad=True)
            z = F.conv2d(
                F.pad(z0, (left, r, t, b)),
                layer["conv_transpose2d"],
                stride=layer["stride"],
                dilation=layer["dilation"],
                groups=layer["groups"],
            )
            (g,) = torch.autograd.grad(z, z0, h.reshape(z.shape))
            h = g.reshape(h.shape[0], -1) + layer["bias"]
        else:
            # a convolution: the pads first, as ONNX's may be asymmetric
            c, hh, ww = layer["input"]
            t, left, b, r = layer["pads"]
            z = F.pad(h.reshape(-1, c, hh, ww), (left, r, t, b))
            z = F.conv2d(
                z, layer["conv2d"], stride=layer["stride"], dilation=layer["dilation"], groups=layer["groups"]
            )
            h = z.reshape(h.shape[0], -1) + layer["bias"]
        h = acts[act](h) if act in acts else h
    return h.numpy()


def conv_net():
    """Conv (asymmetric pads), BatchNormalization, Relu, AveragePool, Transpose, Reshape (to a
    constant shape of batch 1), Concat with a constant, Dropout, Identity, MatMul."""
    nodes = [
        helper.make_node("Conv", ["x", "Wc", "bc"], ["c"], pads=[1, 0, 0, 1], strides=[1, 1]),
        helper.make_node("BatchNormalization", ["c", "s", "o", "mu", "var"], ["bn"]),
        helper.make_node("Relu", ["bn"], ["r"]),
        helper.make_node("AveragePool", ["r"], ["p"], kernel_shape=[2, 2], strides=[1, 1]),
        helper.make_node("Transpose", ["p"], ["t"], perm=[0, 1, 3, 2]),
        helper.make_node("Reshape", ["t", "shape"], ["f"]),
        helper.make_node("Concat", ["f", "extra"], ["cat"], axis=1),
        helper.make_node("Dropout", ["cat"], ["dr"]),
        helper.make_node("Identity", ["dr"], ["id"]),
        helper.make_node("MatMul", ["id", "W"], ["y"]),
    ]
    # the constants of the graph
    init = {
        "Wc": weight(2, 1, 2, 2),
        "bc": weight(2),
        "s": weight(2),
        "o": weight(2),
        "mu": weight(2),
        "var": np.abs(weight(2)) + 0.5,
        "shape": np.array([1, 18], np.int64),
        "extra": weight(1, 2),
        "W": weight(20, 4),
    }
    return model(nodes, [("x", [1, 1, 4, 4])], [("y", [1, 4])], init)


def residual_net():
    """A residual connection: y = W2 relu(W1 x) + x."""
    nodes = [
        helper.make_node("MatMul", ["x", "W1"], ["h"]),
        helper.make_node("Relu", ["h"], ["r"]),
        helper.make_node("MatMul", ["r", "W2"], ["m"]),
        helper.make_node("Add", ["m", "x"], ["y"]),
    ]
    return model(nodes, [("x", [1, 3])], [("y", [1, 3])], {"W1": weight(3, 4), "W2": weight(4, 3)})


def product_net():
    """The product of two branches, which is no affine map: y = relu(W1 x) * x."""
    nodes = [
        helper.make_node("MatMul", ["x", "W1"], ["h"]),
        helper.make_node("Relu", ["h"], ["r"]),
        helper.make_node("Mul", ["r", "x"], ["y"]),
    ]
    return model(nodes, [("x", [1, 3])], [("y", [1, 3])], {"W1": weight(3, 3)})


def attention_net():
    """SAGAN's self-attention on a (2, 2, 2) image: theta^T phi, its softmax, g times its
    transpose, then x + sigma o(...) into a Gemm."""
    nodes = [
        helper.make_node("Conv", ["x", "Wt"], ["t"]),
        helper.make_node("Reshape", ["t", "s1"], ["tr"]),
        helper.make_node("Transpose", ["tr"], ["tt"], perm=[0, 2, 1]),
        helper.make_node("Conv", ["x", "Wp"], ["p"]),
        helper.make_node("Reshape", ["p", "s1"], ["pr"]),
        helper.make_node("MatMul", ["tt", "pr"], ["att"]),
        helper.make_node("Softmax", ["att"], ["sm"], axis=2),
        # the values g, the softmax transposed, their product, then the residual
        helper.make_node("Conv", ["x", "Wg"], ["g"]),
        helper.make_node("Reshape", ["g", "s2"], ["gr"]),
        helper.make_node("Transpose", ["sm"], ["smt"], perm=[0, 2, 1]),
        helper.make_node("MatMul", ["gr", "smt"], ["o"]),
        helper.make_node("Reshape", ["o", "s3"], ["oi"]),
        helper.make_node("Mul", ["sigma", "oi"], ["os"]),
        helper.make_node("Add", ["x", "os"], ["a"]),
        helper.make_node("Flatten", ["a"], ["f"]),
        helper.make_node("Gemm", ["f", "W", "b"], ["y"], transB=1),
    ]
    init = {"Wt": weight(1, 2, 1, 1), "Wp": weight(1, 2, 1, 1), "Wg": weight(2, 2, 1, 1),
            "s1": np.array([1, 1, 4], np.int64), "s2": np.array([1, 2, 4], np.int64),
            "s3": np.array([1, 2, 2, 2], np.int64), "sigma": np.array([0.7], np.float32),
            "W": weight(3, 8), "b": weight(3)}
    return model(nodes, [("x", [1, 2, 2, 2])], [("y", [1, 3])], init)


def vit_net():
    """vit_2023's shapes at opset 9 on a (2, 4, 4) image: 2 x 2 patches by a Conv into 4 tokens of
    3, a class token (ConstantOfShape of the batch, plus the token) and positions, a batch norm of
    the tokens, one attention head with a residual, the mean of the tokens, a batch norm, a Gemm."""
    nodes = [
        helper.make_node("Conv", ["x", "Wc", "bc"], ["p"], kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node("Shape", ["p"], ["ps"]),
        helper.make_node("Slice", ["ps"], ["pb"], starts=[0], ends=[2], axes=[0]),
        helper.make_node("Concat", ["pb", "minus1"], ["s"], axis=0),
        helper.make_node("Reshape", ["p", "s"], ["pr"]),
        helper.make_node("Transpose", ["pr"], ["tok"], perm=[0, 2, 1]),
        # the class token: zeros of (batch, 1, 3) plus the learned token
        helper.make_node("Shape", ["x"], ["xs"]),
        helper.make_node("Gather", ["xs", "zero"], ["n"], axis=0),
        helper.make_node("Unsqueeze", ["n"], ["n1"], axes=[0]),
        helper.make_node("Concat", ["n1", "one", "three"], ["cs"], axis=0),
        helper.make_node("ConstantOfShape", ["cs"], ["c0"]),
        helper.make_node("Add", ["c0", "cls"], ["c"]),
        helper.make_node("Concat", ["c", "tok"], ["t"], axis=1),
        helper.make_node("Add", ["t", "pos"], ["e"]),
        # a batch norm of the tokens' features, between two transposes
        helper.make_node("Transpose", ["e"], ["et"], perm=[0, 2, 1]),
        helper.make_node("BatchNormalization", ["et", "g1", "b1", "m1", "v1"], ["en"]),
        helper.make_node("Transpose", ["en"], ["h"], perm=[0, 2, 1]),
        # one head: softmax(q k^T / sqrt(3)) v, then the residual
        helper.make_node("MatMul", ["h", "Wq"], ["q"]),
        helper.make_node("MatMul", ["h", "Wk"], ["k"]),
        helper.make_node("MatMul", ["h", "Wv"], ["v"]),
        helper.make_node("Transpose", ["k"], ["kt"], perm=[0, 2, 1]),
        helper.make_node("MatMul", ["q", "kt"], ["qk"]),
        helper.make_node("Mul", ["qk", "scale"], ["qs"]),
        helper.make_node("Softmax", ["qs"], ["a"], axis=2),
        helper.make_node("MatMul", ["a", "v"], ["o"]),
        helper.make_node("Add", ["o", "e"], ["r"]),
        helper.make_node("ReduceMean", ["r"], ["mean"], axes=[1], keepdims=0),
        helper.make_node("BatchNormalization", ["mean", "g2", "b2", "m2", "v2"], ["mn"]),
        helper.make_node("Gemm", ["mn", "W", "b"], ["y"], transB=1),
    ]
    var = lambda n: RNG.uniform(0.5, 2, n).astype(np.float32)  # noqa: E731
    init = {"Wc": weight(3, 2, 2, 2), "bc": weight(3), "minus1": np.array([-1], np.int64),
            "zero": np.array(0, np.int64), "one": np.array([1], np.int64),
            "three": np.array([3], np.int64), "cls": weight(1, 1, 3), "pos": weight(1, 5, 3),
            "g1": weight(3), "b1": weight(3), "m1": weight(3), "v1": var(3),
            "Wq": weight(3, 3), "Wk": weight(3, 3), "Wv": weight(3, 3),
            "scale": np.array(1 / np.sqrt(3), np.float32),
            "g2": weight(3), "b2": weight(3), "m2": weight(3), "v2": var(3),
            "W": weight(2, 3), "b": weight(2)}
    return model(nodes, [("x", [1, 2, 4, 4])], [("y", [1, 2])], init, opset=9)


def pensieve_net():
    """nn4sys' pensieve on a (2, 4) input: the last entry of the first row (Slice, then a Gather
    of index -1 on axis 2) and the second row, each through a Gemm and a ReLU, concatenated."""
    nodes = [
        helper.make_node("Slice", ["x", "s0", "s1", "ax1"], ["r0"]),
        helper.make_node("Gather", ["r0", "last"], ["g"], axis=2),
        helper.make_node("Gemm", ["g", "W0", "b0"], ["h0"], transB=1),
        helper.make_node("Relu", ["h0"], ["a0"]),
        helper.make_node("Gather", ["x", "one"], ["r1"], axis=1),
        helper.make_node("Gemm", ["r1", "W1", "b1"], ["h1"], transB=1),
        helper.make_node("Relu", ["h1"], ["a1"]),
        helper.make_node("Concat", ["a0", "a1"], ["c"], axis=1),
        helper.make_node("Gemm", ["c", "W", "b"], ["y"], transB=1),
    ]
    init = {"s0": np.array([0], np.int64), "s1": np.array([1], np.int64),
            "ax1": np.array([1], np.int64), "last": np.array(-1, np.int64),
            "one": np.array(1, np.int64), "W0": weight(3, 1), "b0": weight(3),
            "W1": weight(3, 4), "b1": weight(3), "W": weight(2, 6), "b": weight(2)}
    return model(nodes, [("x", [1, 2, 4])], [("y", [1, 2])], init)


def acopf_net(join_column=False):
    """ml4acopf's shapes on x (1, 3): h = relu(x W), C h^T as a column by two transposes, and
    sum_k relu(h - t_k) d_k (an Unsqueeze broadcast against the thresholds), added; with
    join_column the column itself joins h."""
    nodes = [
        helper.make_node("MatMul", ["x", "W"], ["l"]),
        helper.make_node("Relu", ["l"], ["h"]),
        helper.make_node("Transpose", ["h"], ["ht"], perm=[1, 0]),
        helper.make_node("MatMul", ["C", "ht"], ["ch"]),
        helper.make_node("Transpose", ["ch"], ["c"], perm=[1, 0]),
        helper.make_node("Unsqueeze", ["h", "last"], ["hu"]),
        helper.make_node("Sub", ["hu", "t"], ["ht3"]),
        helper.make_node("Relu", ["ht3"], ["r"]),
        helper.make_node("MatMul", ["r", "d"], ["p"]),
        helper.make_node("Add", ["c", "p"] if not join_column else ["ch", "p"], ["y"]),
    ]
    init = {"W": weight(3, 2), "C": weight(2, 2), "t": weight(4), "d": weight(4),
            "last": np.array([-1], np.int64)}
    return model(nodes, [("x", [1, 3])], [("y", [1, 2] if not join_column else [2, 2])], init)


def sign_net():
    """A binarized network: sign(x W1), + 0.1, sign, then W2 and a Softmax (traffic_signs')."""
    nodes = [
        helper.make_node("MatMul", ["x", "W1"], ["h"]),
        helper.make_node("Sign", ["h"], ["s"]),
        helper.make_node("Add", ["s", "eps"], ["a"]),
        helper.make_node("Sign", ["a"], ["t"]),
        helper.make_node("MatMul", ["t", "W2"], ["z"]),
        helper.make_node("Softmax", ["z"], ["y"]),
    ]
    init = {"W1": weight(3, 4), "eps": np.full((1, 4), 0.1, np.float32), "W2": weight(4, 2)}
    return model(nodes, [("x", [1, 3])], [("y", [1, 2])], init)


def quotient_net(broadcast=False):
    """A quotient of two values: y = x / (relu(W1 x) + 1); with broadcast, the dividend the first
    entry of x (1, 1) against the divisor's three."""
    nodes = [
        helper.make_node("MatMul", ["x", "W1"], ["h"]),
        helper.make_node("Relu", ["h"], ["r"]),
        helper.make_node("Add", ["r", "one"], ["d"]),
    ]
    if broadcast:
        nodes += [helper.make_node("Slice", ["x", "s0", "s1", "ax1"], ["x0"]),
                  helper.make_node("Div", ["x0", "d"], ["y"])]
    else:
        nodes += [helper.make_node("Div", ["x", "d"], ["y"])]
    init = {"W1": weight(3, 3), "one": np.ones((1, 3), np.float32)}
    if broadcast:
        init |= {"s0": np.array([0], np.int64), "s1": np.array([1], np.int64),
                 "ax1": np.array([1], np.int64)}
    return model(nodes, [("x", [1, 3])], [("y", [1, 3])], init)


def trig_net():
    """ml4acopf-shaped: angles t = x[:, :2], magnitudes v = x[:, 2:]; y = [v0 v1 cos(t0 - t1), v0 v1
    sin(t0 - t1)] plus a constant expanded by Where/Equal/ConstantOfShape (opset 13)."""
    # the angles' difference and the magnitudes' product, then cos and sin of it times that
    nodes = [
        helper.make_node("Gather", ["x", "i0"], ["t0"], axis=1),
        helper.make_node("Gather", ["x", "i1"], ["t1"], axis=1),
        helper.make_node("Gather", ["x", "i2"], ["v0"], axis=1),
        helper.make_node("Gather", ["x", "i3"], ["v1"], axis=1),
        helper.make_node("Sub", ["t0", "t1"], ["d"]),
        helper.make_node("Cos", ["d"], ["c"]),
        helper.make_node("Sin", ["d"], ["s"]),
        helper.make_node("Mul", ["v0", "v1"], ["vv"]),
        helper.make_node("Mul", ["vv", "c"], ["p"]),
        helper.make_node("Mul", ["vv", "s"], ["q"]),
        # ml4acopf's shape arithmetic on constants: Expand to Where(shape == -1, 1, shape)
        helper.make_node("ConstantOfShape", ["two"], ["ones"],
                         value=numpy_helper.from_array(np.array([1], np.int64))),
        helper.make_node("Mul", ["ones", "minus"], ["neg"]),
        helper.make_node("Equal", ["shape", "neg"], ["eq"]),
        helper.make_node("Where", ["eq", "ones", "shape"], ["w"]),
        helper.make_node("Expand", ["k", "w"], ["kk"]),
        helper.make_node("Concat", ["p", "q"], ["pq"], axis=1),
        helper.make_node("Add", ["pq", "kk"], ["y"]),
    ]
    init = {"i0": np.array([0], np.int64), "i1": np.array([1], np.int64),
            "i2": np.array([2], np.int64), "i3": np.array([3], np.int64),
            "two": np.array([2], np.int64), "minus": np.array(-1, np.int64),
            "shape": np.array([1, -1], np.int64), "k": np.array([0.5, -0.25], np.float32)}
    return model(nodes, [("x", [1, 4])], [("y", [1, 2])], init)


def pensieve_parallel_net(reshape=True):
    """nn4sys' pensieve_parallel on an input (2, 4) without a batch axis: Reshape to (1, 2, 4),
    each row through a Gemm and a ReLU, cubed and divided by the cubes' sum (Pow, ReduceSum, Div),
    the difference of the two rows' weighted means."""
    nodes = [helper.make_node("Reshape", ["x", "shape"], ["xr"])] if reshape else [
        helper.make_node("Unsqueeze", ["x"], ["xr"], axes=[0])]
    for i in range(2):
        nodes += [
            helper.make_node("Gather", ["xr", f"i{i}"], [f"r{i}"], axis=1),
            helper.make_node("Gemm", [f"r{i}", "W", "b"], [f"h{i}"], transB=1),
            helper.make_node("Relu", [f"h{i}"], [f"a{i}"]),
            helper.make_node("Pow", [f"a{i}", "three"], [f"p{i}"]),
            helper.make_node("ReduceSum", [f"p{i}"], [f"s{i}"], axes=[1], keepdims=1),
            helper.make_node("Div", [f"p{i}", f"s{i}"], [f"q{i}"]),
            helper.make_node("MatMul", [f"q{i}", "v"], [f"m{i}"]),
        ]
    nodes += [helper.make_node("Sub", ["m0", "m1"], ["y"])]
    init = {"shape": np.array([-1, 2, 4], np.int64),
            "i0": np.array(0, np.int64), "i1": np.array(1, np.int64), "W": np.abs(weight(3, 4)) + 0.1,
            "b": np.abs(weight(3)) + 0.5, "three": np.array(3.0, np.float32),
            "v": np.array([[1.0], [2.0], [4.0]], np.float32)}
    return model(nodes, [("x", [2, 4])], [("y", [1, 1])], init, opset=11)


def lyapunov_net():
    """lsnc_relu's shapes: V = sum(x * (x P)) by ReduceSum, beside -((c - (relu(x W) - x)) * s) / t,
    concatenated."""
    nodes = [
        helper.make_node("MatMul", ["x", "P"], ["xp"]),
        helper.make_node("Mul", ["x", "xp"], ["q"]),
        helper.make_node("ReduceSum", ["q", "axes"], ["v"], keepdims=1),
        helper.make_node("MatMul", ["x", "W"], ["h"]),
        helper.make_node("Relu", ["h"], ["r"]),
        helper.make_node("Sub", ["r", "x"], ["d"]),
        helper.make_node("Sub", ["c", "d"], ["e"]),
        helper.make_node("Mul", ["e", "s"], ["f"]),
        helper.make_node("Div", ["f", "t"], ["g"]),
        helper.make_node("Neg", ["g"], ["n"]),
        helper.make_node("Concat", ["v", "n"], ["y"], axis=1),
    ]
    init = {"P": weight(3, 3), "axes": np.array([-1], np.int64), "W": weight(3, 3),
            "c": weight(1, 3), "s": weight(1, 3), "t": np.array(2.0, np.float32)}
    return model(nodes, [("x", [1, 3])], [("y", [1, 4])], init)


def resnet_block():
    """A ResNet block: relu(BN(conv(relu(BN(conv x)))) + BN(conv1x1 x)) as a Gemm's input, the
    shortcut (strided) after the main branch in the graph."""
    bn = {f"{k}{i}": v for i in range(3) for k, v in
          (("s", weight(2)), ("o", weight(2)), ("mu", weight(2)), ("var", np.abs(weight(2)) + 0.5))}
    nodes = [
        helper.make_node("Conv", ["x", "W1", "b1"], ["c1"], pads=[1, 1, 1, 1], strides=[2, 2]),
        helper.make_node("BatchNormalization", ["c1", "s0", "o0", "mu0", "var0"], ["n1"]),
        helper.make_node("Relu", ["n1"], ["r1"]),
        helper.make_node("Conv", ["r1", "W2", "b2"], ["c2"], pads=[1, 1, 1, 1]),
        helper.make_node("BatchNormalization", ["c2", "s1", "o1", "mu1", "var1"], ["n2"]),
        helper.make_node("Conv", ["x", "W3", "b3"], ["c3"], strides=[2, 2]),
        helper.make_node("BatchNormalization", ["c3", "s2", "o2", "mu2", "var2"], ["n3"]),
        helper.make_node("Add", ["n2", "n3"], ["a"]),
        helper.make_node("Relu", ["a"], ["r2"]),
        helper.make_node("Flatten", ["r2"], ["f"]),
        helper.make_node("Gemm", ["f", "W", "b"], ["y"], transB=1),
    ]
    init = {"W1": weight(2, 1, 3, 3), "b1": weight(2), "W2": weight(2, 2, 3, 3), "b2": weight(2),
            "W3": weight(2, 1, 1, 1), "b3": weight(2), "W": weight(3, 8), "b": weight(3), **bn}
    return model(nodes, [("x", [1, 1, 4, 4])], [("y", [1, 3])], init)


def cnn7_net():
    """Conv, Relu, Flatten, Gemm, BatchNormalization of the Gemm's features, Relu, Gemm."""
    nodes = [
        helper.make_node("Conv", ["x", "Wc", "bc"], ["c"], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c"], ["r"]),
        helper.make_node("Flatten", ["r"], ["f"]),
        helper.make_node("Gemm", ["f", "W1", "b1"], ["g"], transB=1),
        helper.make_node("BatchNormalization", ["g", "s", "o", "mu", "var"], ["n"]),
        helper.make_node("Relu", ["n"], ["r2"]),
        helper.make_node("Gemm", ["r2", "W2", "b2"], ["y"], transB=1),
    ]
    init = {"Wc": weight(2, 1, 3, 3), "bc": weight(2), "W1": weight(5, 32), "b1": weight(5),
            "s": weight(5), "o": weight(5), "mu": weight(5), "var": np.abs(weight(5)) + 0.5,
            "W2": weight(3, 5), "b2": weight(3)}
    return model(nodes, [("x", [1, 1, 4, 4])], [("y", [1, 3])], init)


def slice_concat_net():
    """linearizenn's shape: a controller relu(x W1) W2 beside the slice x[:, 2:], concatenated."""
    nodes = [
        helper.make_node("MatMul", ["x", "W1"], ["h"]),
        helper.make_node("Relu", ["h"], ["r"]),
        helper.make_node("MatMul", ["r", "W2"], ["u"]),
        helper.make_node("Slice", ["x", "starts", "ends", "axes"], ["xs"]),
        helper.make_node("MatMul", ["xs", "W3"], ["v"]),
        helper.make_node("Concat", ["u", "v", "k"], ["cat"], axis=1),
        helper.make_node("MatMul", ["cat", "W4"], ["y"]),
    ]
    init = {"W1": weight(4, 6), "W2": weight(6, 1), "W3": weight(2, 1), "k": weight(1, 1),
            "W4": weight(3, 2), "starts": np.array([2], np.int64),
            "ends": np.array([2**62], np.int64), "axes": np.array([1], np.int64)}
    return model(nodes, [("x", [1, 4])], [("y", [1, 2])], init)


def wide_conv_net():
    """Conv (1 -> 16 channels, 3 x 3) on 32 x 32, Relu, Flatten, Gemm of the 14400 entries."""
    nodes = [
        helper.make_node("Conv", ["x", "Wc", "bc"], ["c"]),
        helper.make_node("Relu", ["c"], ["r"]),
        helper.make_node("Flatten", ["r"], ["f"]),
        helper.make_node("Gemm", ["f", "W", "b"], ["y"], transB=1),
    ]
    init = {"Wc": weight(16, 1, 3, 3), "bc": weight(16), "W": weight(5, 14400), "b": weight(5)}
    return model(nodes, [("x", [1, 1, 32, 32])], [("y", [1, 5])], init)


def conv_gemm_matmul_net():
    """Conv (1 -> 2 channels, 3 x 3) on 6 x 6, Flatten, Gemm (alpha 0.5), Relu, MatMul."""
    nodes = [
        helper.make_node("Conv", ["x", "Wc", "bc"], ["c"]),
        helper.make_node("Flatten", ["c"], ["f"]),
        helper.make_node("Gemm", ["f", "W1", "b1"], ["g"], alpha=0.5),
        helper.make_node("Relu", ["g"], ["r"]),
        helper.make_node("MatMul", ["r", "W2"], ["y"]),
    ]
    init = {"Wc": weight(2, 1, 3, 3), "bc": weight(2), "W1": weight(32, 7), "b1": weight(7),
            "W2": weight(7, 3)}
    return model(nodes, [("x", [1, 1, 6, 6])], [("y", [1, 3])], init)


def maxpool_net(kernel=2, stride=1, channels=1, pads=0):
    """A max pool of a (channels, 5, 5) input (ONNX's default stride 1), as the output."""
    out = (5 + 2 * pads - kernel) // stride + 1
    attrs = {"kernel_shape": [kernel, kernel], "strides": [stride, stride]}
    if pads:
        attrs["pads"] = [pads] * 4
    nodes = [
        helper.make_node("MaxPool", ["x"], ["p"], **attrs),
        helper.make_node("Flatten", ["p"], ["y"]),
    ]
    return model(
        nodes, [("x", [1, channels, 5, 5])], [("y", [1, channels * out * out])], {}
    )

# ----------------------------------------  END OF CODE  ----------------------------------------- #
