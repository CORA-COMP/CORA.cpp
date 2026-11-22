"""onnx_network - an ONNX network as a batched torch function, and as a cora.NeuralNetwork

The graph of an ONNX file runs node by node on torch tensors with a leading batch axis, on any
device and in any float dtype (float32 for the search, float64 for re-checks). Points are rows
(B, n) of the flattened input (C order), outputs rows (B, m). A graph that is a chain of affine
maps and element-wise activations lowers to a cora.NeuralNetwork: each affine part becomes one
dense linear layer, its matrix the image of the unit vectors with the additive constants dropped
(so a plain Gemm keeps its weights bit for bit), its bias the image of zero. A graph with
convolutions or two branches joined by an Add (a ResNet) lowers layer by layer, a branch as a
residual input of a layer; a MaxPool as max(a, b) = b + relu(a - b), pairwise over its window.

Syntax:   net = OnnxNetwork.load("net.onnx", dtype=torch.float32, device="cpu")
          y = net(x)                       x (B, n) -> y (B, m), batched, autograd on
          net64 = net.cast(torch.float64)  the same graph in another dtype or on another device
          nn, why = net.to_neural_network()   a cora.NeuralNetwork, or None and the reason
          (layers, acts), why = net.layers()   the same as layer specs and activation names
Inputs:   path - an .onnx or .onnx.gz file with one input and one output
See also: cora.vnnlib, cora.nn_verify, docs/nn_verification.md

Authors:       Claude Opus 5.5
Written:       21-November-2026
Last update:   22-November-2026
Reviewed:      ---
"""

import gzip
import math

import numpy as np
import torch
import torch.nn.functional as F


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #


class OnnxNetwork:
    """An ONNX graph run node by node on batched torch tensors; the constants are in one dtype and
    on one device, loaded once and reused across attacks and bounds."""

    def __init__(self, model, dtype=torch.float32, device="cpu"):
        from onnx import numpy_helper

        graph = model.graph
        self.dtype, self.device = dtype, torch.device(device)
        initializers = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
        inputs = [i for i in graph.input if i.name not in initializers]
        if len(inputs) != 1 or len(graph.output) < 1:
            raise ValueError(f"OnnxNetwork: one input expected, got {[i.name for i in inputs]}")
        self.input_name, self.output_name = inputs[0].name, graph.output[0].name
        self.input_shape, self.output_shape = _dims(inputs[0]), _dims(graph.output[0])
        self.input_onnx_dtype = inputs[0].type.tensor_type.elem_type
        self.num_inputs, self.num_outputs = (
            math.prod(self.input_shape),
            math.prod(self.output_shape),
        )
        # the constants: float ones in the network's dtype, integer ones (shapes, axes) as int64
        self.constants = {name: self._tensor(a) for name, a in initializers.items()}
        self.nodes = []
        for node in graph.node:
            attrs = _attributes(node)
            if node.op_type == "Constant":
                self.constants[node.output[0]] = self._tensor(numpy_helper.to_array(attrs["value"]))
                continue
            self.nodes.append((node.op_type, list(node.input), list(node.output), attrs))
        self.opset = max(
            (o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), default=13
        )

    @classmethod
    def load(cls, path, dtype=torch.float32, device="cpu"):
        """The network of the file, its weights in dtype on device."""
        return cls(_load_model(path), dtype=dtype, device=device)

    def cast(self, dtype=None, device=None):
        """The same network in another dtype or on another device (a copy of the constants)."""
        other = object.__new__(type(self))
        other.__dict__.update(self.__dict__)
        other.dtype = dtype or self.dtype
        other.device = torch.device(device) if device is not None else self.device
        other.constants = {
            k: (v.to(other.device, other.dtype) if v.is_floating_point() else v.to(other.device))
            for k, v in self.constants.items()
        }
        return other

    def activation_elements(self):
        """The number of elements of all values of one sample (the memory of a forward pass)."""
        with torch.no_grad():
            values = dict(self.constants)
            values[self.input_name] = torch.zeros(
                self.input_shape, dtype=self.dtype, device=self.device
            )
            total = 0
            for op, ins, outs, attrs in self.nodes:
                result = self._run(op, [values[name] if name else None for name in ins], attrs)
                for name, value in zip(
                    outs, result if isinstance(result, tuple) else (result,), strict=False
                ):
                    values[name] = value
                    total += value.numel() if value.is_floating_point() else 0
        return total

    def op_types(self):
        """The set of ONNX operators of the graph (Constant nodes are folded)."""
        return {op for op, _, _, _ in self.nodes}

    def __call__(self, x):
        """The outputs (B, m) of the rows x (B, n), in the network's dtype."""
        x = x.to(self.device, self.dtype)
        # one sample of the declared shape per row, the rows by torch.func.vmap
        y = torch.func.vmap(self._sample)(x.reshape(x.shape[0], *self.input_shape))
        return y.reshape(x.shape[0], -1)

    def _sample(self, x):
        """The graph on one input of the declared shape, as ONNX defines it."""
        values = dict(self.constants)
        values[self.input_name] = x
        for op, ins, outs, attrs in self.nodes:
            args = [values[name] if name else None for name in ins]
            result = self._run(op, args, attrs)
            for name, value in zip(
                outs, result if isinstance(result, tuple) else (result,), strict=False
            ):
                values[name] = value
        return values[self.output_name]

    def to_neural_network(self, max_elements=2**27):
        """(cora.NeuralNetwork in float64, None), or (None, why) for a graph that is no chain of
        affine maps and relu/sigmoid/tanh; max_elements caps the unit-vector batches."""
        import cora

        lowered, why = self.layers(max_elements)
        return (cora.NeuralNetwork(*lowered), None) if lowered else (None, why)

    def layers(self, max_elements=2**27):
        """(([layer], activation names), None) of the network in float64, or (None, why): a chain
        without a Conv or an AveragePool as dense layers; else one layer per linear node
        (cora.NeuralNetwork's conv2d, gather and dense dicts, never a matrix of a convolution),
        with the graph keys of a residual connection."""
        if not any(op in STRUCTURED_OPS for op, _, _, _ in self.nodes):
            dense, why = self.dense_layers(max_elements)
            if dense or "joins two layers" not in why:
                return dense, why
        try:
            return self._lower_structured(), None
        except (_NotAChain, NotImplementedError) as reason:
            return None, str(reason)
        except (TypeError, ValueError, RuntimeError, IndexError) as error:  # a node we cannot run
            return None, f"a node fails to lower: {type(error).__name__}: {error}"

    def dense_layers(self, max_elements=2**27):
        """(([(W, b)], activation names), None) of the chain in float64, or (None, why)."""
        try:
            return self._lower(max_elements), None
        except _NotAChain as reason:
            return None, str(reason)

    # ----------------------------------------------------------------------------- internals

    def _tensor(self, array):
        """A constant as a torch tensor: floats in the network's dtype, the rest as int64."""
        t = torch.from_numpy(np.array(array, copy=True))
        if t.is_floating_point():
            return t.to(self.device, self.dtype)
        return t.to(self.device, torch.int64) if t.dtype != torch.bool else t.to(self.device)

    def _run(self, op, args, attrs):
        """One node on its input values."""
        handler = getattr(self, "_op_" + op, None)
        if handler is None:
            raise NotImplementedError(f"OnnxNetwork: the operator {op} is not supported")
        return handler(args, attrs)

    # NOLINTBEGIN(naming): the handlers carry the ONNX operator names (_run dispatches on them)
    # element-wise and arithmetic ------------------------------------------------------------
    def _op_Add(self, a, _):
        return a[0] + a[1]

    def _op_Sub(self, a, _):
        return a[0] - a[1]

    def _op_Mul(self, a, _):
        return a[0] * a[1]

    def _op_Div(self, a, _):
        return a[0] / a[1]

    def _op_Neg(self, a, _):
        return -a[0]

    def _op_Relu(self, a, _):
        return torch.relu(a[0])

    def _op_Sigmoid(self, a, _):
        return torch.sigmoid(a[0])

    def _op_Tanh(self, a, _):
        return torch.tanh(a[0])

    def _op_LeakyRelu(self, a, attrs):
        return F.leaky_relu(a[0], attrs.get("alpha", 0.01))

    def _op_Softmax(self, a, attrs):
        return torch.softmax(a[0], dim=attrs.get("axis", -1))

    def _op_Identity(self, a, _):
        return a[0]

    def _op_Dropout(self, a, _):  # inference: the identity
        return a[0]

    def _op_Cast(self, a, attrs):
        from onnx import TensorProto

        to = attrs["to"]
        if to in (TensorProto.FLOAT, TensorProto.DOUBLE, TensorProto.FLOAT16):
            return a[0].to(self.dtype)
        return a[0].to(torch.int64)

    # products ---------------------------------------------------------------------------------
    def _op_MatMul(self, a, _):
        return torch.matmul(a[0], a[1])

    def _op_Gemm(self, a, attrs):
        # Y = alpha A' B' + beta C with A' = A or its transpose, B' the same
        A = a[0].transpose(-1, -2) if attrs.get("transA", 0) else a[0]
        B = a[1].transpose(-1, -2) if attrs.get("transB", 0) else a[1]
        y = A @ B
        if attrs.get("alpha", 1.0) != 1.0:
            y = attrs["alpha"] * y
        if len(a) > 2 and a[2] is not None:
            y = y + (a[2] if attrs.get("beta", 1.0) == 1.0 else attrs["beta"] * a[2])
        return y

    def _op_Conv(self, a, attrs):
        x, w = a[0], a[1]
        bias = a[2] if len(a) > 2 else None
        k = w.dim() - 2
        if attrs.get("auto_pad", b"NOTSET") not in (b"NOTSET", "NOTSET"):
            raise NotImplementedError("OnnxNetwork: Conv with auto_pad")
        pads = list(attrs.get("pads", [0] * (2 * k)))
        symmetric = pads[:k] == pads[k:]
        if not symmetric:
            x = _pool_padding(x, pads, 0.0)
        conv = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[k]
        return conv(
            x,
            w,
            bias,
            stride=attrs.get("strides", 1),
            padding=pads[:k] if symmetric else 0,
            dilation=attrs.get("dilations", 1),
            groups=attrs.get("group", 1),
        )

    def _op_Upsample(self, a, attrs):
        # nearest only: every entry repeated by the integer scales of the spatial axes
        mode = attrs.get("mode", b"nearest")
        if mode not in (b"nearest", "nearest"):
            raise NotImplementedError("OnnxNetwork: Upsample other than nearest")
        scales = [float(v) for v in (a[1].tolist() if len(a) > 1 else attrs["scales"])]
        return F.interpolate(a[0], scale_factor=scales[2:], mode="nearest")

    def _op_ConvTranspose(self, a, attrs):
        x, w = a[0], a[1]
        bias = a[2] if len(a) > 2 else None
        k = w.dim() - 2
        pads = list(attrs.get("pads", [0] * (2 * k)))
        if (
            pads[:k] != pads[k:]
            or attrs.get("output_shape")
            or attrs.get("auto_pad", b"NOTSET")
            not in (
                b"NOTSET",
                "NOTSET",
            )
        ):
            raise NotImplementedError("OnnxNetwork: ConvTranspose with asymmetric pads or a shape")
        conv = {1: F.conv_transpose1d, 2: F.conv_transpose2d, 3: F.conv_transpose3d}[k]
        return conv(
            x,
            w,
            bias,
            stride=attrs.get("strides", 1),
            padding=pads[:k],
            output_padding=attrs.get("output_padding", 0),
            groups=attrs.get("group", 1),
            dilation=attrs.get("dilations", 1),
        )

    def _op_BatchNormalization(self, a, attrs):
        # inference: (x - mean) / sqrt(var + eps) * scale + bias, per channel (axis 1)
        x, scale, bias, mean, var = a[:5]
        shape = [1, -1] + [1] * (x.dim() - 2)
        factor = scale / torch.sqrt(var + attrs.get("epsilon", 1e-5))
        return (x - mean.reshape(shape)) * factor.reshape(shape) + bias.reshape(shape)

    # pooling ------------------------------------------------------------------------------------
    def _op_MaxPool(self, a, attrs):
        x, k = a[0], attrs["kernel_shape"]
        x = _pool_padding(x, list(attrs.get("pads", [])), -math.inf)
        pool = {1: F.max_pool1d, 2: F.max_pool2d, 3: F.max_pool3d}[len(k)]
        return pool(
            x,
            k,
            stride=attrs.get("strides", 1),
            dilation=attrs.get("dilations", 1),
            ceil_mode=bool(attrs.get("ceil_mode", 0)),
        )

    def _op_AveragePool(self, a, attrs):
        x, k = a[0], attrs["kernel_shape"]
        pads = list(attrs.get("pads", []))
        pool = {1: F.avg_pool1d, 2: F.avg_pool2d, 3: F.avg_pool3d}[len(k)]
        if attrs.get("count_include_pad", 0) or not any(pads):
            x = _pool_padding(x, pads, 0.0)
            return pool(
                x, k, stride=attrs.get("strides", 1), ceil_mode=bool(attrs.get("ceil_mode", 0))
            )
        if pads[: len(k)] != pads[len(k) :]:
            raise NotImplementedError("OnnxNetwork: AveragePool with asymmetric pads")
        return pool(
            x,
            k,
            stride=attrs.get("strides", 1),
            padding=pads[: len(k)],
            ceil_mode=bool(attrs.get("ceil_mode", 0)),
            count_include_pad=False,
        )

    def _op_GlobalAveragePool(self, a, _):
        return a[0].mean(dim=tuple(range(2, a[0].dim())), keepdim=True)

    def _op_Pad(self, a, attrs):
        pads = [
            int(p) for p in (a[1].tolist() if len(a) > 1 and a[1] is not None else attrs["pads"])
        ]
        value = float(a[2]) if len(a) > 2 and a[2] is not None else attrs.get("value", 0.0)
        k = len(pads) // 2
        if any(pads[:2]) or any(pads[k : k + 2]):
            raise NotImplementedError("OnnxNetwork: Pad of the batch or channel axis")
        return _pool_padding(a[0], pads[2:k] + pads[k + 2 :], value)

    # shapes ---------------------------------------------------------------------------------------
    def _op_Flatten(self, a, attrs):
        axis = attrs.get("axis", 1)
        return a[0].reshape(math.prod(a[0].shape[:axis]), -1)

    def _op_Reshape(self, a, _):
        shape = [a[0].shape[i] if t == 0 else int(t) for i, t in enumerate(a[1].tolist())]
        return a[0].reshape(shape)

    def _op_Transpose(self, a, attrs):
        return a[0].permute(*attrs.get("perm", reversed(range(a[0].dim()))))

    def _op_Squeeze(self, a, attrs):
        axes = a[1].tolist() if len(a) > 1 and a[1] is not None else attrs.get("axes")
        if axes is None:
            return a[0].squeeze()
        return a[0].squeeze(tuple(ax % a[0].dim() for ax in axes))

    def _op_Unsqueeze(self, a, attrs):
        axes = a[1].tolist() if len(a) > 1 and a[1] is not None else attrs["axes"]
        x = a[0]
        for ax in sorted(ax % (x.dim() + len(axes)) for ax in axes):
            x = x.unsqueeze(ax)
        return x

    def _op_Concat(self, a, attrs):
        return torch.cat(a, dim=attrs["axis"])

    def _op_Shape(self, a, _):
        return torch.tensor(list(a[0].shape), dtype=torch.int64, device=self.device)

    def _op_Gather(self, a, attrs):
        return torch.index_select(a[0], attrs.get("axis", 0), a[1].reshape(-1)).reshape(
            *a[0].shape[: attrs.get("axis", 0)],
            *a[1].shape,
            *a[0].shape[attrs.get("axis", 0) + 1 :],
        )

    def _op_Slice(self, a, attrs):
        starts, ends = a[1].tolist(), a[2].tolist()
        axes = a[3].tolist() if len(a) > 3 and a[3] is not None else list(range(len(starts)))
        steps = a[4].tolist() if len(a) > 4 and a[4] is not None else [1] * len(starts)
        index = [slice(None)] * a[0].dim()
        for s, e, ax, st in zip(starts, ends, axes, steps, strict=True):
            index[ax] = slice(s, min(e, a[0].shape[ax]), st)
        return a[0][tuple(index)]

    # NOLINTEND(naming)

    # lowering to cora.NeuralNetwork ---------------------------------------------------------------
    def _lower(self, max_elements):
        """The dense layers [(W, b)] and activation names of the chain, in float64 on the CPU."""
        net = self.cast(torch.float64, "cpu")
        layers, activations = [], []
        # each value: its tag (index of the chain input it is affine in, None for a constant),
        # its images of the unit vectors of that input (homogeneous) and of zero (the bias row)
        n = self.num_inputs
        if n * n > max_elements:
            raise _NotAChain(f"an input of {n} dimensions exceeds max_elements")
        basis = torch.eye(n, dtype=torch.float64)
        values = {k: (None, v, v) for k, v in net.constants.items()}
        values[self.input_name] = (
            0,
            basis.reshape(n, *self.input_shape),
            torch.zeros(1, *self.input_shape, dtype=torch.float64),
        )
        tag_size = {0: n}
        for op, ins, outs, attrs in net.nodes:
            args = [values[name] if name else (None, None, None) for name in ins]
            tags = {t for t, _, _ in args if t is not None}
            if len(tags) > 1:
                raise _NotAChain(f"{op} joins two layers (a residual connection)")
            tag = tags.pop() if tags else None
            if tag is None or op in STRUCTURE_OPS:
                out = net._run(op, [v for _, v, _ in args], attrs)
                values[outs[0]] = (None, out, out)
                continue
            if op in ACTIVATIONS:
                # the affine map from the chain input `tag` to here is one layer
                _, lin, zero = args[0]
                layers.append(
                    (lin.reshape(tag_size[tag], -1).T.contiguous(), zero.reshape(-1).clone())
                )
                activations.append(ACTIVATIONS[op])
                size = layers[-1][0].shape[0]
                if size * size > max_elements:  # the unit vectors of the next layer
                    raise _NotAChain(f"a layer of {size} neurons exceeds max_elements")
                tag_size[len(layers)] = size
                eye = torch.eye(size, dtype=torch.float64).reshape(size, *lin.shape[1:])
                values[outs[0]] = (len(layers), eye, torch.zeros_like(eye[:1]))
                continue
            if op not in LINEAR_OPS:
                raise _NotAChain(f"{op} is neither affine nor relu/sigmoid/tanh")
            values[outs[0]] = (tag, *net._lower_affine(op, args, attrs))
            if values[outs[0]][1].numel() > max_elements:
                raise _NotAChain(f"{op}: the image of the unit vectors exceeds max_elements")
        # the last affine part: the output layer, without activation
        tag, lin, zero = values[self.output_name]
        if tag is None:
            raise _NotAChain("the output does not depend on the input")
        layers.append((lin.reshape(tag_size[tag], -1).T.contiguous(), zero.reshape(-1).clone()))
        activations.append("identity")
        return layers, activations

    def _lower_affine(self, op, args, attrs):
        """An affine node on (images of the unit vectors, image of zero): the first without the
        additive constants (so the matrix is exact), the second as it is."""
        tagged = [t is not None for t, _, _ in args]
        lin_args = [v for _, v, _ in args]
        if op in ("Mul", "MatMul", "Gemm") and sum(tagged[:2]) > 1:
            raise _NotAChain(f"{op} of two values that depend on the input")
        if op == "Div" and tagged[1]:
            raise _NotAChain("Div by a value that depends on the input")
        zero = self._run_batched(op, [z for _, _, z in args], tagged, attrs)
        if op in ("Add", "Sub"):
            # the constant drops out; Sub(const, x) is -x
            parts = [v if t else None for v, t in zip(lin_args, tagged, strict=True)]
            if op == "Sub" and parts[1] is not None:
                parts[1] = -parts[1]
            lin = sum(p for p in parts if p is not None)
            return lin.expand(lin.shape[0], *zero.shape[1:]), zero
        if op in ("Gemm", "Conv"):
            return self._run_batched(op, lin_args[:2], tagged[:2], attrs), zero
        if op == "BatchNormalization":
            x, scale, bias, mean, var = lin_args[:5]
            homogeneous = [x, scale, torch.zeros_like(bias), torch.zeros_like(mean), var]
            return self._run_batched(op, homogeneous, tagged[:5], attrs), zero
        if op == "Concat":
            # a constant part of the concatenation is zero on the unit vectors
            parts = [v if t else torch.zeros_like(v) for v, t in zip(lin_args, tagged, strict=True)]
            return self._run_batched(op, parts, tagged, attrs), zero
        if op == "Pad":
            return self._run_batched(
                op, lin_args[:2] + [None], tagged[:2] + [False], {**attrs, "value": 0.0}
            ), zero
        return self._run_batched(op, lin_args, tagged, attrs), zero

    # ------------------------------------------------------------------- lowering with convolutions
    def _lower_structured(self):
        """The layers of a graph with convolutions, one per linear node, none made a matrix: a Conv
        (a BatchNormalization after it folded in, else as a depthwise one, as an AveragePool) a
        conv2d layer, a Gemm or MatMul a dense one, a data move (Transpose, Pad, Slice, Concat with
        constants) a gather, found by running the node on the indices; Flatten and Reshape change
        only the shape. A value is a pending sum of maps of earlier layers' outputs (_Expr): an
        activation or the output makes it layers, the longest term as a chain and each other term
        as a residual of the last one (an Add of two branches, as in a ResNet)."""
        net = self.cast(torch.float64, "cpu")
        layers, activations = [], []
        values = dict(net.constants)
        chain = {self.input_name: _Expr(tuple(self.input_shape), [(-1, [])])}
        done = {}  # id of an expression made layers -> the layer of its value


        def emit(maps, source, residuals=()):
            # the maps as layers, the first reading source, the last adding the residuals
            for k, layer in enumerate(maps):
                graph = {"source": source}
                if k == len(maps) - 1 and residuals:
                    graph["residuals"] = [{"source": j, "map": m} for j, m in residuals]
                layers.append(_with_graph(layer, graph))
                activations.append("identity")
                source = len(layers) - 1
            return source

        def materialize(expr):
            # the expression as layers (once); a bare output of a layer (or the input) stays itself
            if len(expr.terms) == 1 and not expr.terms[0][1] and expr.bias is None:
                return expr.terms[0][0]
            if id(expr) in done:
                return done[id(expr)][0]
            size = math.prod(expr.shape[1:])
            terms = sorted(expr.terms, key=lambda t: -len(t[1]))
            (src, maps), others = terms[0], terms[1:]
            residuals = []
            for j, m in others:
                # a longer branch: its first maps as layers, the last as the residual's map
                if len(m) > 1:
                    j = emit(m[:-1], j)
                residuals.append((j, m[-1] if m else None))
            maps = list(maps) or [_gather(list(range(size)), size, torch.zeros(size))]
            if expr.bias is not None:
                maps[-1] = _add_bias(maps[-1], expr.bias)
            done[id(expr)] = (emit(maps, src, residuals), expr)  # (the expression kept alive)
            return done[id(expr)][0]

        def terms(expr):
            # the terms of an expression, one term if it is made layers already
            if id(expr) in done:
                return [(done[id(expr)][0], [])], None
            return expr.terms, expr.bias

        def single(expr):
            # one term without a pending constant, for a map after it
            if len(expr.terms) > 1 or id(expr) in done:
                return _Expr(expr.shape, [(materialize(expr), [])])
            src, maps = expr.terms[0]
            if expr.bias is None:
                return expr
            size = math.prod(expr.shape[1:])
            maps = list(maps) or [_gather(list(range(size)), size, torch.zeros(size))]
            maps[-1] = _add_bias(maps[-1], expr.bias)
            return _Expr(expr.shape, [(src, maps)])

        def max_pool(expr, shape, out_shape, attrs):
            # max(a, b) = b + relu(a - b): the window's items in pairs, round by round; a round is
            # the differences (a grouped convolution) and their ReLU, then a gather of the ReLUs
            # plus the second item of each pair (and an odd last one) as a residual convolution
            c, (kh, kw) = shape[1], attrs["kernel_shape"]
            ho, wo = out_shape[2], out_shape[3]
            pixels = ho * wo
            src, items, first = materialize(single(expr)), kh * kw, True

            def conv(rows, n_in, from_shape, spec):
                # a grouped convolution: row j of `rows` (out, n_in) per channel
                m = torch.as_tensor(rows, dtype=torch.float64)
                k = m.reshape(-1, 1, kh, kw) if first else m.reshape(-1, n_in, 1, 1)
                k = k.repeat(c, 1, 1, 1)
                out = (1, c * m.shape[0], ho, wo)
                return _conv(k, torch.zeros(k.shape[0]), from_shape, spec, out)

            while items > 1 or first:
                pairs, left = items // 2, items % 2
                spec = (
                    {**{a: attrs[a] for a in ("strides", "dilations") if a in attrs}, "group": c}
                    if first
                    else {"group": c}
                )
                from_shape = shape if first else (1, c * items, ho, wo)
                if pairs == 0:  # a 1 x 1 window: the selection only
                    return emit([conv(np.eye(1), items, from_shape, spec)], src)
                diff, pick = np.zeros((pairs, items)), np.zeros((pairs + left, items))
                for j in range(pairs):
                    diff[j, 2 * j], diff[j, 2 * j + 1] = 1.0, -1.0
                    pick[j, 2 * j + 1] = 1.0
                if left:
                    pick[pairs, items - 1] = 1.0
                relu = emit([conv(diff, items, from_shape, spec)], src)
                activations[relu] = "relu"
                # the ReLU of pair j at item j of each channel, 0 at an odd last item
                index = [
                    ((ch * pairs + j) * pixels + p) if j < pairs else -1
                    for ch in range(c)
                    for j in range(pairs + left)
                    for p in range(pixels)
                ]
                size = c * (pairs + left) * pixels
                gather = _gather(index, c * pairs * pixels, torch.zeros(size))
                src = emit([gather], relu, [(src, conv(pick, items, from_shape, spec))])
                items, first = pairs + left, False
            return src

        for op, ins, outs, attrs in net.nodes:
            linked = [name for name in ins if name in chain]
            if not linked:
                values[outs[0]] = net._run(op, [values.get(n) if n else None for n in ins], attrs)
                continue
            if op in STRUCTURE_OPS:
                # shape arithmetic on the chain: its shape only
                probes = [
                    torch.zeros(1, *chain[n].shape[1:], dtype=torch.float64)
                    if n in chain
                    else values.get(n)
                    if n
                    else None
                    for n in ins
                ]
                values[outs[0]] = net._run(op, probes, attrs)
                continue
            if op in ("Add", "Sub") and len(ins) == 2 and len(set(linked)) == 2:
                # two branches: the sum of their terms
                a, b = chain[ins[0]], chain[ins[1]]
                if op == "Sub" or a.shape != b.shape:
                    raise _NotAChain(f"{op} of two branches of other shapes, or a difference")
                (ta, ba), (tb, bb) = terms(a), terms(b)
                bias = ba if bb is None else bb if ba is None else ba + bb
                chain[outs[0]] = _Expr(a.shape, ta + tb, bias)
                continue
            if len(set(linked)) > 1 or len(linked) > 1:
                raise _NotAChain(f"{op} joins two branches")
            x_name = linked[0]
            expr = chain[x_name]
            shape = expr.shape
            probe = torch.zeros(1, *shape[1:], dtype=torch.float64)
            args = [probe if n == x_name else (values.get(n) if n else None) for n in ins]
            out_shape = tuple(net._run(op, args, attrs).shape)
            size = math.prod(shape[1:])

            def then(layer, expr=expr, out_shape=out_shape):
                # the map after the expression
                src, maps = single(expr).terms[0]
                return _Expr(out_shape, [(src, maps + [layer])])

            if op in ACTIVATIONS:
                # in place only on a layer made just now (nothing else reads it yet)
                before = len(layers)
                k = materialize(expr)
                if k < 0 or k < before or activations[k] != "identity":
                    # (a layer's output that others may read keeps its own)
                    k = emit([_gather(list(range(size)), size, torch.zeros(size))], k)
                else:
                    done.pop(id(expr), None)  # the layer's own output is now the activation's
                activations[k] = ACTIVATIONS[op]
                chain[outs[0]] = _Expr(out_shape, [(k, [])])
                continue
            if op == "Conv":
                w = args[1]
                if w.dim() == 3:  # Conv1D as a 2-D convolution of width 1
                    w = w.unsqueeze(-1)
                b = args[2] if len(args) > 2 and args[2] is not None else torch.zeros(w.shape[0])
                chain[outs[0]] = then(_conv(w, b, shape, attrs, out_shape))
            elif op == "ConvTranspose" and len(shape) == 4:
                # the transpose of the convolution from the output back to the input
                w = args[1]
                cout = out_shape[1]
                b = args[2] if len(args) > 2 and args[2] is not None else torch.zeros(cout)
                spec = _conv(w, torch.zeros(w.shape[0]), out_shape, attrs, shape)
                spec["bias"] = torch.broadcast_to(b.reshape(-1, 1, 1), out_shape[1:]).reshape(-1)
                spec["bias"] = spec["bias"].to(torch.float64).contiguous()
                spec["conv_transpose2d"] = spec.pop("conv2d")
                spec["output"] = spec.pop("input")
                chain[outs[0]] = then(spec)
            elif op == "BatchNormalization":
                scale, shift, mean, var = args[1:5]
                factor = scale / torch.sqrt(var + attrs.get("epsilon", 1e-5))
                c = shape[1]
                fused = _fold_batch_norm(single(expr), factor, shift - mean * factor)
                chain[outs[0]] = fused or then(
                    _conv(factor.reshape(c, 1, 1, 1), shift - mean * factor, shape, {"group": c},
                          out_shape)
                )
            elif (
                op == "AveragePool"
                and not attrs.get("ceil_mode", 0)
                and (attrs.get("count_include_pad", 0) or not any(attrs.get("pads", [])))
            ):
                k, c = list(attrs["kernel_shape"]), shape[1]
                w = torch.full((c, 1, *k), 1.0 / math.prod(k), dtype=torch.float64)
                chain[outs[0]] = then(_conv(w, torch.zeros(c), shape, {**attrs, "group": c},
                                            out_shape))
            elif op in ("Gemm", "MatMul"):
                # the matrix of the unit vectors (small: one per input of a dense layer)
                eye = torch.eye(size, dtype=torch.float64).reshape(size, *shape[1:])
                zero = torch.zeros(1, *shape[1:], dtype=torch.float64)
                lin = net._run_batched(op, [eye] + args[1:2], [True, False], attrs)
                bias = net._run(op, [zero] + args[1:], attrs)
                chain[outs[0]] = then((lin.reshape(size, -1).T.contiguous(), bias.reshape(-1)))
            elif op in ("Add", "Sub") and len(ins) == 2:
                other = args[1] if ins[0] == x_name else args[0]
                if op == "Sub" and ins[1] == x_name:
                    raise _NotAChain("Sub of the chain from a constant")
                delta = torch.broadcast_to(other * (-1.0 if op == "Sub" else 1.0), out_shape)
                delta = delta.reshape(-1)
                bias = delta if expr.bias is None else expr.bias + delta
                chain[outs[0]] = _Expr(out_shape, expr.terms, bias)
            elif op in ("Flatten", "Reshape", "Squeeze", "Unsqueeze", "Identity", "Dropout"):
                chain[outs[0]] = _Expr(out_shape, expr.terms, expr.bias)
            elif op in ("Transpose", "Pad", "Slice", "Concat", "Upsample"):
                # the input index + 1 of each output (0: a constant), and the constants
                idx = torch.arange(1, size + 1, dtype=torch.float64).reshape(1, *shape[1:])
                # a Concat's constant parts and a Pad's value are zeros of the moved indices
                moved_args = [idx if n == x_name else values.get(n) if n else None for n in ins]
                if op == "Concat":
                    moved_args = [
                        a if n == x_name else torch.zeros_like(a)
                        for a, n in zip(moved_args, ins, strict=True)
                    ]
                if op == "Pad" and len(moved_args) > 2:
                    moved_args[2] = None
                moved = net._run(op, moved_args, {**attrs, "value": 0.0} if op == "Pad" else attrs)
                const = net._run(op, args, attrs)
                index = (moved.reshape(-1).round().long() - 1).tolist()
                chain[outs[0]] = then(_gather(index, size, const.reshape(-1)))
            elif op == "MaxPool" and len(shape) == 4:
                if attrs.get("ceil_mode", 0) or any(attrs.get("pads", [])):
                    raise _NotAChain("MaxPool with pads or ceil_mode")
                chain[outs[0]] = _Expr(out_shape, [(max_pool(expr, shape, out_shape, attrs), [])])
            else:
                raise _NotAChain(f"{op} on the chain is not a layer yet")
        if self.output_name not in chain:
            raise _NotAChain("the output does not depend on the input")
        out = chain[self.output_name]
        last = materialize(out)
        if last != len(layers) - 1 or last < 0:
            # the output is an earlier layer's or the input: one more identity layer
            size = math.prod(out.shape[1:])
            emit([_gather(list(range(size)), size, torch.zeros(size))], last)
        return layers, activations

    def _run_batched(self, op, args, batched, attrs):
        """One node on per-sample values: the batched arguments mapped by vmap, the rest as is."""
        index = [i for i, b in enumerate(batched) if b and args[i] is not None]

        def one(*samples):
            full = list(args)
            for i, v in zip(index, samples, strict=True):
                full[i] = v
            return self._run(op, full, attrs)

        return torch.func.vmap(one)(*[args[i] for i in index])


class _NotAChain(Exception):
    """The graph is not a chain of affine maps and element-wise activations."""


# -----------------------------------------  AUXILIARY  ------------------------------------------ #

ACTIVATIONS = {"Relu": "relu", "Sigmoid": "sigmoid", "Tanh": "tanh"}  # what cora's layers know

LINEAR_OPS = {
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Neg",
    "MatMul",
    "Gemm",
    "Conv",
    "BatchNormalization",
    "Flatten",
    "Reshape",
    "Transpose",
    "Squeeze",
    "Unsqueeze",
    "Concat",
    "Identity",
    "Dropout",
    "AveragePool",
    "GlobalAveragePool",
    "Pad",
}

STRUCTURED_OPS = {
    "Conv",
    "ConvTranspose",
    "AveragePool",
    "MaxPool",
}  # lowered layer by layer, never to a matrix

STRUCTURE_OPS = {"Shape", "Gather", "Constant", "Cast", "Slice"}  # shape arithmetic, never data


def _conv(w, b, shape, attrs, out_shape):
    """The conv2d dict of cora.NeuralNetwork: a kernel (out, in / groups, kh, kw) on the input
    (1, C, H[, W]) with ONNX's attributes; a Conv1D is a 2-D one of width 1."""
    k = len(shape) - 2
    pads = list(attrs.get("pads", [0] * (2 * k)))
    stride = list(attrs.get("strides", [1] * k))
    dilation = list(attrs.get("dilations", [1] * k))
    if k == 1:
        pads, stride, dilation = [pads[0], 0, pads[1], 0], stride + [1], dilation + [1]
    if attrs.get("auto_pad", b"NOTSET") not in (b"NOTSET", "NOTSET"):
        raise _NotAChain("Conv with auto_pad")
    size = math.prod(out_shape[1:])
    bias = torch.broadcast_to(b.reshape(-1, *([1] * k)), out_shape[1:]).reshape(size)
    return {
        "conv2d": w.reshape(w.shape[0], w.shape[1], w.shape[2], -1).contiguous(),
        "bias": bias.to(torch.float64).contiguous(),
        "input": [shape[1], shape[2], shape[3] if k == 2 else 1],
        "stride": stride,
        "dilation": dilation,
        "pads": pads,
        "groups": int(attrs.get("group", 1)),
    }


def _gather(index, inputs, bias):
    """The gather dict of cora.NeuralNetwork: output i is input index[i] (negative: 0)."""
    return {
        "gather": list(index),
        "inputs": int(inputs),
        "bias": bias.to(torch.float64).contiguous(),
    }


class _Expr:
    """A value of the lowering: the sum of terms (source layer, maps after it; -1: the input) and
    a constant (flat, or None), of the shape (batch first)."""

    def __init__(self, shape, terms, bias=None):
        self.shape, self.terms, self.bias = tuple(shape), list(terms), bias


def _with_graph(layer, graph):
    """A layer as a dict with its graph keys (a dense pair as {'W', 'bias'})."""
    if not isinstance(layer, dict):
        layer = {"W": layer[0], "bias": layer[1]}
    return {**layer, **graph}


def _fold_batch_norm(expr, factor, shift):
    """The expression with a per-channel x * factor + shift folded into its last map, a conv2d
    (kernel rows and bias scaled); None if that map is no convolution."""
    src, maps = expr.terms[0]
    if not maps or not isinstance(maps[-1], dict) or "conv2d" not in maps[-1]:
        return None
    conv = maps[-1]
    c = conv["conv2d"].shape[0]
    if factor.numel() != c:
        return None
    kernel = conv["conv2d"] * factor.reshape(c, 1, 1, 1)
    bias = (conv["bias"].reshape(c, -1) * factor.reshape(c, 1) + shift.reshape(c, 1)).reshape(-1)
    fused = {**conv, "conv2d": kernel.contiguous(), "bias": bias.contiguous()}
    return _Expr(expr.shape, [(src, maps[:-1] + [fused])])


def _add_bias(layer, delta):
    """The layer with delta added to its bias."""
    if isinstance(layer, dict):
        return {**layer, "bias": layer["bias"] + delta}
    return (layer[0], layer[1] + delta)


def _load_model(path):
    """The ModelProto of an .onnx or .onnx.gz file."""
    import onnx

    if str(path).endswith(".gz"):
        with gzip.open(path, "rb") as stream:
            return onnx.load_model_from_string(stream.read())
    return onnx.load(str(path))


def _attributes(node):
    """The attributes of a node as a dict of Python values."""
    from onnx import helper

    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def _dims(value_info):
    """The dimensions of a graph input or output; a symbolic or unset one is 1."""
    return [d.dim_value if d.dim_value > 0 else 1 for d in value_info.type.tensor_type.shape.dim]


def _pool_padding(x, pads, value):
    """ONNX pads (begin..., end...) applied to the spatial axes of x (N, C, *spatial)."""
    if not pads or not any(pads):
        return x
    k = len(pads) // 2
    flat = []
    for i in reversed(range(k)):  # torch's pad lists the last axis first
        flat += [pads[i], pads[i + k]]
    return F.pad(x, flat, value=value)


# ----------------------------------------  END OF CODE  ----------------------------------------- #
