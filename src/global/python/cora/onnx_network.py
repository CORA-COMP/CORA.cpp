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
Last update:   23-November-2026
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
            self.nodes += _split_as_slices(node.op_type, list(node.input), list(node.output),
                                           attrs, self.constants)
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

    def fold_fixed_inputs(self, lo, hi):
        """The network on the inputs of the box [lo, hi] (flat), each value that reads only inputs
        with lo = hi a constant (an input's mask, nn4sys' mscn: then a product with it or a
        quotient by its sum is linear); equal to this one on the box. The values are computed at the
        box's center in float64; which entries read a free input follows each node: a data move
        moves the mask, an element-wise node joins its arguments', any other node makes its whole
        output free."""
        net = self.cast(torch.float64)
        free = torch.as_tensor(np.asarray(hi) > np.asarray(lo)).reshape(self.input_shape)
        center = (np.asarray(lo, dtype=np.float64) + np.asarray(hi, dtype=np.float64)) / 2
        values = dict(net.constants)
        values[self.input_name] = torch.as_tensor(center).reshape(self.input_shape)
        masks = {self.input_name: free}
        nodes, constants = [], dict(self.constants)
        for op, ins, outs, attrs in net.nodes:
            out = net._run(op, [values.get(n) if n else None for n in ins], attrs)
            out = out if isinstance(out, tuple) else (out,)
            if any(bool(masks[n].any()) for n in ins if n in masks):
                out_masks = _masks(net, op, ins, attrs, values, masks, out)
            else:
                out_masks = [torch.zeros(v.shape, dtype=torch.bool) for v in out]
            # an output without a free entry is a constant in the network's dtype; the node stays
            # if it has another output
            for name, value, mask in zip(outs, out, out_masks, strict=False):
                values[name] = value
                if bool(mask.any()):
                    masks[name] = mask
                else:
                    constants[name] = value.to(self.dtype) if value.is_floating_point() else value
            if any(name in masks for name in outs):
                nodes.append((op, ins, [n if n in masks else "" for n in outs], attrs))
        other = object.__new__(type(self))
        other.__dict__.update(self.__dict__)
        other.nodes, other.constants = nodes, constants
        return other

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

    def layers(self, max_elements=2**27, drop_output_softmax=False):
        """(([layer], activation names), None) of the network in float64, or (None, why): a chain
        without a Conv or an AveragePool as dense layers; else one layer per linear node
        (cora.NeuralNetwork's conv2d, gather and dense dicts, never a matrix of a convolution),
        with the graph keys of a residual connection. drop_output_softmax lowers a Softmax that
        makes the output as the identity (it keeps the order of the outputs: a property that only
        compares outputs holds for the logits iff for the probabilities); dropped_softmax tells."""
        self.dropped_softmax = False
        last = self.nodes[-1] if self.nodes else None
        softmax = last and last[0] == "Softmax" and last[2][0] == self.output_name
        if drop_output_softmax and softmax:
            # the logits as the output, then the graph as it is
            nodes, output = self.nodes, self.output_name
            self.nodes, self.output_name, self.dropped_softmax = nodes[:-1], last[1][0], True
            try:
                return self.layers(max_elements)
            finally:
                self.nodes, self.output_name, self.dropped_softmax = nodes, output, True
        if not any(op in STRUCTURED_OPS for op, _, _, _ in self.nodes):
            dense, why = self.dense_layers(max_elements)
            # a graph (two branches joined, a product of two values) goes layer by layer
            if dense or not any(
                r in why
                for r in ("joins two", "of two values", "fails to lower", "Softmax", "max_elements",
                          "Reduce", "Div by a value")
            ):
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
        except (TypeError, ValueError, RuntimeError, IndexError) as error:  # a node we cannot run
            return None, f"a node fails to lower: {type(error).__name__}: {error}"

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

    def _op_Sign(self, a, _):
        return torch.sign(a[0])

    def _op_Exp(self, a, _):
        return torch.exp(a[0])

    def _op_Floor(self, a, _):
        return torch.floor(a[0])

    def _op_Pow(self, a, _):
        return torch.pow(a[0], a[1])

    def _op_Square(self, a, _):  # Pow(x, 2) as the loader names it (_split_as_slices)
        return a[0] * a[0]

    def _op_Cube(self, a, _):  # Pow(x, 3) as the loader names it (_split_as_slices)
        return a[0] * a[0] * a[0]

    def _op_Sin(self, a, _):
        return torch.sin(a[0])

    def _op_Cos(self, a, _):
        return torch.cos(a[0])

    def _op_Equal(self, a, _):
        return torch.eq(a[0], a[1])

    def _op_Where(self, a, _):
        return torch.where(a[0], a[1], a[2])

    def _op_Expand(self, a, _):
        # ONNX's Expand broadcasts both ways: the shape's 1s keep the value's sizes
        shape = [int(v) for v in a[1].tolist()]
        return a[0] * torch.ones(shape, dtype=a[0].dtype, device=a[0].device)

    def _op_ReduceSum(self, a, attrs):
        # the axes from the input (opset 13) or the attribute; none: all (or noop_with_empty_axes)
        axes = a[1] if len(a) > 1 and a[1] is not None else attrs.get("axes")
        keep = bool(attrs.get("keepdims", 1))
        if axes is None or len(axes) == 0:
            if attrs.get("noop_with_empty_axes", 0):
                return a[0]
            return a[0].sum(dim=tuple(range(a[0].dim())), keepdim=keep)
        return a[0].sum(dim=tuple(int(x) for x in axes), keepdim=keep)

    def _op_ReduceMean(self, a, attrs):
        # the mean over the axes (input from opset 18, else the attribute); none: all
        axes = a[1] if len(a) > 1 and a[1] is not None else attrs.get("axes")
        keep = bool(attrs.get("keepdims", 1))
        if axes is None or len(axes) == 0:
            if attrs.get("noop_with_empty_axes", 0):
                return a[0]
            return a[0].mean(dim=tuple(range(a[0].dim())), keepdim=keep)
        return a[0].mean(dim=tuple(int(x) for x in axes), keepdim=keep)

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

    def _op_ConstantOfShape(self, a, attrs):
        # a tensor of the shape a[0], filled with the value's one entry (default: float 0)
        from onnx import numpy_helper

        value = attrs.get("value")
        fill = numpy_helper.to_array(value).reshape(-1)[0] if value is not None else np.float32(0)
        return self._tensor(np.full([int(d) for d in a[0].tolist()], fill))

    def _op_Shape(self, a, _):
        return torch.tensor(list(a[0].shape), dtype=torch.int64, device=self.device)

    def _op_Gather(self, a, attrs):
        # (a scalar index drops the axis: the shape as a tuple, which may be empty; a negative
        # index counts from the end)
        axis = attrs.get("axis", 0) % a[0].dim()
        index = a[1].reshape(-1) % a[0].shape[axis]
        return torch.index_select(a[0], axis, index).reshape(
            (*a[0].shape[:axis], *a[1].shape, *a[0].shape[axis + 1 :])
        )

    def _op_Slice(self, a, attrs):
        # the bounds from the inputs (opset 10), else from the attributes (opset 1)
        if len(a) > 2:
            starts, ends = a[1].tolist(), a[2].tolist()
            axes = a[3].tolist() if len(a) > 3 and a[3] is not None else None
            steps = a[4].tolist() if len(a) > 4 and a[4] is not None else [1] * len(starts)
        else:
            starts, ends = list(attrs["starts"]), list(attrs["ends"])
            axes, steps = attrs.get("axes"), [1] * len(starts)
        axes = list(range(len(starts))) if axes is None else list(axes)
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
            if tag is None or (op in STRUCTURE_OPS and not _moves_data(op, args[0][0] is not None)):
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
            if op not in LINEAR_OPS | DATA_MOVES:
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
        given = tuple(self.input_shape)
        unbatched = len(given) < 2 or given[0] != 1  # no batch axis (nn4sys' [12, 8])
        chain = {self.input_name: _Expr((1, *given) if unbatched else given, [(-1, [])])}
        chain[self.input_name].unbatched = unbatched
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

        def activate(expr, activation):
            # the expression as a layer with the activation (a fresh layer: no other reader)
            before = len(layers)
            k = materialize(expr)
            if k < 0 or k < before or activations[k] != "identity":
                size = math.prod(expr.shape[1:])
                k = emit([_gather(list(range(size)), size, torch.zeros(size))], k)
            else:
                done.pop(id(expr), None)
            activations[k] = activation
            return k

        def products(a, b, ia, ib, out_shape):
            # the sums over t of a[ia[t, o]] b[ib[t, o]] (index lists (K, O)): the factors [a; b]
            # of every product by two gathers, a product layer ([a o b; 0]), then its first half
            # summed over t by a 1 x 1 convolution
            K, O = ia.shape
            ka, kb = materialize(single(a)), materialize(single(b))
            na, nb = math.prod(a.shape[1:]), math.prod(b.shape[1:])
            none = torch.full((K * O,), -1, dtype=torch.int64)
            ga = _gather(torch.cat([ia.reshape(-1), none]).tolist(), na, torch.zeros(2 * K * O))
            gb = _gather(torch.cat([none, ib.reshape(-1)]).tolist(), nb, torch.zeros(2 * K * O))
            k = activate(_Expr((1, 2 * K * O), [(ka, [ga]), (kb, [gb])]), "product")
            w = torch.cat([torch.ones(K), torch.zeros(K)]).to(torch.float64).reshape(1, 2 * K, 1, 1)
            total = _conv(w, torch.zeros(1), (1, 2 * K, O, 1), {}, (1, 1, O, 1))
            return _Expr(out_shape, [(k, [total])])

        def softmax(expr, axis):
            # 1 / sum_t exp(x_t - x_q) along the axis: the differences by two gathers (one times
            # -1), exp, their sums by a 1 x 1 convolution, then 1 / max(s, 1) (s >= 1: t = q)
            shape = expr.shape
            n, M = math.prod(shape[1:]), shape[axis]
            idx = torch.arange(n).reshape(shape[1:])
            moved = idx.movedim(axis - 1, -1)  # (..., M): the entries along the axis
            # it[t, q]: the entry t of q's row; iq[t, q] = q
            rows = moved.unsqueeze(-2).expand(*moved.shape, M)  # (..., q, t)
            it = torch.empty(M, n, dtype=torch.int64)
            it[:, moved.reshape(-1)] = rows.reshape(-1, M).T
            iq = torch.arange(n).expand(M, n)
            kx = materialize(single(expr))
            zero = torch.zeros(M * n)
            minus = torch.full((M * n,), -1.0, dtype=torch.float64)
            dt = _gather(it.reshape(-1).tolist(), n, zero)
            dq = _gather(iq.reshape(-1).tolist(), n, zero, minus)
            ke = activate(_Expr((1, M * n), [(kx, [dt]), (kx, [dq])]), "exp")
            w = torch.ones(1, M, 1, 1, dtype=torch.float64)
            total = _conv(w, torch.zeros(1), (1, M, n, 1), {}, (1, 1, n, 1))
            return _Expr(shape, [(activate(_Expr((1, n), [(ke, [total])]), "reciprocal"), [])])

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
            if any(getattr(chain[n], "unbatched", False) for n in linked):
                # an input without a batch axis: a Reshape to a batch of 1 reads it, nothing else
                if op != "Reshape" or ins[0] not in chain or ins[1] in chain:
                    raise _NotAChain(f"{op} of an input without a batch axis (only a Reshape)")
                probe = torch.zeros(self.input_shape, dtype=torch.float64)
                out_shape = tuple(net._run(op, [probe, values[ins[1]]], attrs).shape)
                if len(out_shape) < 2 or out_shape[0] != 1:
                    raise _NotAChain("a Reshape of an input without a batch axis to no batch of 1")
                chain[outs[0]] = _Expr(out_shape, chain[ins[0]].terms, chain[ins[0]].bias)
                continue
            if any(getattr(chain[n], "column", False) for n in linked) and (
                len(set(linked)) > 1 or op in STRUCTURE_OPS
            ):
                raise _NotAChain(f"{op} of a column (the batch axis transposed) and more")
            if op in STRUCTURE_OPS and not _moves_data(op, ins[0] in chain):
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
            if op == "Concat" and len(set(linked)) > 1:
                # branches side by side: each term padded into its place, the constants as bias
                shapes = [chain[n].shape if n in chain else tuple(values[n].shape) for n in ins]

                def probes(i, part, shapes=shapes):
                    # zeros of every part's shape, `part` at the i-th
                    zs = [torch.zeros(1, *sh[1:], dtype=torch.float64) for sh in shapes]
                    zs[i] = part
                    return zs

                pieces, bias = [], None
                for i, n in enumerate(ins):
                    if n not in chain:
                        moved = net._run(op, probes(i, values[n].to(torch.float64)), attrs)
                        bias = moved.reshape(-1) if bias is None else bias + moved.reshape(-1)
                        continue
                    # the place of each entry of this part in the output (-1: another part's)
                    size_i = math.prod(shapes[i][1:])
                    idx = torch.arange(1, size_i + 1, dtype=torch.float64)
                    idx = idx.reshape(1, *shapes[i][1:])
                    index = (net._run(op, probes(i, idx), attrs).reshape(-1).round().long() - 1)
                    place = _gather(index.tolist(), size_i, torch.zeros(len(index)))
                    t, b = terms(chain[n])
                    pieces += [(src, list(maps) + [place]) for src, maps in t]
                    if b is not None:
                        moved = torch.cat([b.to(torch.float64), torch.zeros(1)])[index]
                        bias = moved if bias is None else bias + moved
                out_shape = tuple(net._run(op, probes(0, torch.zeros(1, *shapes[0][1:],
                                                                     dtype=torch.float64)),
                                           attrs).shape)
                chain[outs[0]] = _Expr(out_shape, pieces, bias)
                continue
            if op in ("Add", "Sub") and len(ins) == 2 and len(set(linked)) == 2:
                # two branches: the sum of their terms (a difference: b's through -I)
                a, b = chain[ins[0]], chain[ins[1]]
                if a.shape != b.shape:
                    raise _NotAChain(f"{op} of two branches of other shapes")
                (ta, ba), (tb, bb) = terms(a), terms(b)
                if op == "Sub":
                    tb, bb = _scaled_terms(tb, bb, -torch.eye(math.prod(b.shape[1:]),
                                                              dtype=torch.float64))
                bias = ba if bb is None else bb if ba is None else ba + bb
                chain[outs[0]] = _Expr(a.shape, ta + tb, bias)
                continue
            if op == "MatMul" and len(ins) == 2 and len(linked) == 2:
                # a product of two values (attention): A (..., n, K) times B (..., K, m)
                a, b = chain[ins[0]], chain[ins[1]]
                ia = torch.arange(math.prod(a.shape)).reshape(a.shape).unsqueeze(-1)
                ib = torch.arange(math.prod(b.shape)).reshape(b.shape).unsqueeze(-3)
                ia, ib = torch.broadcast_tensors(ia, ib)  # (..., n, K, m)
                out_shape = tuple(ia.shape[:-2]) + (ia.shape[-1],)
                if out_shape[0] != 1:
                    raise _NotAChain("MatMul of two values with a batch of other sizes")
                K = ia.shape[-2]
                chain[outs[0]] = products(
                    a, b, ia.movedim(-2, 0).reshape(K, -1), ib.movedim(-2, 0).reshape(K, -1),
                    out_shape,
                )
                continue
            if op == "Softmax" and len(linked) == 1:
                # a softmax inside the network (the output's may be dropped, layers())
                expr = chain[ins[0]]
                axis = attrs.get("axis", -1 if net.opset >= 13 else 1) % len(expr.shape)
                if net.opset < 13 and axis != len(expr.shape) - 1:
                    raise _NotAChain("Softmax before opset 13 over more than the last axis")
                if axis == 0:
                    raise _NotAChain("Softmax over the batch axis")
                chain[outs[0]] = softmax(expr, axis)
                continue
            if op == "Div" and len(ins) == 2 and len(set(linked)) == 2:
                # a quotient of two values: a times 1 / b (b > 0 in verify, nn4sys' x^3 / sum x^3),
                # b broadcast to a's entries (a product layer of gathers)
                a, b = chain[ins[0]], chain[ins[1]]
                inverse = _Expr(b.shape, [(activate(b, "inverse"), [])])
                ia = torch.arange(math.prod(a.shape)).reshape(a.shape)
                ib = torch.arange(math.prod(b.shape)).reshape(b.shape)
                ia, ib = torch.broadcast_tensors(ia, ib)
                if tuple(ia.shape) != tuple(a.shape):
                    raise _NotAChain("Div of two values that broadcasts the dividend")
                chain[outs[0]] = products(a, inverse, ia.reshape(1, -1), ib.reshape(1, -1),
                                          tuple(a.shape))
                continue
            if op == "Mul" and len(ins) == 2 and len(linked) == 2:
                # a product of two values: a product layer of [a; b] ([a o b; 0]), its first half
                a, b = chain[ins[0]], chain[ins[1]]
                if a.shape != b.shape:
                    raise _NotAChain("Mul of two values of other shapes")
                n = math.prod(a.shape[1:])
                if n > MAX_PRODUCT:
                    raise _NotAChain(f"Mul of two values of {n} entries exceeds {MAX_PRODUCT}")
                eye = torch.eye(n, dtype=torch.float64)
                zero_n = torch.zeros(n, n, dtype=torch.float64)
                (ta, ba), (tb, bb) = terms(a), terms(b)
                ta, ba = _scaled_terms(ta, ba, torch.cat([eye, zero_n]))
                tb, bb = _scaled_terms(tb, bb, torch.cat([zero_n, eye]))
                bias = ba if bb is None else bb if ba is None else ba + bb
                k = activate(_Expr((1, 2 * n), ta + tb, bias), "product")
                first = torch.cat([torch.eye(n), torch.zeros(n, n)], 1).to(torch.float64)
                zero = torch.zeros(n, dtype=torch.float64)
                chain[outs[0]] = _Expr(a.shape, [(k, [(first, zero)])])
                continue
            if len(set(linked)) > 1 or len(linked) > 1:
                raise _NotAChain(f"{op} joins two branches")
            x_name = linked[0]
            expr = chain[x_name]
            shape = expr.shape
            column = getattr(expr, "column", False)
            if op == "Transpose" and len(shape) == 2 and (column or shape[0] == 1):
                # a row (1, n) as a column (n, 1), or back: the same entries, a flag (ml4acopf)
                flipped = _Expr(shape, expr.terms, expr.bias)
                flipped.column = not column
                chain[outs[0]] = flipped
                continue
            if column:
                # a column takes only a constant matrix from the left: C (m, n) x as a dense layer
                if op != "MatMul" or ins[1] != x_name or ins[0] not in values:
                    raise _NotAChain(f"{op} of a column (the batch axis transposed)")
                C = values[ins[0]].to(torch.float64)
                src, maps = single(expr).terms[0]
                zero = torch.zeros(C.shape[0], dtype=torch.float64)
                product = _Expr((1, C.shape[0]), [(src, maps + [(C.contiguous(), zero)])])
                product.column = True
                chain[outs[0]] = product
                continue
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
                c, offset = shape[1], shift - mean * factor
                fused = _fold_batch_norm(single(expr), factor, offset)
                if fused is None and len(shape) == 2:
                    # of features (after a Gemm): a diagonal dense layer
                    fused = then((torch.diag(factor).contiguous(), offset.contiguous()))
                if fused is None and len(shape) == 3:
                    # of a sequence (C, L) (a transformer's tokens): a gather with factors
                    per = [1, c, 1]
                    f = torch.broadcast_to(factor.reshape(per), (1, *shape[1:])).reshape(-1)
                    o = torch.broadcast_to(offset.reshape(per), (1, *shape[1:])).reshape(-1)
                    fused = then(_gather(list(range(size)), size, o.contiguous(), f.contiguous()))
                chain[outs[0]] = fused or then(
                    _conv(factor.reshape(c, 1, 1, 1), offset, shape, {"group": c}, out_shape)
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
            elif op in ("Mul", "Div", "Neg", "Cast"):
                # an element-wise scaling by constants: folded into the map before, else a layer
                if op == "Div" and ins[0] != x_name:
                    raise _NotAChain("Div by a value that depends on the input")
                at = ins.index(x_name)
                full = lambda v, at=at, args=args: [  # noqa: E731
                    v if i == at else a for i, a in enumerate(args)
                ]
                factor = net._run(op, full(torch.ones_like(probe)), attrs)
                if tuple(factor.shape) != tuple(probe.shape):
                    raise _NotAChain(f"{op} broadcasts the value to another shape")
                chain[outs[0]] = _scale(expr, single, then, factor.reshape(-1), shape, out_shape)
            elif op in ("ReduceSum", "ReduceMean"):
                # a sum or a mean over axes (the batch axis of one row dropped): the matrix of
                # ones, or of 1 / count
                axes = args[1] if len(args) > 1 and args[1] is not None else attrs.get("axes", [])
                axes = [int(a) % len(shape) for a in axes] or list(range(1, len(shape)))
                dims = tuple(a for a in axes if a)
                keep = bool(attrs.get("keepdims", 1))
                rows = math.prod(out_shape[1:])
                mean = op == "ReduceMean"
                lin = _chunked_images(
                    lambda unit, dims=dims, keep=keep, mean=mean: (
                        unit.mean(dim=dims, keepdim=keep) if mean else
                        unit.sum(dim=dims, keepdim=keep)
                    )
                    if dims
                    else unit,
                    size, shape, rows,
                )
                chain[outs[0]] = then((lin, torch.zeros(rows, dtype=torch.float64)))
            elif op in ("Gemm", "MatMul"):
                if ins[0] != x_name:
                    raise _NotAChain(f"{op} of a constant times the value")
                # the matrix: the images of the unit vectors, a chunk at a time (after a wide
                # convolution, as malbeware's 61504 entries, the identity alone is 30 GB)
                zero = torch.zeros(1, *shape[1:], dtype=torch.float64)
                bias = net._run(op, [zero] + args[1:], attrs)
                # a row times a constant matrix: its weights (vggnet16's 25088 x 4096 at once)
                lin = _constant_matrix(op, args, attrs, shape)
                if lin is None:
                    lin = _chunked_images(
                        lambda unit, op=op, args=args, attrs=attrs: net._run_batched(
                            op, [unit] + args[1:2], [True, False], attrs
                        ),
                        size, shape, bias.numel(),
                    )
                chain[outs[0]] = then((lin, bias.reshape(-1)))
            elif op in ("Add", "Sub") and len(ins) == 2:
                other = args[1] if ins[0] == x_name else args[0]
                if math.prod(out_shape[1:]) != size:
                    # the constant broadcasts the value to more entries (ml4acopf's x - t_k for
                    # every threshold): the value repeated by a gather first
                    index = torch.broadcast_to(torch.arange(size).reshape(shape), out_shape)
                    expr = then(_gather(index.reshape(-1).tolist(), size,
                                        torch.zeros(index.numel())))
                    shape, size = out_shape, index.numel()
                if op == "Sub" and ins[1] == x_name:
                    # a constant minus the value: the value negated, then the constant added
                    expr = _scale(expr, single, then, -torch.ones(size, dtype=torch.float64),
                                  shape, out_shape)
                    other = -other
                delta = torch.broadcast_to(other * (-1.0 if op == "Sub" else 1.0), out_shape)
                delta = delta.reshape(-1)
                bias = delta if expr.bias is None else expr.bias + delta
                chain[outs[0]] = _Expr(out_shape, expr.terms, bias)
            elif op in ("Flatten", "Reshape", "Squeeze", "Unsqueeze", "Identity", "Dropout"):
                chain[outs[0]] = _Expr(out_shape, expr.terms, expr.bias)
            elif op in ("Transpose", "Pad", "Slice", "Concat", "Upsample", "Gather"):
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
        if getattr(out, "column", False):
            raise _NotAChain("the output is a column (the batch axis transposed)")
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

MAX_PRODUCT = 4096  # entries of a product of two values at most (its maps are dense)

UNIT_VECTOR_CHUNK = 2**22  # entries of unit vectors per batch of a Gemm's or a MatMul's matrix

ACTIVATIONS = {"Relu": "relu", "Sigmoid": "sigmoid", "Tanh": "tanh", "Sign": "sign", "Exp": "exp",
               "Floor": "floor", "Square": "square", "Cube": "cube", "Sin": "sin", "Cos": "cos"}

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

MASK_MOVES = {"Slice", "Gather", "Reshape", "Transpose", "Concat", "Squeeze", "Unsqueeze",
              "Flatten", "Identity", "Dropout", "Pad"}  # fold_fixed_inputs: data moves

MASK_ELEMENTWISE = {"Add", "Sub", "Mul", "Div", "Neg", "Relu", "Sigmoid", "Tanh", "LeakyRelu",
                    "Sign", "Exp", "Cast", "Floor", "Square", "Cube", "Pow", "Sin",
                    "Cos"}  # fold_fixed_inputs: entry by entry

STRUCTURE_OPS = {"Shape", "Gather", "Constant", "Cast", "Slice"}  # shape arithmetic, never data
DATA_MOVES = {"Slice", "Gather", "Cast"}  # ... unless they move a value of the input: then linear


def _moves_data(op, on_data):
    """Whether a STRUCTURE_OPS node moves the entries of a value of the input (its first argument
    is one): a Slice, Gather or Cast of data is a linear map, not shape arithmetic."""
    return op in DATA_MOVES and bool(on_data)


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


def _scaled_terms(terms, bias, M):
    """The terms and the constant of an expression through the dense map M (out, n)."""
    zero = torch.zeros(M.shape[0], dtype=torch.float64)
    terms = [(src, list(maps) + [(M.contiguous(), zero)]) for src, maps in terms]
    return terms, (None if bias is None else M @ bias.to(torch.float64))


def _constant_matrix(op, args, attrs, shape):
    """The matrix (out, in) of a row (1, in) times a constant matrix, a Gemm's alpha B' or a
    MatMul's B, transposed: the images of the unit vectors without computing them (bit for bit
    the same: each image is a row of B' times 1 plus zeros); None for another shape or transA."""
    B = args[1]
    if len(shape) != 2 or shape[0] != 1 or B is None or B.dim() != 2:
        return None
    if op == "Gemm":
        if attrs.get("transA", 0):
            return None
        B = B.T if attrs.get("transB", 0) else B
        alpha = attrs.get("alpha", 1.0)
        B = alpha * B if alpha != 1.0 else B
    if not torch.isfinite(B).all():  # 0 * inf of the unit vectors is nan, not the weight
        return None
    return B.to(torch.float64).T.contiguous()


def _chunked_images(image, size, shape, rows):
    """The matrix (rows, size) of a linear map from the images of the unit vectors of a value of
    the shape (batch 1 first), a chunk of UNIT_VECTOR_CHUNK entries at a time: image maps a batch
    (count, *shape[1:]) to (count, ...). (One buffer of unit vectors and the matrix allocated
    once: chunks allocated anew fragment the heap, and the peak grows with every chunk.)"""
    chunk = min(size, max(1, UNIT_VECTOR_CHUNK // size))
    eye = torch.zeros(chunk, size, dtype=torch.float64)
    lin = torch.empty(rows, size, dtype=torch.float64)
    for start in range(0, size, chunk):
        count = min(chunk, size - start)
        unit = eye[:count]
        unit.zero_()
        unit[:, start : start + count].fill_diagonal_(1.0)
        lin[:, start : start + count] = image(unit.reshape(count, *shape[1:])).reshape(count, -1).T
    return lin


def _scale(expr, single, then, factor, shape, out_shape):
    """The expression times the factor of each entry: folded into the map before it (a dense
    layer's rows, a convolution's kernel per channel), else a depthwise convolution of an image
    or a diagonal dense layer of at most MAX_PRODUCT entries."""
    zero = torch.zeros(factor.numel(), dtype=torch.float64)
    if len(shape) == 4:
        per = factor.reshape(shape[1], -1)
        if bool((per == per[:, :1]).all()):
            c, f = shape[1], per[:, 0].contiguous()
            fused = _fold_batch_norm(single(expr), f, torch.zeros(c, dtype=torch.float64))
            return fused or then(
                _conv(f.reshape(c, 1, 1, 1), torch.zeros(c), shape, {"group": c}, out_shape)
            )
    fused = _fold_batch_norm(single(expr), factor, zero)
    if fused is not None:
        return fused
    if factor.numel() > MAX_PRODUCT:
        # many entries (mscn_2048d's 12288): a gather with factors, never a dense diagonal
        n = factor.numel()
        return then(_gather(list(range(n)), n, zero, factor.to(torch.float64)))
    return then((torch.diag(factor).contiguous(), zero))


def _split_as_slices(op, ins, outs, attrs, constants):
    """A node as the nodes of the graph: a Split of known sizes (its attribute, or a constant
    input) one Slice per output, which the lowering moves like any other; Pow(x, 2) a Square
    and Pow(x, 3) a Cube (the activations x^2, x^3); the rest as it is."""
    if op == "Pow" and len(ins) == 2 and ins[1] in constants:
        exponent = constants[ins[1]]
        powers = {2.0: "Square", 3.0: "Cube"}
        if exponent.numel() == 1 and float(exponent) in powers:
            return [(powers[float(exponent)], ins[:1], outs, {})]
    sizes = attrs.get("split")
    if op == "Split" and sizes is None and len(ins) > 1 and ins[1] in constants:
        sizes = [int(v) for v in constants[ins[1]].tolist()]
    if op != "Split" or sizes is None:
        return [(op, ins, outs, attrs)]
    axis, at, nodes = attrs.get("axis", 0), 0, []
    for name, size in zip(outs, sizes, strict=True):
        nodes.append(("Slice", ins[:1], [name], {"starts": [at], "ends": [at + int(size)],
                                                  "axes": [axis]}))
        at += int(size)
    return nodes


def _masks(net, op, ins, attrs, values, masks, out):
    """Which entries of a node's outputs read a free input (fold_fixed_inputs): a data move
    moves its argument's mask, an element-wise node joins its arguments' (broadcast), any other
    node makes every output entry free."""
    if op in MASK_MOVES:
        # the node on the masks as numbers (a constant argument: no entry free)
        def as_mask(n):
            if n in masks:
                return masks[n].to(torch.float64)
            v = values.get(n) if n else None
            return torch.zeros_like(v) if op == "Concat" and v is not None else v

        moved = net._run(op, [as_mask(n) if n else None for n in ins], attrs)
        return [m > 0 for m in (moved if isinstance(moved, tuple) else (moved,))]
    if op in MASK_ELEMENTWISE:
        joined = torch.zeros(out[0].shape, dtype=torch.bool)
        for n in ins:
            if n in masks:
                joined = joined | masks[n]
        return [joined]
    return [torch.ones(v.shape, dtype=torch.bool) for v in out]


def _gather(index, inputs, bias, scale=None):
    """The gather dict of cora.NeuralNetwork: output i is input index[i] (negative: 0), times
    scale[i] if given."""
    out = {
        "gather": list(index),
        "inputs": int(inputs),
        "bias": bias.to(torch.float64).contiguous(),
    }
    if scale is not None:
        out["scale"] = scale.to(torch.float64).contiguous()
    return out


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
    """The expression with a per-channel x * factor + shift folded into its last map, a conv2d or
    a dense pair (kernel or matrix rows and bias scaled); None if that map is neither."""
    src, maps = expr.terms[0]
    if maps and isinstance(maps[-1], tuple) and maps[-1][0].shape[0] == factor.numel():
        # a dense layer's rows: W' = diag(factor) W, b' = factor b + shift
        W, b = maps[-1]
        fused = ((W * factor.reshape(-1, 1)).contiguous(), (b * factor + shift).contiguous())
        return _Expr(expr.shape, [(src, maps[:-1] + [fused])])
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
