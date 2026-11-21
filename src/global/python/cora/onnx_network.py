"""onnx_network - an ONNX network as a batched torch function, and as a cora.NeuralNetwork

The graph of an ONNX file runs node by node on torch tensors with a leading batch axis, on any
device and in any float dtype (float32 for the search, float64 for re-checks). Points are rows
(B, n) of the flattened input (C order), outputs rows (B, m). A graph that is a chain of affine
maps and element-wise activations lowers to a cora.NeuralNetwork: each affine part becomes one
dense linear layer, its matrix the image of the unit vectors with the additive constants dropped
(so a plain Gemm keeps its weights bit for bit), its bias the image of zero.

Syntax:   net = OnnxNetwork.load("net.onnx", dtype=torch.float32, device="cpu")
          y = net(x)                       x (B, n) -> y (B, m), batched, autograd on
          net64 = net.cast(torch.float64)  the same graph in another dtype or on another device
          nn, why = net.to_neural_network()   a cora.NeuralNetwork, or None and the reason
          (layers, acts), why = net.dense_layers()   the same as [(W, b)] and activation names
Inputs:   path - an .onnx or .onnx.gz file with one input and one output
See also: cora.vnnlib, cora.nn_verify, docs/nn_verification.md

Authors:       Claude Opus 5.5
Written:       21-November-2026
Last update:   ---
Reviewed:      ---
"""

import gzip
import math

import numpy as np
import torch
import torch.nn.functional as F


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

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
STRUCTURE_OPS = {"Shape", "Gather", "Constant", "Cast", "Slice"}  # shape arithmetic, never data


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


# ============================================  MAIN  ============================================ #


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

        lowered, why = self.dense_layers(max_elements)
        return (cora.NeuralNetwork(*lowered), None) if lowered else (None, why)

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


# ----------------------------------------  END OF CODE  ----------------------------------------- #
