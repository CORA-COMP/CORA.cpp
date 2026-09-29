"""Backend-agnostic array constructors: the Python side of the C++ `Tensor`.

    cora.Tensor([[1, 2], [3, 4]])          # torch on the torch backend, numpy on eigen
    cora.zeros(3, 2, device="gpu")         # like Tensor::zeros(size, device)
"""
import numpy as np

from ._cora import backend


def _torch():
    if not backend().startswith("torch"):
        return None
    import torch
    return torch


_NAMES = {"float64": "float64", "double": "float64", "float32": "float32", "float": "float32"}


def _dtype(dtype):
    """The name of a dtype given as a name ("float64", "float32"), a numpy or a torch dtype."""
    name = str(getattr(dtype, "__name__", dtype)).replace("torch.", "")
    if name not in _NAMES:
        raise ValueError(f"cora: unknown dtype {dtype!r}; use 'float64' or 'float32'")
    return _NAMES[name]


def _device(device):
    return "cuda" if device == "gpu" else device


def Tensor(data, dtype="float64", device=None, requires_grad=False):
    """Array of `data` on the current backend; `dtype` is "float64" or "float32", `device`
    ("cpu"/"gpu"/"cuda:N") and `requires_grad` apply to torch only."""
    torch = _torch()
    if torch is None:
        if device not in (None, "cpu") or requires_grad:
            raise ValueError("cora.Tensor: the eigen backend is CPU-only and has no gradients")
        return np.array(data, dtype=_dtype(dtype))
    dtype = getattr(torch, _dtype(dtype))
    return torch.as_tensor(data, dtype=dtype, device=_device(device)).requires_grad_(requires_grad)


def _make(fill, shape, dtype, device):
    torch = _torch()
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    if torch is None:
        if device not in (None, "cpu"):
            raise ValueError("cora: the eigen backend is CPU-only")
        return getattr(np, fill)(shape, dtype=_dtype(dtype))
    return getattr(torch, fill)(shape, dtype=getattr(torch, _dtype(dtype)), device=_device(device))


def zeros(*shape, dtype="float64", device=None):
    """Zeros of the given shape, e.g. zeros(3, 2) or zeros((3, 2))."""
    return _make("zeros", shape, dtype, device)


def ones(*shape, dtype="float64", device=None):
    """Ones of the given shape."""
    return _make("ones", shape, dtype, device)


def eye(n, dtype="float64", device=None):
    """The n-by-n identity."""
    torch = _torch()
    if torch is None:
        return np.eye(n, dtype=_dtype(dtype))
    return torch.eye(n, dtype=getattr(torch, _dtype(dtype)), device=_device(device))


def randn(*shape, dtype="float64", device=None, seed=None):
    """Standard-normal entries; `seed` makes the draw repeatable."""
    torch = _torch()
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    if torch is None:
        return np.random.default_rng(seed).standard_normal(shape).astype(_dtype(dtype))
    g = None if seed is None else torch.Generator().manual_seed(seed)
    draw = torch.randn(shape, generator=g, dtype=torch.float64)
    return draw.to(dtype=getattr(torch, _dtype(dtype)), device=_device(device) or "cpu")
