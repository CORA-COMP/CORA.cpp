"""nn_verify - verify a VNN-COMP instance: falsify, bound, split, with exchangeable strategies

An instance is an ONNX network and a VNN-LIB property, loaded once into an NnProblem (the network
on torch in the search dtype, the boxes and polytopes as tensors, the onnxruntime check of
counterexamples). verify runs the falsification method and the branch and bound in the order of
options.falsification_order; every strategy is chosen by name, as CORA's options.nn
(validateNNoptions.m): falsification_method (cora.nn_attack), bounding_method ("zonotope": the
cora.NeuralNetwork enclosure, DeepZ; "conzonotope" a Phase 1 stub), input_split_heuristic and
neuron_split_heuristic (a Phase 1 stub). A box is verified when, for every polytope of its case,
some row has a lower bound above b + verified_margin. Batches follow memory_budget (bytes).

Syntax:   problem = NnProblem.load("net.onnx", "prop.vnnlib", VerifyOptions(device="cpu"))
          result = verify(problem, timeout=60)     result.status: "sat", "unsat" or "unknown"
See also: cora.onnx_network, cora.vnnlib, cora.nn_attack, cora.counterexample,
          docs/nn_verification.md

Authors:       Claude Opus 5.5
Written:       21-November-2026
Last update:   ---
Reviewed:      ---
"""

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from . import counterexample, nn_attack
from .onnx_network import OnnxNetwork
from .vnnlib import parse_vnnlib, read_text


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

BOUNDING_METHODS = {}  # name -> Bounding subclass
INPUT_SPLIT_HEURISTICS = {}  # name -> function(problem, lo, hi, case) -> dimension per box
NEURON_SPLIT_HEURISTICS = {}  # name -> function; Phase 1


def _float32_box(lo, hi):
    """The float32 box inside [lo, hi] (the smallest float32 >= lo, the largest <= hi), so that a
    point of it passes the input check in float32 and in float64; nearest where it is empty."""
    lo32, hi32 = lo.astype(np.float32), hi.astype(np.float32)
    lo32 = np.where(lo32 < lo, np.nextafter(lo32, np.float32(np.inf)), lo32)
    hi32 = np.where(hi32 > hi, np.nextafter(hi32, np.float32(-np.inf)), hi32)
    empty = lo32 > hi32
    return np.where(empty, lo.astype(np.float32), lo32), np.where(
        empty, hi.astype(np.float32), hi32
    )


def _outer_box(lo, hi):
    """The float64 box that also holds the nearest float32 of each bound: the scoring compares a
    float32 input with the bound rounded to float32, so the bounds must cover that much."""
    near_lo, near_hi = (
        lo.astype(np.float32).astype(np.float64),
        hi.astype(np.float32).astype(np.float64),
    )
    return np.minimum(lo, near_lo), np.maximum(hi, near_hi)


def register_bounding(name):
    """A decorator: registers a Bounding subclass under the option name."""

    def register(cls):
        BOUNDING_METHODS[name] = cls
        return cls

    return register


def register_input_split_heuristic(name):
    """A decorator: registers a function (problem, lo, hi, case) -> dimension to split per box."""

    def register(function):
        INPUT_SPLIT_HEURISTICS[name] = function
        return function

    return register


# ============================================  MAIN  ============================================ #


@dataclass
class VerifyOptions:
    """The options of verify, named after CORA's options.nn where one exists."""

    falsification_method: str = "pgd"
    falsification_order: str = "attack-first"  # "attack-first" or "bound-first"
    attack_options: dict = field(default_factory=dict)
    attack_time_share: float = 0.3  # of the timeout at most, for attack-first
    attack_first_rounds: int = 3  # rounds of restarts at most, for attack-first
    bounding_method: str = "zonotope"
    input_split_heuristic: str = "most-sensitive-input-radius"
    neuron_split_heuristic: str = "none"  # Phase 1
    num_splits: int = 2  # pieces per split dimension
    num_dimensions: int = 1  # dimensions split per box
    verify_dequeue_type: str = "front"  # "front" (breadth first) or "back" (depth first)
    max_verif_iter: float = math.inf
    verified_margin: float = 1e-4  # float32 network vs the float64 enclosure
    memory_budget: int = 0  # bytes per batch; 0: 70% of a GPU's memory, 1 GiB on the CPU
    device: str = "cpu"
    dtype: torch.dtype = torch.float32  # the search; bounds are float64
    max_candidates: int = 8  # onnxruntime checks per attack step
    seed: int = 0

    def validate(self):
        """Raises for an unknown strategy name."""
        for value, known, name in (
            (self.falsification_method, nn_attack.ATTACKS, "falsification_method"),
            (self.bounding_method, BOUNDING_METHODS, "bounding_method"),
            (self.input_split_heuristic, INPUT_SPLIT_HEURISTICS, "input_split_heuristic"),
        ):
            if value not in known:
                raise ValueError(f"VerifyOptions: unknown {name} {value!r}; known: {sorted(known)}")
        if self.neuron_split_heuristic != "none":
            raise NotImplementedError("VerifyOptions: neuron splitting comes with Phase 1")
        if self.falsification_order not in ("attack-first", "bound-first"):
            raise ValueError(
                f"VerifyOptions: unknown falsification_order {self.falsification_order!r}"
            )
        return self


@dataclass
class VerifyResult:
    """The verdict and, for sat, the counterexample (rows x, y) with its text; stats for the CSV."""

    status: str
    x: object = None
    y: object = None
    counterexample: str = ""
    stats: dict = field(default_factory=dict)


class NnProblem:
    """An instance loaded once: the network on torch, the property's boxes and polytopes as
    tensors, and the check of counterexamples, shared by attacks and bounds."""

    def __init__(self, onnx_path, net, prop, vnnlib_text, options):
        self.onnx_path, self.net, self.prop, self.vnnlib_text = onnx_path, net, prop, vnnlib_text
        self.options, self.device = options, torch.device(options.device)
        self.max_candidates = options.max_candidates
        self.memory_budget = options.memory_budget or (
            int(0.7 * torch.cuda.get_device_properties(self.device).total_memory)
            if self.device.type == "cuda" else 2**30)
        # the boxes of the search in float32 (inside the checked box), the outer float64 boxes of
        # the bounds, the polytopes padded to one row count
        boxes = [_float32_box(c.lo, c.hi) for c in prop.cases]
        as_tensor = lambda a: torch.as_tensor(np.asarray(a), device=self.device)  # noqa: E731
        self.lo = as_tensor(np.stack([b[0] for b in boxes])).to(options.dtype)
        self.hi = as_tensor(np.stack([b[1] for b in boxes])).to(options.dtype)
        self.num_boxes = len(boxes)
        outer = [_outer_box(c.lo, c.hi) for c in prop.cases]
        self.outer_lo = torch.as_tensor(np.stack([b[0] for b in outer]), dtype=torch.float64)
        self.outer_hi = torch.as_tensor(np.stack([b[1] for b in outer]), dtype=torch.float64)
        self.case_of_box = torch.arange(self.num_boxes, device=self.device)
        case, A, b = prop.stacked()
        self.poly_case, self.A, self.b = as_tensor(case), as_tensor(A), as_tensor(b)
        self._A32, self._b32 = self.A.to(options.dtype), self.b.to(options.dtype)
        self.sample_bytes = 4 * max(1, net.activation_elements()) * (options.dtype.itemsize)
        self.checks = 0

    @classmethod
    def load(cls, onnx_path, vnnlib_path, options=None):
        """The problem of the files; the network and the property are read once."""
        options = (options or VerifyOptions()).validate()
        net = OnnxNetwork.load(onnx_path, dtype=options.dtype, device=options.device)
        text = read_text(vnnlib_path)
        prop = parse_vnnlib(text, net.num_inputs, net.num_outputs)
        return cls(onnx_path, net, prop, text, options)

    def batch_size(self, kind):
        """How many samples ("attack") fit into the memory budget."""
        return max(1, int(self.memory_budget // self.sample_bytes))

    def loss(self, y, case):
        """Per row: min over the polytopes of its case of max_k (A_k y - b_k); negative inside the
        unsafe set. y (S, m), case (S,)."""
        margin = torch.einsum("pkm,sm->spk", self._A32, y) - self._b32
        worst = margin.amax(dim=2)
        worst = torch.where(self.poly_case[None, :] == case[:, None], worst, torch.inf)
        return worst.amin(dim=1)

    def objective(self, y, case):
        """The loss that attacks descend: per polytope the sum of the violated rows' excess plus a
        tenth of the worst row (which keeps pushing inside), the best polytope of the case."""
        margin = torch.einsum("pkm,sm->spk", self._A32, y) - self._b32
        value = torch.relu(margin).sum(dim=2) + 0.1 * margin.amax(dim=2)
        value = torch.where(self.poly_case[None, :] == case[:, None], value, torch.inf)
        return value.amin(dim=1)

    def confirm(self, x):
        """(x, y) as float32 numpy rows if the counterexample x passes the scoring's check
        (CORRECT), else None; y is onnxruntime's output."""
        self.checks += 1
        x = x.detach().to("cpu", torch.float32).numpy().astype(np.float32)
        sess = counterexample._session_v1(self.onnx_path)
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in sess.get_inputs()[0].shape]
        y = sess.run(None, {sess.get_inputs()[0].name: x.reshape(shape)})[0].reshape(-1)
        text = counterexample.write_counterexample(x, y, self.prop)
        result, _ = counterexample.check_counterexample(
            self.onnx_path, None, text, self.vnnlib_text
        )
        return (x, y, text) if result == counterexample.CORRECT else None


class Bounding:
    """The interface of a bounding method: prepare(problem) once, then lower(lo, hi, case) the
    lower bounds (B, P, K) of A y - b over each box (float64); None if it cannot bound the net."""

    def prepare(self, problem):
        raise NotImplementedError

    def lower(self, lo, hi, case):
        raise NotImplementedError

    def box_bytes(self):
        """The memory of one box in a batch."""
        raise NotImplementedError


@register_bounding("zonotope")
class ZonotopeBounding(Bounding):
    """The output zonotope of cora.NeuralNetwork (a generator per input dimension of positive
    radius, the ReLU, sigmoid and tanh by their parallelograms): A c - sum |A G| per row."""

    def prepare(self, problem):
        self.problem = problem
        self.nn, self.why = problem.net.to_neural_network()
        if self.nn is None:
            return False
        layers = problem.net.dense_layers()[0][0]
        self.widths = [W.shape[0] for W, _ in layers]
        self.A64, self.b64 = problem.A.to("cpu", torch.float64), problem.b.to("cpu", torch.float64)
        return True

    def box_bytes(self):
        n = self.problem.net.num_inputs
        q = n + sum(self.widths[:-1])  # the input generators and one per neuron
        return 8 * 3 * q * max(self.widths + [n])

    def lower(self, lo, hi, case):
        import cora

        lo, hi = lo.to("cpu", torch.float64), hi.to("cpu", torch.float64)
        c, r = 0.5 * (lo + hi), 0.5 * (hi - lo)
        # generators only for the dimensions that some box of the batch perturbs
        active = torch.nonzero((r > 0).any(dim=0)).flatten()
        G = torch.zeros(lo.shape[0], len(active), lo.shape[1], dtype=torch.float64)
        G[:, torch.arange(len(active)), active] = r[:, active]
        Y = self.nn.evaluate(cora.Zonotope(c, G))
        yc = torch.as_tensor(np.asarray(Y.center), dtype=torch.float64).reshape(lo.shape[0], -1)
        yG = torch.as_tensor(np.asarray(Y.generators), dtype=torch.float64).reshape(
            lo.shape[0], -1, yc.shape[1]
        )
        Ac = torch.einsum("pkm,bm->bpk", self.A64, yc)
        AG = torch.einsum("pkm,bgm->bpkg", self.A64, yG).abs().sum(dim=3)
        return Ac - AG - self.b64[None]


@register_bounding("conzonotope")
class ConZonotopeBounding(Bounding):
    """Stub for Phase 1: the latent-space zonotope with the constraints of the splits."""

    def prepare(self, problem):
        raise NotImplementedError("nn_verify: the constrained zonotope comes with Phase 1")


@register_input_split_heuristic("largest-radius")
def split_largest_radius(problem, lo, hi, case):
    """The dimension of the largest radius."""
    return (hi - lo).argmax(dim=1)


@register_input_split_heuristic("most-sensitive-input-radius")
def split_most_sensitive(problem, lo, hi, case):
    """The dimension of the largest radius times |d loss / d x| at the center (CORA's
    most-sensitive-input-radius, with the gradient of the property's loss as the sensitivity)."""
    x = (0.5 * (lo + hi)).to(problem.device, problem.options.dtype).requires_grad_(True)
    with torch.enable_grad():
        loss = problem.loss(problem.net(x), case.to(problem.device))
        (grad,) = torch.autograd.grad(loss.sum(), x)
    score = (hi - lo).to(grad.device, grad.dtype) * grad.abs()
    return score.argmax(dim=1).to(lo.device)


def _split(lo, hi, case, dims, pieces):
    """Every box cut into `pieces` equal parts along its dimension."""
    rows = torch.arange(lo.shape[0])
    width = (hi[rows, dims] - lo[rows, dims]) / pieces
    out_lo, out_hi = [], []
    for k in range(pieces):
        a, b = lo.clone(), hi.clone()
        a[rows, dims] = lo[rows, dims] + k * width
        b[rows, dims] = hi[rows, dims] if k == pieces - 1 else lo[rows, dims] + (k + 1) * width
        out_lo.append(a)
        out_hi.append(b)
    return torch.cat(out_lo), torch.cat(out_hi), case.repeat(pieces)


def branch_and_bound(problem, bounding, deadline, stats):
    """'unsat' if every box is verified, 'sat' (with the counterexample in stats) if a leaf's
    center falsifies, else 'unknown' at the deadline or max_verif_iter."""
    o = problem.options
    lo, hi, case = problem.outer_lo, problem.outer_hi, problem.case_of_box.to("cpu")
    batch = max(1, int(problem.memory_budget // bounding.box_bytes()))
    center = nn_attack.make_attack("center")
    iteration, last = 0, 0.0
    # a batch starts only if the last one's time still fits before the deadline
    while lo.shape[0] and time.monotonic() + last < deadline and iteration < o.max_verif_iter:
        started = time.monotonic()
        # the next batch, from the front (breadth first) or the back (depth first)
        take = (
            slice(0, batch)
            if o.verify_dequeue_type == "front"
            else slice(max(0, lo.shape[0] - batch), None)
        )
        keep = (
            slice(batch, None)
            if o.verify_dequeue_type == "front"
            else slice(0, max(0, lo.shape[0] - batch))
        )
        blo, bhi, bcase = lo[take], hi[take], case[take]
        lo, hi, case = lo[keep], hi[keep], case[keep]
        bounds = bounding.lower(blo, bhi, bcase)
        decided = bounds > o.verified_margin  # (B, P, K): the row excludes the polytope
        excluded = decided.any(dim=2) | (problem.poly_case.cpu()[None, :] != bcase[:, None])
        open_ = ~excluded.all(dim=1)
        stats["boxes"] = stats.get("boxes", 0) + blo.shape[0]
        iteration, last = iteration + 1, time.monotonic() - started
        if not bool(open_.any()):
            continue
        blo, bhi, bcase = blo[open_], bhi[open_], bcase[open_]
        # a cheap falsification of the open boxes: their centers
        sub = _SubProblem(problem, blo, bhi, bcase)
        found = center.run(sub, deadline)
        if found.x is not None:
            stats["found"] = found
            return "sat"
        dims = INPUT_SPLIT_HEURISTICS[o.input_split_heuristic](problem, blo, bhi, bcase)
        slo, shi, scase = _split(blo, bhi, bcase, dims, o.num_splits)
        lo, hi, case = torch.cat([lo, slo]), torch.cat([hi, shi]), torch.cat([case, scase])
    stats["iterations"] = iteration
    return "unsat" if lo.shape[0] == 0 else "unknown"


class _SubProblem:
    """The problem restricted to some boxes (for an attack on the open leaves)."""

    def __init__(self, problem, lo, hi, case):
        self.__dict__.update(problem.__dict__)
        self.parent = problem
        self.lo, self.hi = (
            lo.to(problem.device, problem.options.dtype),
            hi.to(problem.device, problem.options.dtype),
        )
        self.case_of_box, self.num_boxes = case.to(problem.device), lo.shape[0]

    def __getattr__(self, name):
        return getattr(self.parent, name)


def verify(problem, timeout):
    """The verdict on the problem within timeout seconds (wall time from the call)."""
    start = time.monotonic()
    deadline = start + timeout
    o, stats = problem.options, {}
    attack = nn_attack.make_attack(o.falsification_method, **o.attack_options)
    bounding = BOUNDING_METHODS[o.bounding_method]()
    can_bound = bounding.prepare(problem)
    stats["bounding"] = o.bounding_method if can_bound else f"none ({bounding.why})"

    def run_attack(until, rounds=None):
        t = time.monotonic()
        found = attack.run(problem, until, rounds)
        stats["attack_seconds"] = stats.get("attack_seconds", 0.0) + time.monotonic() - t
        stats["attack_iterations"] = stats.get("attack_iterations", 0) + found.iterations
        return found

    def run_bab():
        t = time.monotonic()
        status = branch_and_bound(problem, bounding, deadline, stats)
        stats["bab_seconds"] = time.monotonic() - t
        return status

    # the order of falsification and bounding
    if o.falsification_order == "attack-first":
        found = run_attack(start + o.attack_time_share * timeout, o.attack_first_rounds) \
            if can_bound else run_attack(deadline)
        if found.x is not None:
            return _sat(problem, found, stats)
    status = run_bab() if can_bound else "unknown"
    if status == "sat":
        return _sat(problem, stats.pop("found"), stats)
    if status == "unsat":
        return VerifyResult("unsat", stats=stats)
    found = run_attack(deadline)
    if found.x is not None:
        return _sat(problem, found, stats)
    return VerifyResult("unknown", stats=stats)


def _sat(problem, found, stats):
    """The sat result of a confirmed counterexample."""
    stats["checks"] = problem.checks
    text = counterexample.write_counterexample(found.x, found.y, problem.prop)
    return VerifyResult("sat", found.x, found.y, text, stats)


# ----------------------------------------  END OF CODE  ----------------------------------------- #
