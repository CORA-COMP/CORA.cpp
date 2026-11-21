"""nn_attack - falsification of a neural-network property: the Attack interface and its methods

An attack searches the input boxes of a property for a point whose output lies in the unsafe set,
batched on torch (any device). Every candidate is confirmed by the problem's check (onnxruntime,
as the VNN-COMP scoring replays it) before it counts. Attacks are chosen by name, as CORA's
options.nn.falsification_method: "pgd" (projected sign-gradient descent with random restarts and
step decay, CORA's computePGDAttack batched), "fgsm" (one step), "center" (the box centers);
"zonotack" (the zonotope's vertex, ICLR 2026) is a stub that Phase 1 fills in. A new method
subclasses Attack and registers itself with @register_attack("name").

Syntax:   attack = make_attack("pgd", steps=100, restarts=64)
          result = attack.run(problem, deadline)   problem: cora.nn_verify.NnProblem
Outputs:  result - AttackResult: x and y (None if nothing was found), the counts and the time
See also: cora.nn_verify, docs/nn_verification.md

Authors:       Claude Opus 5.5
Written:       21-November-2026
Last update:   ---
Reviewed:      ---
"""

import time
from dataclasses import dataclass

import torch


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

ATTACKS = {}  # name -> Attack subclass


def _uniform(lo, hi, generator):
    """Points drawn uniformly from the boxes [lo, hi] (rows)."""
    u = torch.rand(lo.shape, generator=generator, device=lo.device, dtype=lo.dtype)
    return lo + u * (hi - lo)


# ============================================  MAIN  ============================================ #


def register_attack(name):
    """A decorator: registers an Attack subclass under the option name."""

    def register(cls):
        ATTACKS[name] = cls
        cls.name = name
        return cls

    return register


def make_attack(name, **options):
    """The attack of the option name with its options."""
    if name not in ATTACKS:
        raise ValueError(
            f"nn_attack: unknown falsification_method {name!r}; known: {sorted(ATTACKS)}"
        )
    return ATTACKS[name](**options)


@dataclass
class AttackResult:
    """What an attack found: x and y (float32 rows of the network's input and output, None if
    nothing), the case of the box, and the effort."""

    x: object = None
    y: object = None
    case: int = -1
    iterations: int = 0
    evaluations: int = 0
    candidates: int = 0
    seconds: float = 0.0


class Attack:
    """The interface: run(problem, deadline, rounds) searches until time.monotonic() reaches
    deadline (or after `rounds` rounds of restarts, where the method has rounds)."""

    name = ""

    def __init__(self, **options):
        self.options = options

    def run(self, problem, deadline, rounds=None):
        raise NotImplementedError

    def _confirm(self, problem, x, loss, case, result):
        """Whether a row of x with loss <= 0 (in the closed unsafe set) passes the problem's check,
        the lowest first, at most problem.max_candidates; the first that does goes into result."""
        rows = torch.nonzero(loss <= 0).flatten()
        rows = rows[torch.argsort(loss[rows])][: problem.max_candidates]
        for i in rows.tolist():
            result.candidates += 1
            found = problem.confirm(x[i])
            if found is not None:
                result.x, result.y, result.case = found[0], found[1], int(case[i])
                return True
        return False


@register_attack("center")
class CenterAttack(Attack):
    """The center of every box (CORA's falsification_method 'center')."""

    def run(self, problem, deadline, rounds=None):
        start, result = time.monotonic(), AttackResult()
        with torch.no_grad():
            x = 0.5 * (problem.lo + problem.hi)
            loss = problem.loss(problem.net(x), problem.case_of_box)
        result.evaluations, result.iterations = x.shape[0], 1
        self._confirm(problem, x, loss, problem.case_of_box, result)
        result.seconds = time.monotonic() - start
        return result


@register_attack("pgd")
class PgdAttack(Attack):
    """Projected gradient descent on the loss of the property (negative inside the unsafe set):
    sign steps of step_size times the box radius, multiplied by decay at the decay_at fractions of
    the steps, from the center and from uniform restarts, in rounds until the deadline."""

    def __init__(
        self,
        steps=100,
        restarts=64,
        step_size=0.25,
        decay=0.5,
        decay_at=(0.5, 0.75),
        rounds=None,
        seed=0,
    ):
        super().__init__(
            steps=steps,
            restarts=restarts,
            step_size=step_size,
            decay=decay,
            decay_at=decay_at,
            rounds=rounds,
            seed=seed,
        )

    def run(self, problem, deadline, rounds=None):
        o, start, result = self.options, time.monotonic(), AttackResult()
        generator = torch.Generator(device=problem.device).manual_seed(o["seed"])
        # every box gets restarts samples per round, in batches the memory budget allows
        per_box = max(
            1, min(o["restarts"], problem.batch_size("attack") // max(1, problem.num_boxes))
        )
        case = problem.case_of_box.repeat_interleave(per_box)
        lo, hi = problem.lo.repeat_interleave(per_box, 0), problem.hi.repeat_interleave(per_box, 0)
        radius = 0.5 * (hi - lo)
        decay_steps = {int(f * o["steps"]) for f in o["decay_at"]}
        round_index = 0
        limit = rounds if rounds is not None else o["rounds"]
        while time.monotonic() < deadline and (limit is None or round_index < limit):
            x = _uniform(lo, hi, generator)
            if round_index == 0:
                x[::per_box] = 0.5 * (lo[::per_box] + hi[::per_box])  # the first one: the center
            alpha = o["step_size"] * radius
            for step in range(o["steps"]):
                if step in decay_steps:
                    alpha = alpha * o["decay"]
                # one sign step on the loss, projected back into the box
                x.requires_grad_(True)
                with torch.enable_grad():
                    y = problem.net(x)
                    (grad,) = torch.autograd.grad(problem.objective(y, case).sum(), x)
                    loss = problem.loss(y.detach(), case)
                with torch.no_grad():
                    x = torch.minimum(torch.maximum(x.detach() - alpha * grad.sign(), lo), hi)
                result.iterations += 1
                result.evaluations += x.shape[0]
                if bool((loss <= 0).any()) and self._confirm(
                    problem, x, loss.detach(), case, result
                ):
                    result.seconds = time.monotonic() - start
                    return result
                if time.monotonic() >= deadline:
                    break
            # the last iterate too
            with torch.no_grad():
                loss = problem.loss(problem.net(x), case)
            if bool((loss <= 0).any()) and self._confirm(problem, x, loss, case, result):
                break
            round_index += 1
        result.seconds = time.monotonic() - start
        return result


@register_attack("fgsm")
class FgsmAttack(PgdAttack):
    """One sign step of the full radius from the center (CORA's falsification_method 'fgsm')."""

    def __init__(self, seed=0):
        super().__init__(steps=1, restarts=1, step_size=1.0, decay_at=(), rounds=1, seed=seed)


@register_attack("zonotack")
class ZonotackAttack(Attack):
    """Stub for Phase 1: the vertex of the output zonotope nearest each unsafe halfspace, mapped
    back through the shared factors (CORA's falsification_method 'zonotack')."""

    def run(self, problem, deadline, rounds=None):
        raise NotImplementedError("nn_attack: zonotack comes with the zonotope verifier (Phase 1)")


# ----------------------------------------  END OF CODE  ----------------------------------------- #
