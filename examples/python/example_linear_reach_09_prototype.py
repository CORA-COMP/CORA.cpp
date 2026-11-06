"""example_linear_reach_09_prototype - a reachability algorithm prototyped in Python, type-checked

A forward-Euler step x + dt A x, applied to the zonotope of the initial set, is registered as the
algorithm "euler" of LinearSys.reach. It is no enclosure (it ignores the error of the step), so it
only shows how a prototype plugs in: sys.reach checks the problem, calls it, and returns its
ReachSet. The package ships type stubs, so `mypy --strict` checks this file against them.

Syntax:   PYTHONPATH=build python examples/python/example_linear_reach_09_prototype.py
Outputs:  the number of steps and the size of the last set of "euler" next to the built-in "standard"
See also: example_linear_reach_02_algorithms, CONTRIBUTING.md (Python stubs)

Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
Written:       04-November-2026
Last update:   ---
Reviewed:      ---
"""
import cora


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# Algorithm ---------------------------------------------------------------------------------

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

@cora.register_algorithm(cora.LinearSys, "euler")
def euler(sys: cora.LinearSys, problem: cora.ReachProblem, time_step: float = 0.1) -> cora.ReachSet:
    """Steps x + dt A x on the zonotope of the initial set (no error term)."""
    step = cora.eye(sys.A.shape[-1]) + time_step * sys.A
    X = problem.initial_set
    assert X is not None
    sets = [X]
    for _ in range(int(round(problem.t_final / time_step))):
        X = X.linear_map(step)
        sets.append(X)
    return cora.ReachSet(time_interval=sets[1:], time_point=sets, time_step=time_step)

# Parameters --------------------------------------------------------------------------------


# ============================================  MAIN  ============================================ #

t_final = 2.0
R0 = cora.Zonotope(cora.Tensor([1.0, 0.5]), cora.Tensor([[0.1, 0.0], [0.0, 0.1]]))

# System Dynamics ---------------------------------------------------------------------------

sys = cora.LinearSys(cora.Tensor([[-1.0, 0.0], [0.0, -2.0]]))

# Reachability Analysis ---------------------------------------------------------------------

Rprototype = sys.reach(R0, t_final=t_final, alg="euler", time_step=0.1)
Rstandard = sys.reach(R0, t_final=t_final, alg="standard", time_step=0.1)

# Evaluation --------------------------------------------------------------------------------

for name, R in (("euler", Rprototype), ("standard", Rstandard)):
    last = R.time_point[-1]
    print(f"{name:>8}: {len(R.time_point)} time points, last center {last.center}")

# ----------------------------------------  END OF CODE  ----------------------------------------- #
