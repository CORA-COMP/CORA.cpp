"""extend - algorithms written in Python, run by the systems like the built-in ones

A prototype of a reachability algorithm registers itself for a system and an operation; the
system then runs it for its name, after the problem checks of the built-in algorithms, and the
result is a cora.ReachSet like theirs (so cora.plot draws it). It may use every public building
block (sys.correction_matrix_state, sys.expm_remainder, Zonotope.linear_map, ...).

    @cora.register_algorithm(cora.LinearSys, "my_euler")
    def my_euler(sys, problem, time_step=0.1):
        ...                          # problem.initial_set, problem.input_set, problem.t_final
        return cora.ReachSet(time_interval=[...], time_point=[...])

    R = sys.reach(initial_set=X0, t_final=1.0, alg="my_euler", time_step=0.05)

The keywords that are not fields of the problem go to the function as its options.
"""

# {(system class name, operation, algorithm name): function(system, problem, **options)}
ALGORITHMS = {}


def register_algorithm(system, name, operation="reach"):
    """A decorator: registers the function as the algorithm `name` of system.operation."""
    def register(function):
        ALGORITHMS[(system.__name__, operation, name)] = function
        return function
    return register
