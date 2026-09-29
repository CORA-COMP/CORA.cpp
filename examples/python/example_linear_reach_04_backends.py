"""example_linear_reach_04_backends - the same code on Eigen and on libtorch

The backend is chosen once, with setBackend (or CORACPP_BACKEND from outside); cora.Tensor makes
numpy arrays on Eigen and torch tensors on libtorch, and everything built from them stays there.
Nothing else in the code changes, and the results come back as the array type of the backend.

Syntax:   PYTHONPATH=build python examples/python/example_linear_reach_04_backends.py
Outputs:  the last center on each backend, and how much the two differ
See also: example_linear_reach_06_gpu
"""
import cora

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

def reachOscillator():
    """The reachable set of a damped oscillator, on whichever backend is current."""
    # Parameters ----------------------------------------------------------------------------

    R0 = cora.Zonotope(cora.Tensor([1.0, 0.0]), 0.1 * cora.eye(2))

    # System Dynamics -----------------------------------------------------------------------

    sys = cora.LinearSys(cora.Tensor([[-0.1, 1.0], [-1.0, -0.1]]))

    # Reachability Analysis -----------------------------------------------------------------

    return sys.reach(R0, timeStep=0.1, tFinal=1.0, taylorTerms=8)

# Backends ----------------------------------------------------------------------------------

backends = ["eigen", "torch"]

# Reachability Analysis ---------------------------------------------------------------------

centers = {}
for spec in backends:
    cora.setBackend(spec)
    R = reachOscillator()
    centers[spec] = R.timeInt[-1].c
    print(f"{spec}: the last center is {centers[spec]} ({type(centers[spec]).__module__})")

# Evaluation --------------------------------------------------------------------------------

gap = abs(centers["eigen"] - centers["torch"].numpy()).max()
print("eigen and torch differ by", float(gap))

# example completed

# ----------------------------------------  END OF CODE  ----------------------------------------- #
