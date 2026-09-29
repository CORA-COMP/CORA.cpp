"""example_zonotope_01 - a zonotope that is rotated and shrunk again and again

Operators as in CORA: A * Z maps the zonotope, and print(Z) shows it. A plain plot(Z) takes the
next color of CORA's order, so every copy has its own (blue, red, yellow, ...).

Syntax:   PYTHONPATH=build python examples/python/example_zonotope_01.py [--save FILE]
Outputs:  the matrix and the zonotope as text, and the figure
See also: example_zonotope_01 of the C++ examples, example_linear_reach_01_5dim
"""
import argparse

import matplotlib.pyplot as plt

from cora import Tensor, Zonotope, plot, setBackend

parser = argparse.ArgumentParser()
parser.add_argument("--save", metavar="FILE", help="write the figure instead of showing it")
args = parser.parse_args()
if args.save:
    import matplotlib

    matplotlib.use("Agg")

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

setBackend("torch")

# Init ---------------------------------------------------------------------------------------

# init zonotope
Z = Zonotope(Tensor([-1, 0]), 0.1 * Tensor([[1, 0], [0, 1]]))

# init rotation matrix
phi = 0.2  # radians
A = Tensor([[Tensor.cos(phi), Tensor.sin(phi)], [-Tensor.sin(phi), Tensor.cos(phi)]]) * 0.975

# diplay
print(A)
print(Z)

# Visualization -----------------------------------------------------------------------------

plt.figure()
for i in range(100):
    plot(Z)
    Z = A * Z

if args.save:
    plt.savefig(args.save, dpi=130, bbox_inches="tight")
else:
    plt.show()

# example completed

# ----------------------------------------  END OF CODE  ----------------------------------------- #
