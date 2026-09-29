"""The same reachable set with numpy arrays: they run on Eigen, torch tensors on libtorch.

    PYTHONPATH=build python examples/python/example_linearSys_reach_numpy.py
"""
import numpy as np

import coracpp

A = np.array([[-0.1, 1.0], [-1.0, -0.1]])
c = np.array([1.0, 0.0])
G = 0.1 * np.eye(2)

R = coracpp.reach(A, c, G, timeStep=0.1, tFinal=1.0, taylorTerms=8)

print("type of the result:", type(R.timeInt_c).__name__)  # ndarray: this ran on Eigen
print("center of the last enclosure:", R.timeInt_c[-1])
