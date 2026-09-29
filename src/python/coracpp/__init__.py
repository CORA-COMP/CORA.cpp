"""CORA.cpp: reachability of linear systems on Eigen (numpy) or libtorch (torch), with CORA's
plotting.

    import coracpp
    R = coracpp.reach(A, c, G, timeStep=0.1, tFinal=2.0)
    coracpp.plot(R)
"""
from ._coracpp import Reach, Specification, randPoint, reach, simulate, simulateRandom
from .colors import CORAcolor
from .plot import (plot, plot_initial_set, plot_interval, plot_points, plot_reach,
                   plot_simulation, plot_specification, plot_zonotope, zonotope_vertices)

__all__ = [
    "CORAcolor", "Reach", "Specification", "plot", "plot_initial_set", "plot_interval",
    "plot_points", "plot_reach", "plot_simulation", "plot_specification", "plot_zonotope",
    "randPoint", "reach", "simulate", "simulateRandom", "zonotope_vertices",
]
