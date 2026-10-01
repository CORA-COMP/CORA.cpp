"""benchmark_linear_verifyFast_ARCH23_heat3D - heat3D benchmark of ARCH 2023

The instances of MATLAB CORA's benchmark_linear_verifyFast_ARCH23_heat3D_<instance>.m, read from
benchmarks/data and verified with LinearSys.verify.

Syntax:   python benchmarks/python/benchmark_linear_verifyFast_ARCH23_heat3D.py [instance ...]
Outputs:  one line 'benchmark,instance,result,time' per instance, as the MATLAB scripts
See also: the C++ program of the same name (benchmarks/cpp)
"""
import benchmarkData

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# Parameters --------------------------------------------------------------------------------

family = ["HEAT01", "HEAT02"]

# Verification ------------------------------------------------------------------------------

# system, R0, U, specifications and verifyAlg come from the data of each instance
benchmarkData.main(family)

# ----------------------------------------  END OF CODE  ----------------------------------------- #
