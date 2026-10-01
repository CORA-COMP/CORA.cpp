"""benchmark_linear_verifyFast_ARCH23_beam - beam benchmark of ARCH 2023

The instances of MATLAB CORA's benchmark_linear_verifyFast_ARCH23_beam_<instance>.m, read from
benchmarks/data and verified with LinearSys.verify.

Syntax:   python benchmarks/python/benchmark_linear_verifyFast_ARCH23_beam.py [instance ...]
Outputs:  one line 'benchmark,instance,result,time' per instance, as the MATLAB scripts
See also: the C++ program of the same name (benchmarks/cpp)
"""
import benchmarkData

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# Parameters --------------------------------------------------------------------------------

family = ["CBC01", "CBC02", "CBC03", "CBF01", "CBF02", "CBF03"]

# Verification ------------------------------------------------------------------------------

# system, R0, U, specifications and verifyAlg come from the data of each instance
benchmarkData.main(family)

# ----------------------------------------  END OF CODE  ----------------------------------------- #
