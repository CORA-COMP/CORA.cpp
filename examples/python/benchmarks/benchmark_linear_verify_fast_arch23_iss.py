"""benchmark_linear_verify_fast_arch23_iss - Spacestation benchmark of ARCH 2023

The instances of MATLAB CORA's benchmark_linear_verifyFast_ARCH23_iss_<instance>.m, read from
benchmarks/data and verified with LinearSys.verify.

Syntax:   python benchmarks/python/benchmark_linear_verifyFast_ARCH23_iss.py [instance ...]
Outputs:  one line 'benchmark,instance,result,time' per instance, as the MATLAB scripts
See also: the C++ program of the same name (benchmarks/cpp)

Authors:       Tobias Ladner
Written:       02-November-2026
Last update:   ---
Reviewed:      ---
"""
import benchmark_data

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# Parameters --------------------------------------------------------------------------------


# ============================================  MAIN  ============================================ #

family = ["ISSC01_ISS02", "ISSC01_ISU02", "ISSF01_ISS01", "ISSF01_ISU01"]

# Verification ------------------------------------------------------------------------------

# system, R0, U, specifications and verifyAlg come from the data of each instance
benchmark_data.main(family)

# ----------------------------------------  END OF CODE  ----------------------------------------- #
