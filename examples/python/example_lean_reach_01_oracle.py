"""example_lean_reach_01_oracle - reachable sets with CORALean as the oracle, in floating point

CORALean computes zonotopes whose float operations are sound: a nominal zonotope plus an error
box that encloses every rounding error. A damped oscillator is propagated in binary64 and in
binary32, the nominal part and the error box are taken apart, and both are compared with CORA.cpp's
own reachable set. A value crosses back into CORA.cpp only when it is exact. Without an oracle
the example prints a note and exits with code 0.

Syntax:   CORACPP_ORACLE="cd CORALean && lake exe oracle" PYTHONPATH=build python examples/python/example_lean_reach_01_oracle.py
Outputs:  per dtype the final interval of the nominal part, the error box and whether lean encloses
See also: example_linear_reach_01_5dim, example_lean_reach_01_oracle of the C++ examples

Authors:       Claude Sonnet 5.5
Written:       30-September-2026
Last update:   ---
Reviewed:      ---
"""
import os
import sys

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #


# ============================================  MAIN  ============================================ #

if not os.environ.get("CORACPP_ORACLE"):
    print("set CORACPP_ORACLE to the command that starts the CORALean oracle")
    sys.exit(0)


# Parameters --------------------------------------------------------------------------------


# every value is a dyadic number, so it is exact in binary64 and in binary32
A = np.array([[0.0, 1.0], [-1.0, -0.25]])
c, G = np.array([1.0, 0.0]), 0.25 * np.eye(2)
time_step, t_final = 0.125, 2.0  # a power of two: exact in every dtype
taylor_terms, zonotope_order = 8, 10

# CORA.cpp's own result, in double without a rounding-error account
ref = cora.LinearSys(cora.Tensor(A)).reach(cora.Zonotope(cora.Tensor(c), cora.Tensor(G)), time_step=time_step, t_final=t_final, taylor_terms=taylor_terms)
last = ref.time_point[-1].interval()
print(f"CORA.cpp:  x1 in [{float(last.inf[0]):.6f}, {float(last.sup[0]):.6f}]")

# The Oracle --------------------------------------------------------------------------------

for dtype in ("binary64", "ieee:binary32"):
    with cora.lean.dtype(dtype):  # the lean objects in the block are created in this dtype
        R = cora.lean.LinearSys(A).reach(cora.lean.Zonotope(c, G), t_final, time_step=time_step, taylor_terms=taylor_terms, zonotope_order=zonotope_order)

    # the nominal zonotope and the error box that the sound float operations added
    nominal, error = R.time_point[-1].gather()
    hull = nominal.interval()
    print(f"{dtype}: x1 in [{float(hull.inf[0]):.6f}, {float(hull.sup[0]):.6f}], "
          f"error box radius {float(error.sup[0]):.3g}")

    # the enclosure (nominal plus error) must contain CORA.cpp's interval at every time
    encloses = True
    for lean_set, z in zip(R.time_point, ref.time_point):
        box, I = lean_set.interval().gather(), z.interval()
        encloses &= bool(np.all(np.asarray(box.inf) <= np.asarray(I.inf) + 1e-9)
                         and np.all(np.asarray(box.sup) >= np.asarray(I.sup) - 1e-9))
    print(f"  encloses the CORA.cpp sets at all {len(R.time_point)} time points: "
          f"{'yes' if encloses else 'NO'}")

# Rounding ----------------------------------------------------------------------------------

# 0.1 is not a binary32 number: round_to returns the rounded value and an enclosure of the error
value, error = cora.lean.Tensor.from_array(np.array([[0.1]])).round_to("ieee:binary32")
inf_, sup_ = error.gather().inf, error.gather().sup
print(f"0.1 in binary32 is {float(value.gather()[0, 0])}, "
      f"the error lies in [{float(inf_[0]):.3g}, {float(sup_[0]):.3g}]")

# example completed

# ----------------------------------------  END OF CODE  ----------------------------------------- #
