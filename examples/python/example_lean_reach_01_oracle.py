"""example_lean_reach_01_oracle - reachable sets with CORALean as the oracle, in floating point

CORALean computes zonotopes whose float operations are sound: a nominal zonotope plus an error
box that encloses every rounding error. A damped oscillator is propagated in binary64 and in
binary32, the nominal part and the error box are taken apart, and both are compared with CORA.cpp's
own reachable set. A value crosses back into CORA.cpp only when it is exact.

Syntax:   CORACPP_ORACLE="cd CORALean && lake exe oracle" PYTHONPATH=build python examples/python/example_lean_reach_01_oracle.py
Outputs:  per dtype the final interval of the nominal part, the radius of the error box and whether
          the lean sets enclose CORA.cpp's; without an oracle a note and exit code 0
See also: example_linear_reach_01_5dim, example_lean_reach_01_oracle of the C++ examples
"""
import os
import sys

import numpy as np

import cora

if not os.environ.get("CORACPP_ORACLE"):
    print("set CORACPP_ORACLE to the command that starts the CORALean oracle")
    sys.exit(0)

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# Parameters --------------------------------------------------------------------------------

# every value is a dyadic number, so it is exact in binary64 and in binary32
A = np.array([[0.0, 1.0], [-1.0, -0.25]])
c, G = np.array([1.0, 0.0]), 0.25 * np.eye(2)
timeStep, tFinal = 0.125, 2.0  # a power of two: exact in every dtype
taylorTerms, zonotopeOrder = 8, 10

# CORA.cpp's own result, in double without a rounding-error account
ref = cora.LinearSys(cora.Tensor(A)).reach(cora.Zonotope(cora.Tensor(c), cora.Tensor(G)),
                                           timeStep, tFinal, taylorTerms)
last = ref.timePoint[-1].interval()
print(f"CORA.cpp:  x1 in [{float(last.inf[0]):.6f}, {float(last.sup[0]):.6f}]")

# The Oracle --------------------------------------------------------------------------------

for dtype in ("binary64", "ieee:binary32"):
    cora.lean.setDType(dtype)  # the lean objects below are created in this dtype
    R = cora.lean.LinearSys(A).reach(cora.lean.Zonotope(c, G), timeStep, tFinal, taylorTerms,
                                     zonotopeOrder)

    # the nominal zonotope and the error box that the sound float operations added
    nominal, error = R.timePoint[-1].gather()
    hull = nominal.interval()
    print(f"{dtype}: x1 in [{float(hull.inf[0]):.6f}, {float(hull.sup[0]):.6f}], "
          f"error box radius {float(error.sup[0]):.3g}")

    # the enclosure (nominal plus error) must contain CORA.cpp's interval at every time
    encloses = True
    for lean_set, z in zip(R.timePoint, ref.timePoint):
        box, I = lean_set.interval().gather(), z.interval()
        encloses &= bool(np.all(np.asarray(box.inf) <= np.asarray(I.inf) + 1e-9)
                         and np.all(np.asarray(box.sup) >= np.asarray(I.sup) - 1e-9))
    print(f"  encloses the CORA.cpp sets at all {len(R.timePoint)} time points: "
          f"{'yes' if encloses else 'NO'}")

# Rounding ----------------------------------------------------------------------------------

# 0.1 is not a binary32 number: roundTo returns the rounded value and an enclosure of the error
cora.lean.setDType("binary64")
value, error = cora.lean.Tensor.fromArray(np.array([[0.1]])).roundTo("ieee:binary32")
inf_, sup_ = error.gather().inf, error.gather().sup
print(f"0.1 in binary32 is {float(value.gather()[0, 0])}, "
      f"the error lies in [{float(inf_[0]):.3g}, {float(sup_[0]):.3g}]")

# example completed

# ----------------------------------------  END OF CODE  ----------------------------------------- #
