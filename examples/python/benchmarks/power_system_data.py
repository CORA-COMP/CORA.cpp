"""power_system_data - writes TSPS25 (the IEEE 14 bus power system) to examples/benchmarks/data

MATLAB CORA converts the power system cases of models/powerSystems/*.mat into differential-algebraic
equations with powerSystem2cora (symbolic toolbox) and writes them to models/powerSystemsConverted
as <case>_dyn.m and <case>_con.m. This script reads those files and the cases, and writes the
equations as text with the numbers of the cases: nothing is retyped. The equations stay in the
MATLAB syntax of the generated files (x(i), y(i), u(i), pi, sin, cos, +, -, *, /, ^); the
benchmark program parses them. The state, input and constraint counts are those of powerSystem2cora
(priv_symPowerVariables): 3 states per generator, generator + 2 input + total buses inputs, and
2 per bus constraints (one less with a slack bus). The folder of the cases defaults to
models/powerSystems next to the converted folder.

Syntax:   python examples/python/benchmarks/power_system_data.py <converted folder> [cases folder]
Inputs:   the folder of the <case>_dyn.m and <case>_con.m files, and that of the cases (optional)
Outputs:  examples/benchmarks/data/TSPS25.json
See also: benchmark_nonlinear_das_reach_arch23_powersystem.cpp, heat3d_data.py

Authors:       Claude Sonnet 5.5
Written:       11-November-2026
Last update:   ---
Reviewed:      ---
"""
import json
import os
import re
import sys

import scipy.io


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "benchmarks", "data")
CASES = ["IEEE14", "IEEE14_sub1", "IEEE14_sub2", "IEEE14_fault", "IEEE14_fault_sub1"]


def equations(path, name):
    """The right-hand sides of f(i,1)=...; (or g(i,1)=...;) of a generated file, in order."""
    rows = {}
    with open(path) as fh:
        for line in fh:
            m = re.match(r"\s*[fg]\((\d+),1\)\s*=\s*(.*);\s*$", line)
            if m:
                rows[int(m.group(1))] = m.group(2)
    if sorted(rows) != list(range(1, len(rows) + 1)):
        raise SystemExit(f"{name}: the rows of {path} are not 1..{len(rows)}")
    return [rows[i] for i in range(1, len(rows) + 1)]


def vector(value):
    """A MATLAB array of the case as a list of ints (a scalar or empty one included)."""
    flat = value.flatten() if hasattr(value, "flatten") else [value]
    return [int(v) for v in flat]


def model(name, converted, cases):
    """One case: the equations, the sizes of powerSystem2cora, the bus sets and the voltages."""
    case = scipy.io.loadmat(
        os.path.join(cases, name + ".mat"), squeeze_me=True, struct_as_record=False
    )
    bus = case["bus"]
    generators, loads, inputs = len(vector(bus.generator)), len(vector(bus.load)), len(vector(bus.input))
    total = generators + loads
    dyn = equations(os.path.join(converted, name + "_dyn.m"), name)
    con = equations(os.path.join(converted, name + "_con.m"), name)
    sizes = dict(
        states=3 * generators,
        inputs=generators + 2 * inputs + total,
        constraints=2 * total - (1 if vector(bus.slack) else 0),
    )
    if len(dyn) != sizes["states"] or len(con) != sizes["constraints"]:
        raise SystemExit(f"{name}: the equations do not have the sizes {sizes}")
    out = {"name": name, **sizes, "dyn": dyn, "con": con}
    out["bus"] = {k: vector(getattr(bus, k)) for k in ("generator", "load", "slack", "input", "fault")}
    out["VM"] = [float(v) for v in case["VM"].flatten()]
    return out


# ============================================  MAIN  ============================================ #

def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    converted = sys.argv[1]
    cases = sys.argv[2] if len(sys.argv) > 2 else os.path.join(converted, "..", "powerSystems")
    data = {
        "format": "cora-benchmark-dae-1",
        "benchmark": "TSPS25",
        "models": [model(name, converted, cases) for name in CASES],
    }
    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, "TSPS25.json"), "w") as fh:
        json.dump(data, fh, indent=1)
        fh.write("\n")


if __name__ == "__main__":
    main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
