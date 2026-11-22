"""run_instance - one VNN-COMP instance: the verdict of the C++ verifier into the result file

The Python side of prepare_instance.sh and run_instance.sh (VNN-COMP's tool interface). It loads
the network and the property once (cora.nn_verify.load), runs the C++ cora.NeuralNetwork.verify
with the VNN-COMP preset of the category (cora.vnncomp_options) within the timeout (less the time spent
starting), and writes the result file: "sat" and the counterexample in the format of the
property's VNN-LIB version, "unsat", or "unknown". Next to it, RESULT.stats.json holds the time
and memory of the run (peak RSS, torch's peak on a GPU) and the counts of the attack and the
branch and bound. --prepare only checks that the files load. The timeout counts from
VNNCOMP_START (seconds since the epoch, set by run_instance.sh), else from the start of Python.

Syntax:   python scripts/vnncomp/run_instance.py CATEGORY ONNX VNNLIB RESULT TIMEOUT [--device D]
          python scripts/vnncomp/run_instance.py --prepare CATEGORY ONNX VNNLIB
See also: scripts/vnncomp/run_instance.sh, scripts/vnncomp/run_dev_set.py, cora.nn_verify

Authors:       Claude Opus 5.5
Written:       21-November-2026
Last update:   ---
Reviewed:      ---
"""

import argparse
import json
import os
import resource
import sys
import time

import cora
import cora.nn_verify
import torch


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--backend", default=os.environ.get("CORACPP_VNNCOMP_BACKEND", "eigen"),
                        help="the Tensor backend of the verifier: eigen or torch (cuda: torch)")
    parser.add_argument("category")
    parser.add_argument("onnx")
    parser.add_argument("vnnlib")
    parser.add_argument("result", nargs="?")
    parser.add_argument("timeout", nargs="?", type=float)
    args = parser.parse_args()
    start = float(os.environ.get("VNNCOMP_START", time.time() - time.process_time()))
    backend = "torch" if args.device.startswith("cuda") else args.backend
    options = options_for(args.category, args.device if backend == "torch" else "")
    load_start = time.monotonic()
    instance = cora.nn_verify.load(args.onnx, args.vnnlib, backend=backend, device=args.device)
    load_seconds = time.monotonic() - load_start
    if args.prepare:
        return 0
    # the budget: the timeout less the start and a safety margin
    budget = max(0.0, args.timeout - (time.time() - start) - SAFETY)
    try:
        result = cora.nn_verify.verify(instance, budget, options)
    except NotImplementedError as error:  # an operator or a property we do not support yet
        result = cora.nn_verify.VerifyResult("unknown", stats={"unsupported": str(error)})
    write_result(args.result, result)
    rss, cuda = peak_memory()
    stats = {
        k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in result.stats.items()
    }
    stats.update(
        status=result.status,
        load_seconds=load_seconds,
        budget=budget,
        seconds=time.time() - start,
        peak_rss_mib=rss,
        peak_cuda_mib=cuda,
    )
    with open(args.result + ".stats.json", "w") as out:
        json.dump(stats, out)
    return 0


# -----------------------------------------  AUXILIARY  ------------------------------------------ #

SAFETY = 1.0  # seconds kept for writing the result before the timeout


def options_for(category, device):
    """The verify options of a benchmark category: the VNN-COMP preset (cora.vnncomp_options, CORA's
    getDefaultVNNCOMPoptions with Phase 2's bounds) with the PGD attack around it, and the overrides
    of the environment variable CORACPP_VNNCOMP_OPTIONS (a JSON object of NnVerifyOptions fields)."""
    overrides = json.loads(os.environ.get("CORACPP_VNNCOMP_OPTIONS", "{}"))
    return cora.vnncomp_options(category, device=device, **overrides)


def write_result(path, result):
    """The result file of VNN-COMP: the verdict, then the counterexample for sat."""
    with open(path, "w") as out:
        out.write(result.status + "\n")
        if result.status == "sat":
            out.write(result.counterexample)


def peak_memory():
    """The peak resident memory of this process and torch's peak on a CUDA device, in MiB."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KiB on Linux
    cuda = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
    return rss, cuda


if __name__ == "__main__":
    sys.exit(main())

# ----------------------------------------  END OF CODE  ----------------------------------------- #
