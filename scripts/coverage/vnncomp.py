"""vnncomp - which CORA rows a VNN-COMP instance of each benchmark executes, as benchmarks.py for ARCH

The coverage build of the Python package (GCC --coverage, -O0, the C++ on Eigen, the CPU) runs one
instance per benchmark through scripts/vnncomp/run_instance.py, each on its own, and gcov says which
library files (src/, include/) executed a line. The rows and their status follow benchmarks.py: the
`// CORA:` tags of the files, the status of coverage/status.json. The instance of a benchmark is the
first row of its instances.csv (VNN-LIB 1.0, else 2.0), its timeout at most --timeout seconds.

Syntax:   python scripts/coverage/vnncomp.py --sets DIR --build DIR [--timeout S] [--only NAME]
          python scripts/coverage/vnncomp.py --files
Reads:    DIR/<benchmark>/<version>/instances.csv and its files, the coverage build of the Python
          package (its module, and the repository's .venv), the tags of the sources
Writes:   coverage/vnncomp.json (per benchmark: the instance, its verdict and the files it executed)
Checks:   --files: every measured file exists (pull request CI)
Exit code: 0 if all holds (or the file is written), else 1 with what differs.

Authors:       Claude Opus 5.5
Written:       27-November-2026
Last update:   ---
Reviewed:      ---
"""
import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = REPO / "coverage" / "vnncomp.json"

sys.path.insert(0, str(HERE))

import benchmarks as bm  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sets", help="the benchmarks: DIR/<benchmark>/<version>/instances.csv")
    parser.add_argument("--build", help="the coverage build of the Python package")
    parser.add_argument("--timeout", type=float, default=60.0, help="the most seconds per instance")
    parser.add_argument("--only", action="append", help="measure only these benchmarks (repeatable)")
    parser.add_argument("--files", action="store_true", help="check that every measured file exists")
    args = parser.parse_args()
    committed = json.loads(OUT.read_text(encoding="utf-8"))["instances"] if OUT.exists() else {}
    # --files: the check of pull requests, no build
    if args.files:
        gone = [f"  {name}: {f} no longer exists" for name, entry in committed.items()
                for f in entry.get("files", []) if not (REPO / f).exists()]
        if gone:
            print("coverage/vnncomp.json lists files that a change deleted or renamed; measure again:\n"
                  + "\n".join(gone))
        return 1 if gone else 0
    if not (args.sets and args.build):
        parser.error("--sets and --build are needed to measure")
    # each benchmark on its own; the others keep their committed measurement
    measured = dict(committed)
    for name in sorted(p.name for p in Path(args.sets).iterdir() if p.is_dir()):
        if args.only and name not in args.only:
            continue
        found = first_instance(Path(args.sets) / name)
        if found is None:
            print(f"{name}: no instance", flush=True)
            continue
        measured[name] = measure(Path(args.build).resolve(), name, *found, args.timeout)
        print(f"{name}: {measured[name].get('result', measured[name].get('skipped'))}, "
              f"{len(measured[name].get('files', []))} files", flush=True)
    OUT.write_text(json.dumps({"instances": dict(sorted(measured.items()))}, indent=1) + "\n",
                   encoding="utf-8")
    print(f"vnncomp: wrote {OUT.relative_to(REPO)}")
    return 0


# -----------------------------------------  AUXILIARY  ------------------------------------------ #

def first_instance(bench):
    """`(version directory, onnx field, vnnlib, timeout)` of the first instance, or None."""
    for version in ("1.0", "2.0"):
        table = bench / version / "instances.csv"
        if table.exists():
            with table.open(newline="", encoding="utf-8") as f:
                rows = [r for r in csv.reader(f) if r]
            if rows:
                # the onnx field may be VNN-LIB 2.0's list of networks, with commas: the timeout is last
                return bench / version, rows[0][0], rows[0][1], float(rows[0][-1])
    return None


def measure(build, name, base, onnx, vnnlib, timeout, cap):
    """`{"instance", "timeout", "result", "files"}`, or `{"instance", "skipped"}` if it failed."""
    for counter in build.rglob("*.gcda"):
        counter.unlink()
    limit = min(timeout, cap)
    instance = {"instance": [onnx, vnnlib], "timeout": limit}
    env = dict(os.environ, PYTHONPATH=str(build), CORACPP_NUM_THREADS="8", OMP_NUM_THREADS="8")
    # the paths of the instance relative to its version directory, as run_dev_set.py prefixes them
    with tempfile.TemporaryDirectory() as tmp:
        result = Path(tmp) / "result.txt"
        command = [str(REPO / ".venv" / "bin" / "python"), str(REPO / "scripts/vnncomp/run_instance.py"),
                   name, network_field(base, onnx), str(base / vnnlib.removeprefix("./")), str(result),
                   str(limit), "--device", "cpu"]
        # its own process group, so that the time limit stops the whole run
        process = subprocess.Popen(command, env=env, start_new_session=True,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        try:
            _, err = process.communicate(timeout=limit + 120)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            return {**instance, "skipped": f"over {limit + 120:g} s"}
        if process.returncode != 0 or not result.exists():
            return {**instance, "skipped": (err.strip().splitlines() or ["failed"])[-1][:200]}
        verdict = result.read_text(encoding="utf-8").split("\n")[0].strip()
    return {**instance, "result": verdict, "files": bm.gcov_files(build, REPO)}


def network_field(base, field):
    """The onnx field with the directory in front: a path, or each path of a list of networks."""
    from ast import literal_eval
    try:
        pairs = literal_eval(field.strip())
    except (SyntaxError, ValueError):
        pairs = None
    if not isinstance(pairs, list):
        return str(base / field.removeprefix("./"))
    return repr([(n, str(base / p.removeprefix("./"))) for n, p in pairs])


if __name__ == "__main__":
    sys.exit(main())

# ----------------------------------------  END OF CODE  ----------------------------------------- #
