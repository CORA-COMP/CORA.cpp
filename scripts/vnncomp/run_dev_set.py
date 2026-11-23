"""run_dev_set - runs a dev instance set through the VNN-COMP tool interface into a results CSV

Every instance of DIR/<version>/instances.csv runs as VNN-COMP's run_single_instance.sh runs it:
prepare_instance.sh, then run_instance.sh with the instance's timeout, each a process of its own,
killed at the timeout plus 60 s. A row of the CSV has the scoring's six columns (category, onnx,
vnnlib, prepare time, result, run time) and then ours: the instance's timeout, the peak RSS of
the run (MiB, from wait4), torch's peak on a GPU, the attack's and the branch and bound's
seconds, the boxes bounded and the bounding method. Result files go to OUT_DIR/results/.
--one-per-network runs the first instance of each network only: a memory sweep of a dev set.

Syntax:   python scripts/vnncomp/run_dev_set.py DIR --out results.csv [--version 2.0] [--jobs 3]
Inputs:   --max-timeout T (caps the timeouts), --limit N, --threads K, --tool DIR, --one-per-network
See also: scripts/vnncomp/score.py, scripts/vnncomp/make_dev_set.py

Authors:       Claude Opus 5.5
Written:       21-November-2026
Last update:   23-November-2026
Reviewed:      ---
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--version", default="2.0")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-timeout", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tool", default=HERE)
    parser.add_argument("--one-per-network", action="store_true")
    args = parser.parse_args()
    with open(os.path.join(args.dir, args.version, "instances.csv"), newline="") as stream:
        rows = [r for r in csv.reader(stream) if r]
    # the first instance of each network only, for a memory sweep
    if args.one_per_network:
        first = {}
        rows = [first.setdefault(r[0], r) for r in rows if r[0] not in first]
    if args.limit:
        rows = rows[: args.limit]
    args.results = os.path.join(os.path.dirname(os.path.abspath(args.out)), "results")
    os.makedirs(args.results, exist_ok=True)
    # the instances in parallel, the rows in the order of instances.csv
    with ThreadPoolExecutor(args.jobs) as pool:
        out = list(pool.map(lambda kr: run_one(args, *kr), enumerate(rows)))
    with open(args.out, "w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerows(out)
    with open(os.path.splitext(args.out)[0] + ".columns.txt", "w") as stream:
        stream.write(",".join(COLUMNS) + "\n")
    print(f"{len(out)} instances -> {args.out}")
    return 0


# -----------------------------------------  AUXILIARY  ------------------------------------------ #

HERE = os.path.dirname(os.path.abspath(__file__))

COLUMNS = [
    "category",
    "onnx",
    "vnnlib",
    "prepare_seconds",
    "result",
    "run_seconds",
    "timeout",
    "peak_rss_mib",
    "peak_cuda_mib",
    "attack_seconds",
    "bab_seconds",
    "boxes",
    "bounding",
]


def timed(command, limit, env):
    """(exit code, wall seconds, peak RSS in MiB) of a command killed after limit seconds."""
    start = time.monotonic()
    proc = subprocess.Popen(command, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    deadline = start + limit
    while True:
        pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
        if pid:
            break
        if time.monotonic() > deadline:
            proc.kill()
            pid, status, usage = os.wait4(proc.pid, 0)
            return 124, time.monotonic() - start, usage.ru_maxrss / 1024
        time.sleep(0.02)
    code = os.waitstatus_to_exitcode(status)
    return code, time.monotonic() - start, usage.ru_maxrss / 1024


def run_one(args, k, row):
    """The CSV row of one instance."""
    onnx, prop, timeout = row
    timeout = float(timeout)
    if args.max_timeout:
        timeout = min(timeout, args.max_timeout)
    base = os.path.join(args.dir, args.version)
    onnx, prop = (
        os.path.join(base, onnx.removeprefix("./")),
        os.path.join(base, prop.removeprefix("./")),
    )
    category = os.path.basename(os.path.normpath(args.dir))
    result_file = os.path.join(args.results, f"{k:05d}.txt")
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), CORACPP_NUM_THREADS=str(args.threads))
    # prepare (600 s at most), then the run, killed 60 s after its timeout
    tool = lambda name: os.path.join(args.tool, name)  # noqa: E731
    code, prepare, _ = timed([tool("prepare_instance.sh"), "v1", category, onnx, prop], 600, env)
    with open(result_file, "w") as out:
        out.write("no_result_in_file\n")
    command = [tool("run_instance.sh"), "v1", category, onnx, prop, result_file, str(timeout)]
    code, run, rss = timed(command, timeout + 60, env)
    # the result as run_single_instance.sh reads it
    if code == 124:
        result = "run_instance_timeout"
    elif code != 0:
        result = f"error_exit_code_{code}"
    else:
        with open(result_file) as stream:
            result = "".join(stream.readline().split()) or "no_result_in_file"
    # the scoring's six columns, then the run's own statistics
    stats = {}
    if os.path.exists(result_file + ".stats.json"):
        with open(result_file + ".stats.json") as stream:
            stats = json.load(stream)
    seconds = lambda key: f"{stats.get(key, 0.0):.3f}"  # noqa: E731
    return [category, onnx, prop, f"{prepare:.3f}", result, f"{run:.3f}", timeout, f"{rss:.1f}",
            f"{stats.get('peak_cuda_mib', 0.0):.1f}", seconds("attack_seconds"),
            seconds("bab_seconds"), stats.get("boxes", 0), stats.get("bounding", "")]


if __name__ == "__main__":
    sys.exit(main())

# ----------------------------------------  END OF CODE  ----------------------------------------- #
