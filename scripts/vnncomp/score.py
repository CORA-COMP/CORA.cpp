"""score - scores a results CSV of a dev instance set by the VNN-COMP 2026 rules

The rules of vnncomp2026_results/SCORING/process_results.py (get_score) for one tool: a correct
holds or violated is +10, an incorrect one -150, the rest 0. A violated counts if its
counterexample passes the scoring's check (cora.counterexample, CORRECT or CORRECT_UP_TO_TOLERANCE);
a missing or failing one is -150. A holds is incorrect if the expected verdicts (expected.csv of
the dev set) say sat, or a counterexample of another run is CORRECT. A solve whose run time, less
the overhead (the fastest solve), exceeds the instance's timeout is a timeout. Without other tools
there is no normalization to the best tool: the score is reported with its maximum, 10 per instance.

Syntax:   python scripts/vnncomp/score.py results.csv [--expected DIR/expected.csv] [--json out.json]
Inputs:   --max-rss-mib M: the memory check (an instance above M is listed; exit status 1)
Outputs:  per category: verified, falsified, unknown, incorrect, score, maximum, the times and peaks
See also: scripts/vnncomp/run_dev_set.py, cora.counterexample

Authors:       Claude Opus 5.5
Written:       21-November-2026
Last update:   23-November-2026
Reviewed:      ---
"""
import argparse
import csv
import json
import os
import sys

import cora.counterexample


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("csv")
    parser.add_argument("--expected", default="")
    parser.add_argument("--json", default="")
    parser.add_argument("--max-rss-mib", type=float, default=0.0)
    args = parser.parse_args()
    # the rows, the expected verdicts, the result files beside the CSV
    with open(args.csv, newline="") as stream:
        rows = [r for r in csv.reader(stream) if r]
    expected = {}
    if args.expected and os.path.exists(args.expected):
        with open(args.expected, newline="") as stream:
            expected = {(os.path.basename(o), os.path.basename(p)): v for o, p, v in csv.reader(stream)}
    results_dir = os.path.join(os.path.dirname(os.path.abspath(args.csv)), "results")
    scores = score_rows(rows, results_dir, expected)
    # one line per category
    for cat, s in scores.items():
        print(f"{cat}: {s['verified']} verified, {s['falsified']} falsified, {s['unknown']} unknown, "
              f"{s['incorrect']} incorrect; score {s['score']} of {s['maximum']} "
              f"({100.0 * s['score'] / max(1, s['maximum']):.1f}%); {s['run_seconds']:.0f} s, "
              f"peak RSS {s['peak_rss_mib']:.0f} MiB; checks {s['checks']}")
    # the memory check: every instance within the budget
    over = over_memory(rows, args.max_rss_mib) if args.max_rss_mib else []
    for onnx, prop, rss in over:
        print(f"over {args.max_rss_mib:.0f} MiB: {rss:.0f} MiB, {os.path.basename(onnx)} "
              f"{os.path.basename(prop)}")
    if args.json:
        with open(args.json, "w") as out:
            json.dump(scores, out, indent=2)
    return 1 if over else 0


# -----------------------------------------  AUXILIARY  ------------------------------------------ #

PENALTY_INCORRECT = -150

SUBSTITUTIONS = [("unsat", "holds"), ("sat", "violated"), ("timed-out", "timeout"),
                 ("no_result_in_file", "unknown"), ("prepare_instance_error_", "unknown"),
                 ("run_instance_timeout", "timeout"), ("prepare_instance_timeout", "timeout"),
                 ("error_exit_code_", "error"), ("error_nonmaximal", "unknown")]


def normalized(result):
    """The scoring's CSV_SUBSTITUTIONS: the first prefix that matches."""
    result = result.lower()
    if not result:
        return "unknown"
    for prefix, name in SUBSTITUTIONS:
        if result.startswith(prefix):
            return name
    return result


def counterexample_text(results_dir, k):
    """The counterexample of row k: the result file without its first line."""
    path = os.path.join(results_dir, f"{k:05d}.txt")
    if not os.path.exists(path):
        return ""
    with open(path) as stream:
        return "".join(stream.readlines()[1:])


def over_memory(rows, limit):
    """The rows (onnx, vnnlib, peak RSS) whose peak RSS in MiB exceeds limit."""
    return [(r[1], r[2], float(r[7])) for r in rows if len(r) > 7 and float(r[7]) > limit]


def score_rows(rows, results_dir, expected):
    """Per category the counts and the score of the rows."""
    solved = [float(r[5]) for r in rows if normalized(r[4]) in ("holds", "violated")]
    overhead = min(solved) if solved else 0.0
    out = {}
    for k, r in enumerate(rows):
        cat, onnx, prop, result = r[0], r[1], r[2], normalized(r[4])
        s = out.setdefault(cat, {"instances": 0, "verified": 0, "falsified": 0, "unknown": 0,
                                 "incorrect": 0, "score": 0, "run_seconds": 0.0,
                                 "peak_rss_mib": 0.0, "checks": {}})
        s["instances"] += 1
        s["run_seconds"] += float(r[5])
        if len(r) > 7:
            s["peak_rss_mib"] = max(s["peak_rss_mib"], float(r[7]))
        timeout = float(r[6]) if len(r) > 6 else float("inf")
        if result in ("holds", "violated") and float(r[5]) - overhead > timeout:
            result = "timeout"
        truth = expected.get((os.path.basename(onnx), os.path.basename(prop)))
        # the verdict against its check
        if result == "violated":
            check, _ = cora.counterexample.check_counterexample(onnx, prop, counterexample_text(results_dir, k))
            s["checks"][check] = s["checks"].get(check, 0) + 1
            good = check in (cora.counterexample.CORRECT, cora.counterexample.CORRECT_UP_TO_TOLERANCE)
            s["falsified" if good else "incorrect"] += 1
            s["score"] += 10 if good else PENALTY_INCORRECT
        elif result == "holds":
            good = truth != "sat"
            s["verified" if good else "incorrect"] += 1
            s["score"] += 10 if good else PENALTY_INCORRECT
        else:
            s["unknown"] += 1
    for s in out.values():
        s["maximum"] = 10 * s["instances"]
        s["overhead_seconds"] = overhead
    return out


if __name__ == "__main__":
    sys.exit(main())

# ----------------------------------------  END OF CODE  ----------------------------------------- #
