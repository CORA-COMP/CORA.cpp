"""instruction_counts - compares the instruction counts of the micro-benchmarks of two builds.

Syntax:   python scripts/ci/instruction_counts.py --base <micro> --head <micro> [--threshold 3]
Inputs:   two builds of benchmarks/micro (the program `micro`), one per commit, built in the same job
Method:   every benchmark runs under `valgrind --tool=callgrind`, counting the instructions of its
          measured region only (deterministic); without valgrind, under `perf stat -e instructions:u`
          (the whole process, noisy).
Output:   a table of base, head and the change in percent; exit code 1 if one grew by more than the
          threshold (percent, default 3), else 0.
"""
import argparse
import re
import shutil
import subprocess
import sys

REGION = "micro_measured_region"
VALGRIND_COUNT = re.compile(r"Collected\s*:\s*(\d+)")
PERF_COUNT = re.compile(r"^\s*(\d+),[^,]*,instructions", re.M)


def parse_valgrind(text):
    """The instruction count of callgrind's summary (events collected inside the measured region)."""
    found = VALGRIND_COUNT.search(text)
    if not found:
        raise ValueError("no 'Collected' line in the output of callgrind")
    return int(found.group(1).replace(",", ""))


def parse_perf(text):
    """The instruction count of `perf stat -x,`."""
    found = PERF_COUNT.search(text)
    if not found:
        raise ValueError("no instructions line in the output of perf stat")
    return int(found.group(1))


def command(program, name):
    """The profiling command and the parser of its output."""
    if shutil.which("valgrind"):
        return ["valgrind", "--tool=callgrind", "--callgrind-out-file=/dev/null",
                f"--toggle-collect={REGION}", program, name], parse_valgrind
    if shutil.which("perf"):
        return ["perf", "stat", "-x,", "-e", "instructions:u", program, name], parse_perf
    raise SystemExit("instruction_counts: neither valgrind nor perf is installed")


def count(program, name):
    """The instructions of one benchmark of one build."""
    argv, parse = command(program, name)
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SystemExit(f"instruction_counts: {' '.join(argv)} failed:\n{result.stderr}")
    return parse(result.stderr)


def names_of(program):
    """The benchmarks a build of micro lists."""
    out = subprocess.run([program], capture_output=True, text=True, check=True).stdout
    return out.split()


def compare(base, head, threshold):
    """(table lines, the names that grew by more than threshold percent) of two {name: count}."""
    lines, grown = [f"{'benchmark':28} {'base':>15} {'head':>15} {'change':>8}"], []
    for name in sorted(head):
        change = 100.0 * (head[name] - base[name]) / base[name] if base.get(name) else float("inf")
        lines.append(f"{name:28} {base.get(name, 0):>15,} {head[name]:>15,} {change:>+7.2f}%")
        if change > threshold:
            grown.append(name)
    return lines, grown


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", required=True, help="the micro program built against the base commit")
    parser.add_argument("--head", required=True, help="the micro program built against the head commit")
    parser.add_argument("--threshold", type=float, default=3.0, help="percent of growth that fails")
    args = parser.parse_args()
    names = names_of(args.head)
    base = {name: count(args.base, name) for name in names}
    head = {name: count(args.head, name) for name in names}
    lines, grown = compare(base, head, args.threshold)
    print("\n".join(lines))
    if grown:
        print(f"\nmore than {args.threshold}% more instructions: {', '.join(grown)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
