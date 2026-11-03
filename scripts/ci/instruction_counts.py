"""instruction_counts - compares the instruction counts of the micro-benchmarks of two builds.

Syntax:   python scripts/ci/instruction_counts.py --base <micro> --head <micro> [--threshold 3]
Inputs:   two builds of benchmarks/micro (the program `micro`), one per commit, built in the same job
Method:   every benchmark runs under `valgrind --tool=callgrind`, counting the instructions of its
          measured region only (deterministic); without valgrind, under `perf stat -e instructions:u`
          (the whole process, noisy).
Output:   a table of base, head and the change in percent; exit code 1 if one grew by more than the
          threshold (percent, default 3), else 0. A benchmark the base does not have is listed as new.
Accepting a growth: a commit of the change (git log <base>..HEAD, with --accept-from <base>) names it
          in a line `Accept-instructions: <benchmark>: <reason>`; the reason is required and printed.
          The acceptance lives in the history of that one change, never in a list in the tree.
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
        if name not in base:
            lines.append(f"{name:28} {'-':>15} {head[name]:>15,} {'new':>8}")
            continue
        change = 100.0 * (head[name] - base[name]) / base[name] if base[name] else float("inf")
        lines.append(f"{name:28} {base[name]:>15,} {head[name]:>15,} {change:>+7.2f}%")
        if change > threshold:
            grown.append(name)
    return lines, grown


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", required=True, help="the micro program built against the base commit")
    parser.add_argument("--head", required=True, help="the micro program built against the head commit")
    parser.add_argument("--threshold", type=float, default=3.0, help="percent of growth that fails")
    parser.add_argument("--accept-from", help="base commit: growth named by `Accept-instructions:` in base..HEAD passes")
    args = parser.parse_args()
    names, known = names_of(args.head), set(names_of(args.base))
    # a benchmark the base does not have yet is listed as new, not compared
    base = {name: count(args.base, name) for name in names if name in known}
    head = {name: count(args.head, name) for name in names}
    lines, grown = compare(base, head, args.threshold)
    print("\n".join(lines))
    accepted = accepted_growth(args.accept_from) if args.accept_from else {}
    for name in [g for g in grown if g in accepted]:
        print(f"\naccepted growth of {name}: {accepted[name]}")
    grown = [g for g in grown if g not in accepted]
    if grown:
        print(f"\nmore than {args.threshold}% more instructions: {', '.join(grown)}")
        return 1
    return 0


ACCEPT = re.compile(r"^Accept-instructions:\s*([\w.-]+)\s*:\s*(\S.*)$", re.M)


def accepted_growth(base, messages=None):
    """{benchmark: reason} of the `Accept-instructions: <name>: <reason>` lines in the messages of
    the commits base..HEAD (or of `messages`, for tests); a line without a reason does not count."""
    if messages is None:
        messages = subprocess.run(["git", "log", "--format=%B", f"{base}..HEAD"], capture_output=True,
                                  text=True, check=True).stdout
    return {m.group(1): m.group(2).strip() for m in ACCEPT.finditer(messages)}


if __name__ == "__main__":
    sys.exit(main())
