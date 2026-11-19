"""lines - line coverage of the library by the unit and the oracle tests, from gcov

Syntax:   python scripts/coverage/lines.py run --build-dir <dir> --suite unit|oracle --out <file>
          python scripts/coverage/lines.py report <file>... [--summary]
          python scripts/coverage/lines.py changed <file>... --base <ref> [--min 90]
          python scripts/coverage/lines.py ratchet <file>... [--update]
Inputs:   run - a build of the preset `coverage` with the test programs of the suite
          report, changed, ratchet - the files that `run` wrote, one per suite
Outputs:  the lines of src/ and include/ per file (run), a table per suite and combined (report)
See also: scripts/coverage/benchmarks.py (files, not lines, per benchmark), CONTRIBUTING.md

`changed` and `ratchet` exit with 1 below their limits. A suite is a ctest label: `run` clears the
counters, runs the label, and reads them with `gcov --json-format`, also for the objects that no test
program of the suite links (their lines count as not executed). The benchmarks and the examples are not
measured. `changed` asks that the lines a branch adds under src/ and include/ are covered (executable
lines only, by either suite); `ratchet` compares the totals with coverage/lines.json, which only rises.

Authors:       Claude Sonnet 5.5
Written:       16-November-2026
Last update:   ---
Reviewed:      ---
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "coverage" / "lines.json"
TOLERANCE = 0.1  # percentage points that a run may lose against the baseline: the tests of threads
FILE_LIMIT = 70.0  # a file below it counts as thin
MIN_FILE_LINES = 10  # smaller files are never thin: one gcov line on a class head moves 3 lines to 67%


def read_gcov(text, root):
    """{file: (executable lines, executed lines)} of the `gcov --json-format --stdout` documents in
    `text`, for the files under src/ and include/ of `root`."""
    # the documents follow one another in the output of gcov
    decoder, position, files = json.JSONDecoder(), 0, {}
    root = Path(root).resolve()
    while True:
        position = text.find("{", position)
        if position < 0:
            return {name: (sorted(a), sorted(b)) for name, (a, b) in files.items()}
        document, position = decoder.raw_decode(text, position)
        base = Path(document.get("current_working_directory", "."))
        for entry in document.get("files", []):
            path = Path(entry["file"])
            try:
                name = (path if path.is_absolute() else base / path).resolve().relative_to(root).as_posix()
            except ValueError:
                continue
            if not name.startswith(("src/", "include/")):
                continue
            # a line is executable if gcov lists it, executed if some document counts it
            executable, executed = files.setdefault(name, (set(), set()))
            for line in entry.get("lines", []):
                executable.add(line["line_number"])
                if line.get("count", 0) > 0:
                    executed.add(line["line_number"])


def merge(measurements):
    """The union of several {file: (executable, executed)} maps: a line is executed if any run did."""
    out = {}
    for files in measurements:
        for name, (executable, executed) in files.items():
            a, b = out.setdefault(name, (set(), set()))
            a.update(executable)
            b.update(executed)
    return {name: (sorted(a), sorted(b)) for name, (a, b) in out.items()}


def totals(files):
    """(executed, executable, thin files) of a map: the lines, and the files of at least
    MIN_FILE_LINES lines below FILE_LIMIT."""
    executed = sum(len(b) for _, b in files.values())
    executable = sum(len(a) for a, _ in files.values())
    thin = sum(1 for a, b in files.values()
               if len(a) >= MIN_FILE_LINES and 100.0 * len(b) / len(a) < FILE_LIMIT)
    return executed, executable, thin


def percent(executed, executable):
    return 100.0 * executed / executable if executable else 100.0


def load(paths):
    """{suite: {file: (executable, executed)}} from the files `run` wrote."""
    out = {}
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        out[data["suite"]] = {n: (v["executable"], v["executed"]) for n, v in data["files"].items()}
    return out


def added_lines(base, root):
    """{file: set of added line numbers} of the diff from `base` to the working tree HEAD, under
    src/ and include/ (only C++ sources and headers)."""
    diff = subprocess.run(["git", "diff", "-U0", "--no-color", f"{base}...HEAD", "--", "src", "include"],
                          cwd=root, text=True, capture_output=True, check=True).stdout
    out, name = {}, None
    for line in diff.split("\n"):
        if line.startswith("+++ b/"):
            name = line[6:]
        elif line.startswith("@@") and name and name.endswith((".cpp", ".h")):
            m = re.search(r"\+(\d+)(?:,(\d+))?", line)
            start, count = int(m.group(1)), int(m.group(2) or 1)
            out.setdefault(name, set()).update(range(start, start + count))
    return out


def baseline_numbers(suites):
    """What the ratchet keeps per suite and combined: the percentage and the thin files."""
    result = {}
    for name, files in {**suites, "combined": merge(suites.values())}.items():
        executed, executable, thin = totals(files)
        result[name] = {"percent": round(percent(executed, executable), 2), "thin_files": thin}
    return result


def command_run(args):
    """Clears the counters of the build, runs one label, writes the lines it executed."""
    build = Path(args.build_dir).resolve()
    for counter in build.rglob("*.gcda"):
        counter.unlink()
    ran = subprocess.run(["ctest", "--test-dir", str(build), "-L", f"^{args.suite}$", "-j", str(args.jobs),
                          "--output-on-failure"], text=True)
    # an unlinked object has no counters, but its lines are the library's: they count as not executed
    notes = sorted(build.rglob("*.gcda"))
    notes += [note for note in sorted(build.rglob("*.gcno")) if not note.with_suffix(".gcda").exists()]
    files = {}
    for start in range(0, len(notes), 100):
        out = subprocess.run(["gcov", "--json-format", "--stdout", *map(str, notes[start:start + 100])],
                             text=True, capture_output=True, cwd=build).stdout
        files = merge([files, read_gcov(out, ROOT)])
    data = {"suite": args.suite, "files": {n: {"executable": a, "executed": b} for n, (a, b) in files.items()}}
    Path(args.out).write_text(json.dumps(data, sort_keys=True) + "\n", encoding="utf-8")
    executed, executable, thin = totals(files)
    print(f"{args.suite}: {executed}/{executable} lines ({percent(executed, executable):.1f}%), "
          f"{thin} files below {FILE_LIMIT:.0f}%; the tests {'passed' if ran.returncode == 0 else 'FAILED'}")
    return 0


def command_report(args):
    """The lines per suite and combined, as a Markdown table."""
    suites = load(args.files)
    rows = [("suite", "lines", "covered", f"files below {FILE_LIMIT:.0f}%")]
    for name, files in {**suites, "combined": merge(suites.values())}.items():
        executed, executable, thin = totals(files)
        rows.append((name, str(executable), f"{executed} ({percent(executed, executable):.1f}%)", str(thin)))
    print("\n".join("| " + " | ".join(r) + " |" for r in rows[:1] + [("---",) * 4] + rows[1:]))
    return 0


def command_changed(args):
    """Fails if the lines that the branch adds are covered by less than --min percent."""
    files = merge(load(args.files).values())
    covered = total = 0
    missed = []
    for name, lines in sorted(added_lines(args.base, ROOT).items()):
        executable, executed = (set(x) for x in files.get(name, ([], [])))
        for line in sorted(lines & executable):
            total += 1
            if line in executed:
                covered += 1
            else:
                missed.append(f"{name}:{line}")
    share = percent(covered, total)
    print(f"changed lines: {covered}/{total} covered ({share:.1f}%), the limit is {args.min:.0f}%")
    if share < args.min:
        print("not covered: " + ", ".join(missed[:60]) + (" ..." if len(missed) > 60 else ""))
        return 1
    return 0


def command_ratchet(args):
    """Fails if a suite lost more than TOLERANCE points, or has more thin files, than the baseline."""
    now = baseline_numbers(load(args.files))
    if args.update:
        old = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else {}
        # the baseline never falls: it keeps the higher percentage and the fewer thin files
        new = {k: {"percent": max(v["percent"], old.get(k, {}).get("percent", 0)),
                   "thin_files": min(v["thin_files"], old.get(k, {}).get("thin_files", v["thin_files"]))}
               for k, v in now.items()}
        BASELINE.write_text(json.dumps(new, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"lines: wrote {BASELINE}")
        return 0
    # without a baseline nothing can fall
    if not BASELINE.exists():
        print("lines: no baseline yet (run with --update on a measured main)")
        return 0
    old = json.loads(BASELINE.read_text(encoding="utf-8"))
    bad = []
    for name, base in old.items():
        if name not in now:
            continue
        if now[name]["percent"] < base["percent"] - TOLERANCE:
            bad.append(f"{name}: {now[name]['percent']}% lines, the baseline is {base['percent']}%")
        if now[name]["thin_files"] > base["thin_files"]:
            bad.append(f"{name}: {now[name]['thin_files']} files below {FILE_LIMIT:.0f}%, the baseline "
                       f"has {base['thin_files']}")
    for line in bad:
        print("lines: " + line)
    return 1 if bad else 0


# ============================================  MAIN  ============================================ #

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--build-dir", default="build/coverage")
    run.add_argument("--suite", choices=("unit", "oracle"), required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--jobs", type=int, default=4)
    sub.add_parser("report").add_argument("files", nargs="+")
    changed = sub.add_parser("changed")
    changed.add_argument("files", nargs="+")
    changed.add_argument("--base", required=True)
    changed.add_argument("--min", type=float, default=90.0)
    # the commands and their arguments
    ratchet = sub.add_parser("ratchet")
    ratchet.add_argument("files", nargs="+")
    ratchet.add_argument("--update", action="store_true")
    args = parser.parse_args()
    return {"run": command_run, "report": command_report, "changed": command_changed,
            "ratchet": command_ratchet}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

# ----------------------------------------  END OF CODE  ----------------------------------------- #
