"""check_quality - clang-format, clang-tidy and ruff over the changed files, or over everything.

Syntax:   python scripts/ci/check_quality.py format|tidy|lint [--base REV | --all] [--fix] [-j N]
Checks:   format  clang-format (.clang-format) on the C++ files: the file is what clang-format prints.
          tidy    clang-tidy (.clang-tidy) on the translation units of build/python/compile_commands.json
                  (--build-dir); a changed header is analysed through the one source that includes it.
          lint    ruff (ruff.toml) on scripts/, tests/ and src/global/python/cora/.
Scope:    --base REV: the files changed since the merge base with REV and in the working tree
          (default origin/main); --all: every tracked file (the nightly).
Options:  --fix rewrites the files (format, lint) or applies the fixes of clang-tidy (tidy).
Exit code: 0 if clean, 1 with the findings, 2 if a tool or the compile database is missing.

Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
Written:       04-November-2026
Last update:   ---
Reviewed:      ---
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import affected_tests  # noqa: E402  (the include graph, shared with the test selection)

ROOT = affected_tests.ROOT
CPP = (".cpp", ".h")
LINT_ROOTS = ("scripts/", "tests/", "src/global/python/cora/")
INSTALL = "pip install -r scripts/requirements-quality.txt (scripts/setup_local.sh does it)"


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.splitlines()


def candidates(base):
    """The files in scope, relative to ROOT: the tracked ones, or those changed since base."""
    if base is None:
        return git("ls-files")
    merge_base = git("merge-base", base, "HEAD")[0]
    changed = git("diff", "--name-only", "--diff-filter=d", merge_base)
    return [f for f in changed if os.path.exists(os.path.join(ROOT, f))]


def tool(name):
    path = shutil.which(name)
    if path is None:
        print(f"check_quality: {name} not found; {INSTALL}", file=sys.stderr)
        sys.exit(2)
    return path


def run(command, **kwargs):
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True, **kwargs)


def check_format(files, fix):
    files = [f for f in files if f.endswith(CPP)]
    if not files:
        return 0
    exe = tool("clang-format")
    for chunk in (files[i:i + 100] for i in range(0, len(files), 100)):
        result = run([exe, "-i" if fix else "--dry-run", "--Werror", *chunk])
        if result.returncode and not fix:
            print(result.stderr.strip())
            print("check_quality: format: run `python scripts/ci/check_quality.py format --fix`")
            return 1
    return 0


def check_lint(files, fix):
    files = [f for f in files if f.endswith(".py") and f.startswith(LINT_ROOTS)]
    if not files:
        return 0
    result = run([tool("ruff"), "check", "--force-exclude", "--no-cache", *(["--fix"] if fix else []), *files])
    print((result.stdout + result.stderr).strip())
    return 1 if result.returncode else 0


def translation_units(database):
    """{source file relative to ROOT} of the compile database (the build's own sources excluded)."""
    with open(database, encoding="utf-8") as handle:
        entries = json.load(handle)
    units = set()
    for entry in entries:
        path = os.path.relpath(entry["file"], ROOT).replace(os.sep, "/")
        if not path.startswith(("..", "build")):
            units.add(path)
    return units


def tidy_units(files, everything, units):
    """The sources to analyse: the changed ones, and for a changed header the one source that includes it."""
    if everything:
        return sorted(units)
    chosen = {f for f in files if f in units}
    headers = [f for f in files if f.endswith(".h")]
    if headers:
        known = affected_tests.list_files()
        graph = affected_tests.includes_of(known, read)
        for header in headers:
            users = sorted(u for u in units if u in graph and header in closure(graph, u))
            own = [u for u in users if stem(u) == stem(header)]
            if users and not chosen & set(users):
                chosen.add((own or users)[0])
    return sorted(chosen)


def stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def read(name):
    with open(os.path.join(ROOT, name), encoding="utf-8", errors="replace") as handle:
        return handle.read()


def closure(graph, root):
    seen, stack = set(), [root]
    while stack:
        node = stack.pop()
        if node not in seen:
            seen.add(node)
            stack.extend(graph.get(node, ()))
    return seen


def check_tidy(files, everything, fix, build_dir, jobs):
    database = os.path.join(build_dir, "compile_commands.json")
    if not os.path.exists(database):
        print(f"check_quality: {database} is missing; configure first: cmake --preset python", file=sys.stderr)
        return 2
    exe = tool("clang-tidy")
    units = tidy_units([f for f in files if f.endswith(CPP)], everything, translation_units(database))
    if not units:
        return 0

    def analyse(unit):
        # GCC's flags the compile database holds, and OpenMP (clang-tidy has no omp.h), are dropped
        extra = ["-extra-arg=-Wno-unknown-warning-option", "-extra-arg=-fno-openmp"]
        repair = ["--fix", "--warnings-as-errors=-*"] if fix else []  # a fix is not applied to an error
        result = run([exe, "-p", build_dir, "--quiet", *extra, *repair, unit])
        return unit, result.returncode, result.stdout

    findings = {}  # deduplicated: a header finding repeats in every source that includes it
    with ThreadPoolExecutor(jobs) as pool:
        for unit, code, output in pool.map(analyse, units):
            for block in re.split(r"(?m)^(?=\S+:\d+:\d+: (?:warning|error):)", output):
                if re.match(r"\S+:\d+:\d+: (?:warning|error):", block):
                    findings[block.split("\n")[0]] = block.rstrip()
            if code and not output.strip():
                findings[unit] = f"{unit}: clang-tidy failed"
    print("\n".join(findings.values()))
    print(f"check_quality: tidy: {len(units)} files, {len(findings)} findings")
    return 1 if findings else 0


# ============================================  MAIN  ============================================ #

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("what", choices=("format", "tidy", "lint"))
    parser.add_argument("--base", default="origin/main", help="the revision the change is measured from")
    parser.add_argument("--all", action="store_true", help="every tracked file, not only the changed ones")
    parser.add_argument("--fix", action="store_true")
    parser.add_argument("--build-dir", default=os.path.join(ROOT, "build", "python"))
    parser.add_argument("-j", "--jobs", type=int, help="parallel clang-tidy runs (default: all CPUs)",
                        default=int(os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", os.cpu_count() or 1)))
    args = parser.parse_args()
    files = candidates(None if args.all else args.base)
    if args.what == "format":
        return check_format(files, args.fix)
    if args.what == "lint":
        return check_lint(files, args.fix)
    return check_tidy(files, args.all, args.fix, args.build_dir, args.jobs)


if __name__ == "__main__":
    sys.exit(main())

# ----------------------------------------  END OF CODE  ----------------------------------------- #
