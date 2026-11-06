"""affected_tests - which ctest tests a change can affect, as a regex for `ctest -R`.

Syntax:   python scripts/ci/affected_tests.py <base> <head>      the files changed between two commits
          python scripts/ci/affected_tests.py --files a.cpp b.h  the given files
Output:   a regex such as ^(cont_set/zonotope/test_zonotope_plus|conventions)$, or ALL (run every
          test), or NONE (nothing to run: the change is documentation only).
Method:   the #include graph of src/, include/, tests/ and examples/ gives, for every test and
          example, the files it depends on. A source file of the library is a dependency of every
          test that includes one of its own headers (the stem's header, else the headers of its
          mirrored include folder). A change to the tensor, the backends, the public global headers,
          the build (CMake, presets), CI, or to a file this script knows nothing about runs ALL.

Authors:       Tobias Ladner
Written:       02-November-2026
Last update:   ---
Reviewed:      ---
"""
import argparse
import os
import re
import subprocess
import sys


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SOURCE_DIRS = ("src", "include", "tests", "examples")
INCLUDE_ROOTS = ("include", "src", "tests")  # where a quoted include is looked up after its own folder
INCLUDE = re.compile(r'^\s*#\s*include\s*[<"]([^">]+)[">]', re.M)
# Changes that can affect anything: shared code, the build, the CI itself.
ALL_PATTERNS = (r"^src/global/(tensor|backend)/", r"^include/cora/global/", r"(^|/)CMakeLists\.txt$",
                r"\.cmake$", r"\.cmake\.in$", r"^CMakePresets\.json$", r"^\.github/", r"^scripts/ci/",
                r"^include/cora/config\.h\.in$", r"^tests/testing\.h$", r"^scripts/environment\.yml$",
                r"^scripts/requirements.*\.txt$")
# Changes no test of the core presets can see: documentation, the competition entry, other languages' examples.
NO_TEST_PATTERNS = (r"\.md$", r"^docs/", r"^LICENSE", r"^\.gitignore$", r"^scripts/competition/",
                    r"^tests/competition/", r"^examples/(matlab|benchmarks)/")
CPP_SUFFIXES = (".cpp", ".h")


def list_files():
    """Every C++ and Python file of the source folders, relative to ROOT with forward slashes."""
    files = []
    for folder in SOURCE_DIRS:
        for directory, _, names in os.walk(os.path.join(ROOT, folder)):
            for name in names:
                if name.endswith(CPP_SUFFIXES + (".py", ".in")):
                    files.append(os.path.relpath(os.path.join(directory, name), ROOT).replace(os.sep, "/"))
    return sorted(files)


def resolve(include, owner, files):
    """The file a #include names, or None (system headers, the generated cora/config.h resolves to its template)."""
    candidates = [os.path.normpath(os.path.join(os.path.dirname(owner), include))]
    candidates += [os.path.normpath(os.path.join(root, include)) for root in INCLUDE_ROOTS]
    for candidate in candidates:
        candidate = candidate.replace(os.sep, "/")
        for name in (candidate, candidate + ".in"):
            if name in files:
                return name
    return None


def includes_of(files, read):
    """{file: the files it includes directly}."""
    known = set(files)
    graph = {}
    for name in files:
        if name.endswith(CPP_SUFFIXES):
            found = (resolve(i, name, known) for i in INCLUDE.findall(read(name)))
            graph[name] = {f for f in found if f and f != name}
    return graph


def own_headers(source, files):
    """The public headers a library source file implements: its stem's header, else its folder's."""
    parts = source.split("/")[1:-1]  # src/<parts>/<stem>.cpp
    parts = [p for p in parts if p != "private"]
    stem = os.path.splitext(source.split("/")[-1])[0]
    folder = "include/cora/" + "/".join(parts)
    parent = "include/cora/" + "/".join(parts[:-1])
    stems = [f"{folder}/{stem}.h", f"{parent}/{stem}.h", f"{parent}/{parts[-1]}.h" if parts else "",
             os.path.splitext(source)[0] + ".h"]  # the last: a private header next to its source
    for header in stems:
        if header in files:
            return {header}
    return {f for f in files if f.endswith(".h") and os.path.dirname(f) == folder.rstrip("/")}


def test_name(source):
    """The ctest name tests/CMakeLists-style: tests/<folder>/<name>.cpp -> <folder>/<name>."""
    return os.path.splitext(source[len("tests/"):])[0]


def tests_of(files):
    """{ctest name: the root file of the test}."""
    found = {"conventions": "tests/global/test_coding_conventions.py"}
    for f in files:
        if f.startswith("tests/") and re.search(r"/test_[^/]+\.(cpp|py)$", f) and not f.startswith(
                ("tests/competition/", "tests/python/", "tests/global/test_coding_conventions.py")):
            found[test_name(f)] = f
        elif re.match(r"^examples/cpp/example_[^/]+\.cpp$", f):
            found["examples/" + os.path.splitext(os.path.basename(f))[0]] = f
    return found


def dependencies(files, read):
    """{ctest name: every file it depends on, itself included}."""
    graph = includes_of(files, read)
    sources = [f for f in graph if f.startswith("src/") and f.endswith(".cpp") and "/python/" not in f]
    provides = {s: own_headers(s, set(files)) for s in sources}

    def closure(roots, only=""):
        seen, stack = set(), list(roots)
        while stack:
            f = stack.pop()
            if f not in seen and f.startswith(only):
                seen.add(f)
                stack.extend(graph.get(f, ()))
        return seen

    result = {}
    for name, root in tests_of(files).items():
        visible = closure([root])  # what the test can name: a library source joins through these
        deps = set(visible)
        joined = set()
        while True:
            more = {s for s in sources if s not in joined and provides[s] & visible}
            if not more:
                break
            joined |= more
            for source in more:  # its private headers and what they include, its public includes as leaves
                private = closure([source], "src/")
                visible |= private
                deps |= private | graph[source]
        result[name] = deps
    return result


def matches(patterns, path):
    return any(re.search(p, path) for p in patterns)


def select(changed, files, read):
    """ALL, NONE or a regex of the ctest names affected by the changed files."""
    changed = [c for c in changed if not matches(NO_TEST_PATTERNS, c)]
    if any(matches(ALL_PATTERNS, c) for c in changed):
        return "ALL"
    known = set(files)
    deps = dependencies(files, read)
    chosen = set()
    for c in changed:
        if c.startswith(("tests/python/", "examples/python/")) or c == "scripts/run_example.py":
            chosen.add("python")
        elif c not in known:
            return "ALL"  # deleted, renamed or unknown: be safe
        else:
            if c.startswith(("src/", "include/")):
                chosen.add("python")  # the Python package wraps the whole library
            if c.startswith(("include/", "tests/install/")):
                chosen.add("install/consumer")
            chosen |= {t for t, d in deps.items() if c in d}
    if not chosen:
        return "NONE"
    chosen.add("conventions")
    return "^(" + "|".join(sorted(re.escape(t) for t in chosen)) + ")$"


def changed_files(base, head):
    """The files that differ between two commits (merge-base diff, as a pull request shows them)."""
    out = subprocess.run(["git", "diff", "--name-only", f"{base}...{head}"], cwd=ROOT, check=True,
                         capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if line]


# ============================================  MAIN  ============================================ #

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("revisions", nargs="*", help="<base> <head>")
    parser.add_argument("--files", nargs="+", help="changed files instead of two commits")
    args = parser.parse_args()
    if args.files:
        changed = args.files
    elif len(args.revisions) == 2:
        changed = changed_files(*args.revisions)
    else:
        parser.error("give <base> <head> or --files")

    def read(name):
        with open(os.path.join(ROOT, name), encoding="utf-8", errors="replace") as handle:
            return handle.read()

    print(select(changed, list_files(), read))


if __name__ == "__main__":
    sys.exit(main())

# ----------------------------------------  END OF CODE  ----------------------------------------- #
