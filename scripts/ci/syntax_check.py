"""syntax_check - parse every translation unit of tests/ in a configured build, without compiling it.

Syntax:   python scripts/ci/syntax_check.py <build-dir> [-j N] [--under tests/]
          python scripts/ci/syntax_check.py <build-dir> --make-pch     precompile libtorch's header, print its path
Method:   the compile commands of <build-dir>/compile_commands.json for the files under the folder, with
          the output and the dependency files dropped and -fsyntax-only added, run N at a time. A libtorch
          build includes libtorch's headers in every translation unit, so a name that is ambiguous only
          there (detail::) shows up here, in a fraction of the time of a build.
Pch:      parsing libtorch's headers takes most of the time of a translation unit. --make-pch precompiles
          <torch/torch.h> with the flags of one test; configuring again with -include <that header> in
          CMAKE_CXX_FLAGS makes every later compile (this check and the build) read the precompiled one.
Exit code: 0 if all parse, else 1, with the compiler's messages of the first failures.

Authors:       Claude Sonnet 5.5
Written:       17-November-2026
Last update:   ---
Reviewed:      ---
"""
import argparse
import concurrent.futures
import json
import os
import shlex
import subprocess
import sys
import time


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("build")
    parser.add_argument("-j", type=int, default=os.cpu_count())
    parser.add_argument("--under", default="tests/")
    parser.add_argument("--make-pch", action="store_true")
    args = parser.parse_args()
    # the entries of the folder, from the compile database that the configure step wrote
    with open(os.path.join(args.build, "compile_commands.json"), encoding="utf-8") as stream:
        entries = json.load(stream)
    if args.make_pch:
        print(make_pch(entries, args.build))
        return 0
    marker = os.sep + args.under.replace("/", os.sep)
    entries = [e for e in entries if marker in e["file"]]
    # parse them side by side; the first failures are printed in full
    start = time.time()
    failed = []
    with concurrent.futures.ThreadPoolExecutor(args.j) as pool:
        for name, output in pool.map(run, entries):
            if output:
                failed.append(name)
                if len(failed) <= 5:
                    print(output[:3000])
    print(f"syntax_check: {len(entries)} files in {time.time() - start:.0f} s, {len(failed)} failed")
    return 1 if failed else 0


# -----------------------------------------  AUXILIARY  ------------------------------------------ #

DROPPED_WITH_VALUE = ("-o", "-MF", "-MT", "-MQ")

DROPPED = ("-c", "-MD", "-MMD", "-MP")


def syntax_command(entry):
    """The compile command of one entry as an argument list that only parses its file."""
    arguments = shlex.split(entry["command"])
    kept = []
    skip = False
    for argument in arguments:
        if skip:
            skip = False
        elif argument in DROPPED_WITH_VALUE:
            skip = True
        elif argument not in DROPPED:
            kept.append(argument)
    return kept + ["-fsyntax-only"]


def make_pch(entries, build):
    """Precompiles <torch/torch.h> with the flags of the first test of the compile database; its header's path."""
    entry = next(e for e in entries if os.sep + "tests" + os.sep in e["file"])
    header = os.path.join(os.path.abspath(build), "torch_pch.h")
    with open(header, "w", encoding="utf-8") as stream:
        stream.write("#include <torch/torch.h>\n")
    arguments = [a for a in syntax_command(entry)[:-1] if a != entry["file"]]
    subprocess.run(arguments + ["-x", "c++-header", header, "-o", header + ".gch"], cwd=entry["directory"], check=True)
    return header


def run(entry):
    """(file, the compiler's output) of one translation unit; the output is empty when it parses."""
    result = subprocess.run(syntax_command(entry), cwd=entry["directory"], capture_output=True, text=True, check=False)
    return entry["file"], ("" if result.returncode == 0 else result.stdout + result.stderr)


if __name__ == "__main__":
    sys.exit(main())

# ----------------------------------------  END OF CODE  ----------------------------------------- #
