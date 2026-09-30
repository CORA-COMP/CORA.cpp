"""run_example - runs the open example, C++ or Python, the same way (the F5 of VS Code)

Syntax:   scripts/with_env.sh python scripts/run_example.py examples/<cpp|python>/<name>.<cpp|py> [args]
Runs:     a Python example after scripts/build.sh python, in this process, so that breakpoints in it work
          under the Python debugger; a C++ one built with scripts/build.sh <name>, then run.
"""
import os
import runpy
import subprocess
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
WITH_ENV = os.path.join(ROOT, "scripts", "with_env.sh")


def main(argv):
    if not argv:
        sys.exit("run_example: name an example, e.g. examples/python/example_linear_reach_01_5dim.py")
    path = os.path.abspath(argv[0])
    stem, extension = os.path.splitext(os.path.basename(path))
    if extension == ".py":
        if subprocess.call([WITH_ENV, "scripts/build.sh", "python"], cwd=ROOT) != 0:
            sys.exit("run_example: building the Python package failed")
        sys.argv = [path] + argv[1:]
        runpy.run_path(path, run_name="__main__")
    elif extension == ".cpp":
        if subprocess.call([WITH_ENV, "scripts/build.sh", stem], cwd=ROOT) != 0:
            sys.exit(f"run_example: building {stem} failed")
        build = os.environ.get("CORACPP_BUILD", os.path.join(ROOT, "build"))
        sys.exit(subprocess.call([os.path.join(build, "examples", "cpp", stem)], cwd=ROOT))
    else:
        sys.exit(f"run_example: '{os.path.basename(path)}' is neither a .py nor a .cpp example; "
                 "open an example first")


if __name__ == "__main__":
    main(sys.argv[1:])
