"""run_example - runs the open example, C++ or Python, the same way (the F5 of VS Code)

Syntax:   scripts/with_env.sh python scripts/run_example.py examples/<cpp|python>/<name>.<cpp|py> [args]
          (Windows: python scripts/run_example.py ..., with CORACPP_BUILD the preset's build folder)
Runs:     a Python example after scripts/build.sh python, in this process, so that breakpoints in it work
          under the Python debugger; a C++ one built with scripts/build.sh <name>, then run.
"""
import os
import runpy
import subprocess
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
WITH_ENV = os.path.join(ROOT, "scripts", "with_env.sh")
VSENV = os.path.join(ROOT, "scripts", "vsenv.cmd")
BUILD = os.environ.get("CORACPP_BUILD", os.path.join(ROOT, "build"))


def call(command):
    """Runs `command`; on Windows its output is forwarded, so that no console window of its own opens."""
    if os.name != "nt":
        return subprocess.call(command, cwd=ROOT)
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               creationflags=subprocess.CREATE_NO_WINDOW)
    for line in iter(process.stdout.readline, b""):
        sys.stdout.write(line.decode("utf-8", "replace"))
        sys.stdout.flush()
    return process.wait()


def build(target):
    """Builds `target` (a program name, or python) and returns the exit code."""
    if os.name == "nt":
        return call([VSENV, "cmake", "--build", BUILD, "--target", target])
    return call([WITH_ENV, "scripts/build.sh", target])


def main(argv):
    if not argv:
        sys.exit("run_example: name an example, e.g. examples/python/example_linear_reach_01_5dim.py")
    path = os.path.abspath(argv[0])
    stem, extension = os.path.splitext(os.path.basename(path))
    if extension == ".py":
        if build("python") != 0:
            sys.exit("run_example: building the Python package failed")
        sys.argv = [path] + argv[1:]
        runpy.run_path(path, run_name="__main__")
    elif extension == ".cpp":
        if build(stem) != 0:
            sys.exit(f"run_example: building {stem} failed")
        program = os.path.join(BUILD, "examples", "cpp", stem + (".exe" if os.name == "nt" else ""))
        sys.exit(call([program]))
    else:
        sys.exit(f"run_example: '{os.path.basename(path)}' is neither a .py nor a .cpp example; "
                 "open an example first")


if __name__ == "__main__":
    main(sys.argv[1:])
