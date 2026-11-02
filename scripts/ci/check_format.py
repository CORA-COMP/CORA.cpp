"""check_format - the whitespace rules of the tracked text files, and that the workflows parse.

Syntax:   python scripts/ci/check_format.py
Checks:   no trailing whitespace, no carriage returns, no tabs in YAML, a final newline, and every
          .github/workflows/*.yml is valid YAML (when PyYAML is installed).
Exit code: 0 if all hold, else 1, with file:line and what is wrong.
"""
import os
import subprocess
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SUFFIXES = (".cpp", ".h", ".py", ".md", ".yml", ".yaml", ".cmake", ".txt", ".sh", ".json", ".in")


def problems(name, text):
    """The rule violations of one file's text, as 'name:line: what'."""
    found = []
    lines = text.split("\n")
    for number, line in enumerate(lines, 1):
        if line.endswith("\r"):
            found.append(f"{name}:{number}: carriage return")
        elif line != line.rstrip():
            found.append(f"{name}:{number}: trailing whitespace")
        if name.endswith((".yml", ".yaml")) and "\t" in line:
            found.append(f"{name}:{number}: tab in YAML")
    if text and not text.endswith("\n"):
        found.append(f"{name}: no final newline")
    return found


def tracked():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True).stdout
    return [f for f in out.splitlines() if f.endswith(SUFFIXES)]


def workflow_problems():
    try:
        import yaml
    except ImportError:
        return []
    found = []
    folder = os.path.join(ROOT, ".github", "workflows")
    for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        try:
            with open(os.path.join(folder, name), encoding="utf-8") as handle:
                yaml.safe_load(handle)
        except yaml.YAMLError as error:
            found.append(f".github/workflows/{name}: {str(error).splitlines()[0]}")
    return found


def main():
    found = []
    for name in tracked():
        if os.path.exists(os.path.join(ROOT, name)):
            with open(os.path.join(ROOT, name), encoding="utf-8", errors="replace", newline="") as handle:
                found += problems(name, handle.read())
    found += workflow_problems()
    print("\n".join(found[:200]) if found else "check_format: ok")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
