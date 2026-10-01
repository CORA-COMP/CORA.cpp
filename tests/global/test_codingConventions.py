"""test_codingConventions - checks the coding conventions of CORA.cpp, as CORA's test of that name.

Syntax:   python tests/global/test_codingConventions.py
Checks:   every file of src/ starts its docstring with its own name; an operation (a file in a
          set's, a system's or the specification's folder) has Syntax, Inputs, Outputs and See
          also blocks; the code sits between the BEGIN CODE and END OF CODE markers, auxiliary functions
          (named aux_...) between the AUXILIARY and the MAIN marker, the operation below MAIN, which has two empty lines above it; no more than 25 lines run without a comment; long files have section lines; lines are at most 100
          characters; public names are camelCase and types PascalCase; every option switch tests
          all its options and ends in a descriptive error; every operation has a test; every
          example has its code between BEGIN CODE and END OF CODE, split into sections.
Exit code: 0 if all conventions hold, else 1, with the files and what is wrong.
"""
import glob
import os
import re
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
MAX_LINES_WITHOUT_COMMENT = 25
MAX_LINE_LENGTH = 100
BEGIN_MARKER = "// ----------------------------------------  BEGIN CODE  ---------------------------------------- //"
END_MARKER = "// ---------------------------------------  END OF CODE  ---------------------------------------- //"
AUX_MARKER = "// ----------------------------------------  AUXILIARY  ----------------------------------------- //"
MAIN_MARKER = "// ===========================================  MAIN  =========================================== //"
SECTION = re.compile(r"^\s*(//|#) [A-Z][A-Za-z ]+ -{10,}$")
# Directories whose .cpp files are operations, each in a folder named after its class.
OPERATION_ROOTS = ("src/contSet", "src/contDynamics", "src/specification", "src/nn")


def rel(path):
    return os.path.relpath(path, ROOT).replace(os.sep, "/")


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return f.read().replace("\r\n", "\n").split("\n")


def files(pattern):
    return sorted(glob.glob(os.path.join(ROOT, pattern), recursive=True))


def header_comment(lines):
    """The leading block of // comment lines."""
    block = []
    for line in lines:
        if not line.startswith("//"):
            break
        block.append(line)
    return block


def is_operation(path):
    r = rel(path)
    return path.endswith(".cpp") and r.startswith(OPERATION_ROOTS) and "/private/" not in r \
        and os.path.splitext(os.path.basename(path))[0] != os.path.basename(os.path.dirname(path))


# ---------------------------------------------------------------- the checks of a source file

def check_docstring(path, lines, issues):
    stem = os.path.splitext(os.path.basename(path))[0]
    block = header_comment(lines)
    if not block:
        issues.append("no docstring at the top")
        return
    if not re.match(rf"^// {re.escape(stem)} - \S", block[0]):
        issues.append(f"the docstring should start with '// {stem} - <summary>'")
    if len(block) > 1 and block[1] != "//":
        issues.append("an empty comment line should follow the first docstring line")
    text = "\n".join(block)
    if is_operation(path):
        for label in ("Syntax:", "Inputs:", "Outputs:", "See also:"):
            if label not in text:
                issues.append(f"the {label[:-1]} block is missing from the docstring")
        syntax = re.search(r"Syntax:(.*?)(?=\n// [A-Z][a-z]+( [a-z]+)?:|\Z)", text, re.S)
        if syntax and not re.search(rf"\b{re.escape(stem)}\(", syntax.group(1)):
            issues.append(f"the Syntax block should show a call of {stem}")


def check_markers(path, lines, issues, needs_main=True, library=True):
    """BEGIN CODE ... END OF CODE around the code, the auxiliary functions between the AUXILIARY
    and the MAIN marker, and the main operation below MAIN (sources only; headers have no MAIN)."""
    at = {name: [i for i, l in enumerate(lines) if l == m] for name, m in
          (("BEGIN", BEGIN_MARKER), ("END", END_MARKER), ("AUXILIARY", AUX_MARKER), ("MAIN", MAIN_MARKER))}
    is_source = path.endswith(".cpp")
    required = ["BEGIN", "END"] + (["MAIN"] if is_source and needs_main else [])
    for name in required:
        if len(at[name]) != 1:
            issues.append(f"exactly one {name} marker line is expected: "
                          f"{dict(BEGIN=BEGIN_MARKER, END=END_MARKER, MAIN=MAIN_MARKER)[name]}")
    for name in ("AUXILIARY", "MAIN"):
        if len(at[name]) > 1 or (name == "MAIN" and not is_source and at[name]):
            issues.append(f"unexpected {name} marker line(s)")
    if any(len(at[name]) != 1 for name in required):
        return
    begin, end = at["BEGIN"][0], at["END"][0]
    if begin > end:
        issues.append("BEGIN CODE should come before END OF CODE")
    if any(l.strip() for l in lines[end + 1:]):
        issues.append("nothing but empty lines should follow END OF CODE")
    for i, l in enumerate(lines[:begin]):
        if re.match(r"^(namespace|class|struct|template|inline|static|using|enum)\b", l):
            issues.append(f"line {i + 1}: code above BEGIN CODE; only the docstring and includes belong there")
            break
    main = at["MAIN"][0] if at["MAIN"] else None
    aux = at["AUXILIARY"][0] if at["AUXILIARY"] else None
    order = [x for x in (begin, aux, main, end) if x is not None]
    if order != sorted(order):
        issues.append("the markers should be in the order BEGIN CODE, AUXILIARY, MAIN, END OF CODE")
    if main is not None and not (main > 2 and lines[main - 1] == "" and lines[main - 2] == ""
                                 and lines[main - 3] != ""):
        issues.append("the MAIN marker line should have exactly two empty lines above it")
    for i in [begin, end] + [x for x in (aux, main) if x is not None]:
        if (i > 0 and lines[i - 1] != "") or (i + 1 < len(lines) and lines[i + 1] != ""):
            issues.append("a marker line should have an empty line above and below")
            break

    # The anonymous namespace is the auxiliary code: functions named aux_..., defined above MAIN.
    in_anon, anon_lines = False, []
    for i, line in enumerate(lines):
        if line == "namespace {":
            in_anon = True
            anon_lines.append(i)
        elif line.startswith("} // namespace") and not line.startswith("} // namespace cora"):
            in_anon = False
            anon_lines.append(i)
        elif in_anon and library and line and not line.startswith((" ", "//", "}", "using", "struct", "class", "enum", "#")):
            m = re.search(r"\b(\w+)\(", line)
            if not m or not line.rstrip().endswith(("{", ";", ",")):
                continue
            if not m.group(1).startswith("aux_"):
                issues.append(f"line {i + 1}: a function in an anonymous namespace should be called aux_{m.group(1)}")
            elif not line.rstrip().endswith("{"):
                issues.append(f"line {i + 1}: auxiliary functions are defined above MAIN, not declared ahead")
    if anon_lines and library:
        if aux is None:
            issues.append("an anonymous namespace is auxiliary code and needs the AUXILIARY marker line: " + AUX_MARKER)
        elif main is not None and any(i < aux or i > main for i in anon_lines):
            issues.append("the anonymous namespace should lie between the AUXILIARY and the MAIN marker")
    elif aux is not None:
        issues.append("an AUXILIARY marker without auxiliary code; remove it")


def check_sections(path, lines, issues):
    """A file of 120 lines or more is split by section lines ('// Title ------...'): one more
    than it has hundreds of lines."""
    if len(lines) < 120:
        return
    sections = sum(1 for line in lines if SECTION.match(line))
    needed = 1 + len(lines) // 100
    if sections < needed:
        issues.append(f"{len(lines)} lines need at least {needed} section lines "
                      f"('// Title ---...' up to column 100), found {sections}")


def check_comment_density(path, lines, issues):
    run = 0
    for i, line in enumerate(lines):
        run = 0 if "//" in line else run + 1
        if run > MAX_LINES_WITHOUT_COMMENT:
            issues.append(f"line {i + 1}: more than {MAX_LINES_WITHOUT_COMMENT} lines without a comment")
            return


def check_line_length(path, lines, issues):
    for i, line in enumerate(lines):
        if len(line) > MAX_LINE_LENGTH:
            issues.append(f"line {i + 1}: {len(line)} characters, at most {MAX_LINE_LENGTH}")


def check_naming(path, lines, issues):
    if path.endswith(".h"):
        for i, line in enumerate(lines):
            code = line.split("//")[0]
            # A trailing underscore marks a private member and is fine.
            for m in re.finditer(r"(?<![:\w])([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\(", code):
                if not m.group(1).startswith(("aux_", "priv_")):
                    issues.append(f"line {i + 1}: '{m.group(1)}' is not camelCase")
            for m in re.finditer(r"\b(?:struct|class|enum class)\s+(\w+)", code):
                if not re.match(r"^[A-Z][A-Za-z0-9]*$", m.group(1)):
                    issues.append(f"line {i + 1}: the type '{m.group(1)}' is not PascalCase")


def check_options(path, lines, issues):
    text = "\n".join(lines)
    option = re.search(r'==\s*"[^"]+"|==\s*\w+::\w+|\bswitch\s*\(', text)
    if option and not path.endswith(".h"):
        if "throw std::" not in text:
            issues.append("it tests an option but never throws: an unknown option should raise an error")
    for m in re.finditer(r'throw std::\w+\(\s*"((?:[^"\\]|\\.)*)"', text):
        if len(m.group(1)) < 20:
            issues.append(f"the error message '{m.group(1)}' is not descriptive")


def check_source(path):
    lines = read_lines(path)
    issues = []
    check_docstring(path, lines, issues)
    check_markers(path, lines, issues)
    check_comment_density(path, lines, issues)
    check_sections(path, lines, issues)
    check_line_length(path, lines, issues)
    check_naming(path, lines, issues)
    check_options(path, lines, issues)
    return issues


# --------------------------------------------------------------- tests and examples

def check_tests_exist(issues_by_file):
    for path in files("src/**/*.cpp"):
        if not is_operation(path):
            continue
        parts = rel(path).split("/")
        op, cls, area = os.path.splitext(parts[-1])[0], parts[-2], "/".join(parts[1:-2])
        pattern = f"tests/{area}/{cls}/test_{cls}_{op}*.cpp" if area else f"tests/{cls}/test_{cls}_{op}*.cpp"
        if not files(pattern):
            issues_by_file.setdefault(rel(path), []).append(f"no test: expected {pattern}")


def check_example(path):
    lines = read_lines(path)
    issues = []
    stem = os.path.splitext(os.path.basename(path))[0]
    comment = "//" if path.endswith(".cpp") else "#"
    first = lines[0].lstrip('"# /') if lines else ""
    if not first.startswith(stem):
        issues.append(f"the docstring should start with '{stem} - <summary>'")
    begin = BEGIN_MARKER if comment == "//" else "# -" + BEGIN_MARKER[3:-3] + "- #"
    end = END_MARKER if comment == "//" else "# -" + END_MARKER[3:-3] + "- #"
    if lines.count(begin) != 1 or lines.count(end) != 1:
        issues.append(f"exactly one BEGIN CODE and one END OF CODE marker line is expected: {begin}")
    elif lines.index(begin) > lines.index(end) or any(l.strip() for l in lines[lines.index(end) + 1:]):
        issues.append("BEGIN CODE comes before END OF CODE, and nothing follows END OF CODE")
    elif any(SECTION.match(l) for l in lines[:lines.index(begin)]):
        issues.append("the sections belong between BEGIN CODE and END OF CODE")
    if sum(1 for line in lines if SECTION.match(line)) < 2:
        issues.append(f"the example should be split into sections ('{comment} Parameters ----...')")
    return issues


def check_test_files(issues_by_file):
    for path in files("tests/**/test_*.cpp"):
        lines = read_lines(path)
        stem = os.path.splitext(os.path.basename(path))[0]
        if not lines or not lines[0].startswith(f"// {stem} - "):
            issues_by_file.setdefault(rel(path), []).append(f"a test should start with '// {stem} - <what it tests>'")
        for i, line in enumerate(lines):
            if len(line) > 110:
                issues_by_file.setdefault(rel(path), []).append(f"line {i + 1}: {len(line)} characters, at most 110")


def check_python_files(issues_by_file):
    """The Python package and its tests: a docstring naming the file (tests), and line lengths."""
    for path in files("src/global/python/cora/*.py") + files("tests/python/test_*.py"):
        lines = read_lines(path)
        found = []
        stem = os.path.splitext(os.path.basename(path))[0]
        is_test = "/tests/" in path.replace("\\", "/")
        if not lines or not lines[0].startswith('"""'):
            found.append("the module should start with a docstring")
        elif is_test and not lines[0].startswith(f'"""{stem} - '):
            found.append(f"the docstring should start with '{stem} - <what it tests>'")
        limit = 110 if is_test else MAX_LINE_LENGTH
        for i, line in enumerate(lines):
            if len(line) > limit:
                found.append(f"line {i + 1}: {len(line)} characters, at most {limit}")
        if found:
            issues_by_file.setdefault(rel(path), []).extend(found)


def check_python_tests_cover_the_api(issues_by_file):
    """Every class the package exports has a Python test file: tests/python/test_<class>*.py."""
    init = "\n".join(read_lines(os.path.join(ROOT, "src/global/python/cora/__init__.py")))
    for name in ("Zonotope", "Interval", "LinearSys", "NonlinearSys", "NeuralNetwork", "Specification"):
        if f"{name}" not in init:
            continue
        stem = name[0].lower() + name[1:]
        if not files(f"tests/python/test_{stem}*.py"):
            issues_by_file.setdefault("src/global/python/cora/__init__.py", []).append(
                f"no Python test for {name}: expected tests/python/test_{stem}*.py")


def main():
    issues_by_file = {}
    for path in files("src/**/*.cpp") + files("src/**/*.h"):
        if rel(path) == "src/global/python/bindings.cpp":
            found = []
            check_markers(path, read_lines(path), found, needs_main=False, library=False)
        else:
            found = check_source(path)
        if found:
            issues_by_file[rel(path)] = found
    check_tests_exist(issues_by_file)
    check_test_files(issues_by_file)
    check_python_files(issues_by_file)
    check_python_tests_cover_the_api(issues_by_file)
    for path in files("examples/cpp/*.cpp") + files("examples/python/example_*.py"):
        found = check_example(path)
        if found:
            issues_by_file[rel(path)] = found

    print("Checking the coding conventions of CORA.cpp ..")
    for path in sorted(issues_by_file):
        print(f"  {path}")
        for issue in issues_by_file[path]:
            print(f"    - {issue}")
    if issues_by_file:
        print(f"{sum(len(v) for v in issues_by_file.values())} issues in {len(issues_by_file)} files")
        return 1
    print("all conventions hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
