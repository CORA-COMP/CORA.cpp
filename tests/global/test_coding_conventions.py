"""test_coding_conventions - checks the coding conventions of CORA.cpp, as CORA's test of that name.

Syntax:   python tests/global/test_coding_conventions.py
Checks:   layout - every file starts its docstring with its own name; an operation (a file in a
          set's, a system's or the specification's folder) has Syntax, Inputs, Outputs and See also
          blocks; the code sits between BEGIN CODE and END OF CODE, auxiliary functions (aux_...)
          between AUXILIARY and MAIN; at most 25 lines without a comment, 100 characters a line;
          long files have section lines;
          naming - files, functions, variables and members snake_case, types PascalCase; every
          operation and set class names its CORA file (// CORA: <path>.m) or why it has none
          (// CORA.cpp only: <reason>);
          checks - a public operation starts with the guard (detail::Entry entry;); CORA_CHECK
          only in a guarded function, never in an _impl or under private/; no raw throw outside
          global/error.cpp (CORA_CHECK for input, CORA_FAIL for failures); an option switch ends
          in an error;
          tests - every operation has a test, and its tests call it with an invalid input
          (test::raises with the identifier) and with the checks off (ChecksScope); every
          example has its code between BEGIN CODE and END OF CODE, split into sections;
          one code path - no branch on the backend outside global/backend and global/tensor, and
          no duplicated code (token clones of CLONE_TOKENS tokens or more);
          the data decides - library code makes tensors from raw numbers on the current backend
          (Tensor::zeros, from_data, Tensor({...})) only in a raw-number entry point, a function
          marked CORA_FROM_NUMBERS, or in the tensor, backend and binding folders;
          tensor operations - every operation of ops.def that passes gradients has a gradcheck
          test, and docs/tensor.md lists every operation with the tags of ops.def.
Exit code: 0 if all conventions hold, else 1, with the files and what is wrong.
"""
import glob
import os
import re
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
MAX_LINES_WITHOUT_COMMENT = 25
MAX_LINE_LENGTH = 100
CLONE_TOKENS = 100  # this many tokens repeated (literals abstracted) is duplicated code
BEGIN_MARKER = "// ----------------------------------------  BEGIN CODE  ---------------------------------------- //"
END_MARKER = "// ---------------------------------------  END OF CODE  ---------------------------------------- //"
AUX_MARKER = "// ----------------------------------------  AUXILIARY  ----------------------------------------- //"
MAIN_MARKER = "// ===========================================  MAIN  =========================================== //"
SECTION = re.compile(r"^\s*(//|#) [A-Z][A-Za-z ]+ -{10,}$")
# Directories whose .cpp files are operations, each in a folder named after its class.
OPERATION_ROOTS = ("src/cont_set/", "src/cont_dynamics/", "src/specification/", "src/nn/")
# Folders of the public headers of the set classes, systems and specifications (CORA tag).
CLASS_HEADER_ROOTS = ("include/cora/cont_set/", "include/cora/cont_dynamics/",
                      "include/cora/specification/", "include/cora/nn/")
# Operations that take no input, so no test can give them an invalid one.
NO_INPUT_OPERATIONS = {"display", "interval"}
# Not an operation of a class: the table of the binary operations.
NOT_OPERATIONS = {"src/cont_set/cont_set/registry.cpp"}
CORA_TAG = re.compile(r"^// (CORA: \S+\.m\b|CORA\.cpp only: \S)")
# The Python bridge for MATLAB CORA keeps CORA's names (intervalMatrix, linearSys, timeInt):
# toPy/fromPy of MATLAB CORA call them.
CORA_NAMED_PYTHON = {"src/global/python/cora/matlab.py"}
# Where code may ask which backend, device or dtype a tensor has.
BACKEND_FOLDERS = ("src/global/backend/", "src/global/tensor/", "include/cora/global/backend/",
                   "include/cora/global/tensor/", "src/global/python/",
                   "include/cora/cora.h")  # the umbrella header includes the torch backend
# camelCase names of libraries the code calls (Eigen, libtorch, pybind11, the standard library).
FOREIGN_NAMES = {
    "cwiseAbs", "cwiseMax", "cwiseMin", "cwiseProduct", "rowwise", "colwise", "setZero", "setOnes",
    "setIdentity", "maxCoeff", "minCoeff", "squaredNorm", "lpNorm", "asDiagonal", "isApprox",
    "isZero", "isFinite", "hasNaN", "selfadjointView", "triangularView", "unaryExpr", "binaryExpr",
    "fullPivLu", "partialPivLu", "colPivHouseholderQr", "householderQr", "jacobiSvd", "bdcSvd",
    "matrixU", "matrixV", "singularValues", "eigenvalues", "eigenvectors", "isfinite", "isnan",
    "isinf", "def_static", "def_property_readonly", "def_readonly", "def_readwrite",
    "noalias", "innerSize", "outerSize", "middleRows", "middleCols", "topRows", "bottomRows",
    "leftCols", "rightCols", "topLeftCorner", "bottomRightCorner", "segment", "conservativeResize",
    "mutable_data", "requires_grad_", "manualSeed", "manual_seed", "numpyArray", "setNbThreads",
    "nbThreads",
}


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


def is_operation_file(r):
    """Whether a file, given by its path relative to the repository, is a public operation."""
    return r.endswith(".cpp") and r.startswith(OPERATION_ROOTS) and "/private/" not in r \
        and os.path.splitext(os.path.basename(r))[0] != os.path.basename(os.path.dirname(r))


def needs_cora_tag_file(r):
    """Whether a file (relative path) names its CORA file: an operation or a class header."""
    return is_operation_file(r) or (r.startswith(CLASS_HEADER_ROOTS) and r.endswith(".h"))


def is_operation(path):
    return is_operation_file(rel(path))


def needs_cora_tag(path):
    return needs_cora_tag_file(rel(path))


def code_only(text):
    """The text with comments and string and character literals blanked (lines are kept)."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(re.sub(r"[^\n]", " ", text[i:j]))
            i = j
        elif c == '"' or (c == "'" and not (i > 0 and text[i - 1].isalnum())):
            j = i + 1
            while j < n and text[j] != c and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            out.append(c + " " * max(0, min(j, n) - i - 1) + (c if j < n else ""))
            i = j + 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def functions(code_lines):
    """The functions defined at namespace level: (name, first line, last line) with lines
    0-based; a definition starts at column 0 and its body closes with '}' at column 0."""
    found, i = [], 0
    while i < len(code_lines):
        line = code_lines[i]
        m = re.match(r"^[A-Za-z].*?([~\w]+)\s*\(", line)
        if m and not re.match(r"^(namespace|struct|class|enum|using|template|typedef|return|if|for|while)\b", line):
            # the body opens on this line or a following signature line
            j = i
            while j < len(code_lines) and "{" not in code_lines[j] and not code_lines[j].rstrip().endswith(";"):
                j += 1
            if j < len(code_lines) and "{" in code_lines[j]:
                if code_lines[j].rstrip().endswith("}") and j == i and code_lines[j].count("{") == code_lines[j].count("}"):
                    found.append((m.group(1), i, j))
                    i = j + 1
                    continue
                k = j + 1
                while k < len(code_lines) and not code_lines[k].startswith("}"):
                    k += 1
                found.append((m.group(1), i, k))
                i = k + 1
                continue
        i += 1
    return found


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
    if is_operation(path) and rel(path) not in NOT_OPERATIONS:
        for label in ("Syntax:", "Inputs:", "Outputs:", "See also:"):
            if label not in text:
                issues.append(f"the {label[:-1]} block is missing from the docstring")
        syntax = re.search(r"Syntax:(.*?)(?=\n// [A-Z][a-z]+( [a-z]+)?:|\Z)", text, re.S)
        if syntax and not re.search(rf"\b{re.escape(stem)}\(", syntax.group(1)):
            issues.append(f"the Syntax block should show a call of {stem}")
    if needs_cora_tag(path) and not any(CORA_TAG.match(line) for line in block):
        issues.append("the docstring should name the CORA file, '// CORA: <path>.m', or say "
                      "'// CORA.cpp only: <reason>'")


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
        elif line.rstrip() == "} // namespace" and in_anon:
            in_anon = False
            anon_lines.append(i)
        elif in_anon and library and line and not line.startswith((" ", "//", "}", "using", "struct", "class", "enum", "#")):
            m = re.match(r"^(?:[\w:<>,*&]+\s+)+[*&]?(\w+)\(", line)
            if not m or not line.rstrip().endswith(("{", ";", ",", "(")):
                continue
            # the end of the signature, which may run over several lines
            j = i
            while j + 1 < len(lines) and not lines[j].rstrip().endswith(("{", ";")):
                j += 1
            if not m.group(1).startswith("aux_"):
                issues.append(f"line {i + 1}: a function in an anonymous namespace should be called aux_{m.group(1)}")
            elif not lines[j].rstrip().endswith("{"):
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


def check_line_length(path, lines, issues, limit=MAX_LINE_LENGTH):
    for i, line in enumerate(lines):
        if len(line) > limit:
            issues.append(f"line {i + 1}: {len(line)} characters, at most {limit}")


def check_naming(path, code_lines, issues):
    """Types PascalCase; functions, variables and members snake_case (constants kName)."""
    for i, code in enumerate(code_lines):
        for m in re.finditer(r"\b(?:struct|class|enum class|enum)\s+(\w+)\s*[:{;]", code):
            if not re.match(r"^[A-Z][A-Za-z0-9]*$", m.group(1)):
                issues.append(f"line {i + 1}: the type '{m.group(1)}' is not PascalCase")
        # a camelCase name that the code defines or calls itself (not a member of another library)
        for m in re.finditer(r"(?<![\w.>~])(?:([A-Za-z_]\w*)::)?([a-z][a-z0-9]*[A-Z]\w*)\b", code):
            scope, name = m.group(1), m.group(2)
            if name in FOREIGN_NAMES or re.match(r"^k[A-Z]", name) or (scope and scope[0].islower() and scope not in ("cora", "detail")):
                continue
            if m.start() > 0 and code[m.start() - 1] == ":":  # a::b::name, the scope is a library
                continue
            issues.append(f"line {i + 1}: '{name}' is not snake_case")
        for m in re.finditer(r"\benum class (\w+)[^{]*\{([^}]*)\}", code):
            for value in re.findall(r"\b([A-Za-z]\w*)\b", m.group(2)):
                if value != value.lower():
                    issues.append(f"line {i + 1}: the enumerator '{value}' is not snake_case")


def check_file_name(path, issues):
    stem = os.path.splitext(os.path.basename(path))[0]
    if not re.match(r"^[a-z0-9_]+$", stem):
        issues.append(f"the file name '{stem}' is not snake_case")


def check_checks(path, code_lines, issues):
    """The guard of a public operation, CORA_CHECK only in guarded functions, no raw throw."""
    r = rel(path)
    text = "\n".join(code_lines)
    if r != "src/global/error.cpp" and re.search(r"\bthrow\b", text):
        issues.append("a raw throw: use CORA_CHECK for an input and CORA_FAIL for a failure")
    if "/private/" in r and "CORA_CHECK" in text:
        issues.append("CORA_CHECK under private/: the public entry point checks, private code never")
    for name, first, last in functions(code_lines):
        body = code_lines[first:last + 1]
        has_check = any("CORA_CHECK" in l for l in body)
        guarded = any(re.search(r"\bdetail::(Entry|Validation) entry\b", l) for l in body)
        if has_check and name.endswith("_impl"):
            issues.append(f"line {first + 1}: CORA_CHECK in {name}: an _impl is unchecked")
        elif has_check and not guarded:
            issues.append(f"line {first + 1}: CORA_CHECK in {name}, which has no guard "
                          "'detail::Entry entry;'")
    if not r.startswith(OPERATION_ROOTS) or "/private/" in r or r in NOT_OPERATIONS or not r.endswith(".cpp"):
        return
    # below MAIN, every function but an _impl starts with the guard
    main = next((i for i, l in enumerate(read_lines(path)) if l == MAIN_MARKER), None)
    if main is None:
        return
    for name, first, last in functions(code_lines):
        if first < main or name.endswith("_impl") or name.startswith("operator"):
            continue
        body = [l.strip() for l in code_lines[first:last + 1]]
        opened = next(k for k, l in enumerate(body) if "{" in l)
        rest = [l for l in body[opened + 1:] if l]
        if not rest or not re.match(r"^(const )?detail::(Entry|Validation) entry;", rest[0]):
            if body[opened].rstrip().endswith("}"):
                continue  # a one-line forwarding body
            if len(rest) == 2 and re.match(r"^return detail::\w+\(.*\);$", rest[0]):
                continue  # forwards to a guarded entry of the library (detail::dispatch)
            issues.append(f"line {first + 1}: {name} should begin with 'detail::Entry entry;'")


def check_options(path, lines, issues):
    text = "\n".join(lines)
    option = re.search(r'==\s*"[^"]+"|\bswitch\s*\(', code_only(text))
    if option and not path.endswith(".h"):
        if "CORA_FAIL" not in text and "CORA_CHECK" not in text:
            issues.append("it tests an option but never fails: an unknown option should raise an error")
    code = "\n".join(l for l in lines if not l.lstrip().startswith("//"))
    for m in re.finditer(r'CORA_FAIL\(\s*[\w:]+,\s*"((?:[^"\\]|\\.)*)"\s*\)|'
                         r'\[[&=]?\]\s*\{\s*return std::string\("((?:[^"\\]|\\.)*)"\)', code):
        message = m.group(1) or m.group(2)
        if len(message) < 20:
            issues.append(f"the error message '{message}' is not descriptive")


def check_one_code_path(path, code_lines, issues):
    r = rel(path)
    if r.startswith(BACKEND_FOLDERS):
        return
    for i, code in enumerate(code_lines):
        if re.search(r"#\s*if(def|ndef)?\s.*CORACPP_(TORCH|PYTHON_TORCH)\b", code) or \
                re.search(r"\bis_(torch|eigen|lean|host)\s*\(|\.backend_name\s*\(|\bDType::", code):
            issues.append(f"line {i + 1}: a branch on the backend or dtype; the data decides "
                          "(only global/backend and global/tensor ask)")


# A tensor made from raw numbers on the current backend, not like an operand.
FACTORY = re.compile(r"\bTensor::(zeros|ones|eye|from_data)\s*\(|\bTensor\s*\(\s*\{"
                     r"|\bTensor\s+\w+\s*[({]\s*\{|\bbackend\s*\(\s*\)\s*\.\s*make\s*\(")
# The attribute of a raw-number entry point (random sets, constants without a context).
FROM_NUMBERS = "CORA_FROM_NUMBERS"


def check_factories(path, code_lines, issues):
    """Library code makes tensors like an operand; only a function marked CORA_FROM_NUMBERS (and
    the tensor, backend and binding folders) makes them from raw numbers."""
    if rel(path).startswith(BACKEND_FOLDERS):
        return
    marked = set()
    for _, first, last in functions(code_lines):
        if code_lines[first].startswith(FROM_NUMBERS):
            marked.update(range(first, last + 1))
    for i, code in enumerate(code_lines):
        if FACTORY.search(code) and i not in marked:
            issues.append(f"line {i + 1}: a tensor from raw numbers on the current backend; make it "
                          f"like an operand (zeros_like, Tensor::like) or mark the function {FROM_NUMBERS}")


def check_source(path):
    lines = read_lines(path)
    code_lines = code_only("\n".join(lines)).split("\n")
    issues = []
    check_docstring(path, lines, issues)
    check_markers(path, lines, issues)
    check_comment_density(path, lines, issues)
    check_sections(path, lines, issues)
    check_line_length(path, lines, issues)
    check_file_name(path, issues)
    check_naming(path, code_lines, issues)
    check_checks(path, code_lines, issues)
    check_options(path, lines, issues)
    check_one_code_path(path, code_lines, issues)
    check_factories(path, code_lines, issues)
    return issues


# --------------------------------------------------------------- tests and examples

def operation_tests(path):
    """The class, the operation and the test files of an operation file."""
    parts = rel(path).split("/")
    op, cls, area = os.path.splitext(parts[-1])[0], parts[-2], "/".join(parts[1:-2])
    pattern = f"tests/{area}/{cls}/test_{cls}_{op}*.cpp" if area else f"tests/{cls}/test_{cls}_{op}*.cpp"
    return pattern, files(pattern)


def check_tests_exist(issues_by_file):
    for path in files("src/**/*.cpp"):
        if not is_operation(path):
            continue
        pattern, found = operation_tests(path)
        issues = issues_by_file.setdefault(rel(path), [])
        if not found:
            issues.append(f"no test: expected {pattern}")
            continue
        text = "\n".join("\n".join(read_lines(f)) for f in found)
        op = os.path.splitext(os.path.basename(path))[0]
        if op not in NO_INPUT_OPERATIONS and not re.search(r"\braises\(", text):
            issues.append(f"its tests ({pattern}) call it with no invalid input: "
                          "test::raises(f, Error::<identifier>)")
        if "ChecksScope" not in text:
            issues.append(f"its tests ({pattern}) never call it with the checks off: ChecksScope")
        if not issues:
            del issues_by_file[rel(path)]


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
    check_file_name(path, issues)
    if path.endswith(".cpp"):
        check_naming(path, code_only("\n".join(lines)).split("\n"), issues)
    return issues


def check_test_files(issues_by_file):
    for path in files("tests/**/test_*.cpp") + files("tests/**/*.h"):
        lines = read_lines(path)
        found = []
        stem = os.path.splitext(os.path.basename(path))[0]
        if not lines or not lines[0].startswith(f"// {stem} - "):
            found.append(f"a test should start with '// {stem} - <what it tests>'")
        check_line_length(path, lines, found, limit=110)
        check_file_name(path, found)
        check_naming(path, code_only("\n".join(lines)).split("\n"), found)
        if found:
            issues_by_file.setdefault(rel(path), []).extend(found)


def check_python_files(issues_by_file):
    """The Python package, its tests and examples: a docstring naming the file, line lengths,
    snake_case functions."""
    for path in (files("src/global/python/cora/*.py") + files("tests/python/test_*.py")
                 + files("examples/python/**/*.py") + files("scripts/**/*.py")):
        lines = read_lines(path)
        found = []
        stem = os.path.splitext(os.path.basename(path))[0]
        is_test = "/tests/" in path.replace("\\", "/")
        if not lines or not lines[0].startswith('"""'):
            found.append("the module should start with a docstring")
        elif is_test and not lines[0].startswith(f'"""{stem} - '):
            found.append(f"the docstring should start with '{stem} - <what it tests>'")
        if "/src/" in path.replace("\\", "/"):
            check_line_length(path, lines, found)
        elif is_test:
            check_line_length(path, lines, found, limit=110)
        if not stem.startswith("_"):
            check_file_name(path, found)
        if rel(path) not in CORA_NAMED_PYTHON:
            for i, line in enumerate(lines):
                m = re.match(r"^\s*def ([A-Za-z_]\w*)\(", line)
                if m and not re.match(r"^_?_?[a-z0-9_]+$", m.group(1)) and m.group(1) not in ("setUp", "tearDown"):
                    found.append(f"line {i + 1}: the function '{m.group(1)}' is not snake_case")
                m = re.match(r"^\s*class ([A-Za-z_]\w*)", line)
                if m and not re.match(r"^_?[A-Z][A-Za-z0-9]*$", m.group(1)):
                    found.append(f"line {i + 1}: the class '{m.group(1)}' is not PascalCase")
        if found:
            issues_by_file.setdefault(rel(path), []).extend(found)


def check_python_tests_cover_the_api(issues_by_file):
    """Every class the package exports has a Python test file: tests/python/test_<class>*.py."""
    init = "\n".join(read_lines(os.path.join(ROOT, "src/global/python/cora/__init__.py")))
    for name in ("Zonotope", "Interval", "LinearSys", "NonlinearSys", "NeuralNetwork", "Specification"):
        if name not in init:
            continue
        stem = re.sub(r"(?<!^)([A-Z])", r"_\1", name).lower()
        if not files(f"tests/python/test_{stem}*.py"):
            issues_by_file.setdefault("src/global/python/cora/__init__.py", []).append(
                f"no Python test for {name}: expected tests/python/test_{stem}*.py")


# --------------------------------------------------------------- tensor operations

OPS_DEF = "include/cora/global/tensor/ops.def"
OP_ROW = re.compile(r'^CORA_TENSOR_OP\((\w+), (\w+), (\w+), (\w+), (\d+), "(.*)"\)$')
GRADCHECK_TEST = "tests/global/tensor/test_tensor_gradcheck_torch.cpp"
TENSOR_DOC = "docs/tensor.md"
DOC_ROW = re.compile(r"^\| `(\w+)` \| (\w+) \| (\w+) \| (\w+)(?: (\d+))? \|")


def tensor_ops():
    """The rows of ops.def: name, kind, gradient, agreement, k."""
    rows = []
    for line in read_lines(os.path.join(ROOT, OPS_DEF)):
        m = OP_ROW.match(line)
        if m:
            rows.append(m.groups()[:5])
    return rows


def check_tensor_ops(issues_by_file):
    """A gradcheck per operation that passes gradients; docs/tensor.md lists every operation with
    the kind, gradient tag and agreement of ops.def, and no other."""
    rows = tensor_ops()
    test = "\n".join(read_lines(os.path.join(ROOT, GRADCHECK_TEST)))
    for name, _, gradient, _, _ in rows:
        if gradient != "not_differentiable" and f'gradcheck("{name}"' not in test:
            issues_by_file.setdefault(GRADCHECK_TEST, []).append(
                f"no gradcheck(\"{name}\", ...): {name} is {gradient} in ops.def")
    doc = {}
    for line in read_lines(os.path.join(ROOT, TENSOR_DOC)):
        m = DOC_ROW.match(line)
        if m:
            name, kind, gradient, agreement, k = m.groups()
            doc[name] = (kind, gradient, agreement, k or "0")
    for name, kind, gradient, agreement, k in rows:
        if doc.pop(name, None) != (kind, gradient, agreement, k):
            issues_by_file.setdefault(TENSOR_DOC, []).append(
                f"the row of `{name}` should read: | `{name}` | {kind} | {gradient} | "
                f"{agreement}{'' if agreement == 'exact' else ' ' + k} |")
    for name in doc:
        issues_by_file.setdefault(TENSOR_DOC, []).append(f"`{name}` is not an operation of ops.def")


# --------------------------------------------------------------- duplicated code

# Tables whose rows look alike, with why.
CLONE_EXEMPT = {"src/global/error.cpp": "the table of CORA's error identifiers"}
TOKEN = re.compile(r"[A-Za-z_]\w*|\d[\w.]*|::|->|[^\s\w]")


def tokens(path):
    """The tokens of the code between BEGIN CODE and END OF CODE, literals abstracted, with the
    line of each."""
    lines = read_lines(path)
    begin = lines.index(BEGIN_MARKER) if BEGIN_MARKER in lines else 0
    end = lines.index(END_MARKER) if END_MARKER in lines else len(lines)
    code = code_only("\n".join(lines)).split("\n")
    out = []
    for i in range(begin + 1, end):
        if code[i].lstrip().startswith("#"):
            continue
        for t in TOKEN.findall(code[i]):
            out.append(("0" if t[0].isdigit() else t, i + 1))
    return out


def check_clones(issues_by_file):
    """Code repeated in two places (token clones of CLONE_TOKENS tokens): merge it into one
    function. The python bindings and the competition entry are left out."""
    seen = {}
    sources = [p for p in files("src/**/*.cpp") + files("src/**/*.h") + files("include/**/*.h")
               if not rel(p).startswith("src/global/python/") and rel(p) not in CLONE_EXEMPT]
    reported = set()
    for path in sources:
        toks = tokens(path)
        kinds = [t for t, _ in toks]
        i = 0
        while i + CLONE_TOKENS <= len(kinds):
            key = hash(tuple(kinds[i:i + CLONE_TOKENS]))
            where = (rel(path), toks[i][1], i)
            if key in seen:
                other = seen[key]
                if other[0] != where[0] or i - other[2] >= CLONE_TOKENS:
                    pair = (other[0], where[0])
                    if pair not in reported:
                        reported.add(pair)
                        issues_by_file.setdefault(where[0], []).append(
                            f"line {where[1]}: {CLONE_TOKENS} tokens repeat {other[0]} line {other[1]}; "
                            "merge them into one function")
                    i += CLONE_TOKENS
                    continue
            else:
                seen[key] = where
            i += 1


def main():
    issues_by_file = {}
    for path in files("src/**/*.cpp") + files("src/**/*.h") + files("include/**/*.h"):
        if rel(path).startswith("src/global/python/"):
            found = []
            lines = read_lines(path)
            check_markers(path, lines, found, needs_main=False, library=False)
            check_file_name(path, found)
            check_naming(path, code_only("\n".join(lines)).split("\n"), found)
            check_checks(path, code_only("\n".join(lines)).split("\n"), found)
        else:
            found = check_source(path)
        if found:
            issues_by_file[rel(path)] = found
    check_tests_exist(issues_by_file)
    check_test_files(issues_by_file)
    check_python_files(issues_by_file)
    check_python_tests_cover_the_api(issues_by_file)
    check_clones(issues_by_file)
    check_tensor_ops(issues_by_file)
    for path in files("examples/cpp/**/*.cpp") + files("examples/python/example_*.py"):
        if "/benchmarks/" in rel(path):
            found = []
            check_file_name(path, found)
        else:
            found = check_example(path)
        if found:
            issues_by_file.setdefault(rel(path), []).extend(found)

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
