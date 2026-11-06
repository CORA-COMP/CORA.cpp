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
          lint - a NOLINT (clang-tidy suppression) names its checks and its reason;
          tensor operations - every operation of ops.def that passes gradients has a gradcheck
          test, and docs/tensor.md lists every operation with the tags of ops.def.
Exit code: 0 if all conventions hold, else 1, with the files and what is wrong.
"""
import ast
import glob
import os
import re
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "ci"))
import layout  # noqa: E402  (the layout rules: docstring, author block, markers, comments)
from layout import BEGIN_MARKER, END_MARKER, MAIN_MARKER, SECTION  # noqa: E402

MAX_LINE_LENGTH = 100
CLONE_TOKENS = 100  # this many tokens repeated (literals abstracted) is duplicated code
# Directories whose .cpp files are operations, each in a folder named after its class.
OPERATION_ROOTS = ("src/cont_set/", "src/matrix_set/", "src/cont_dynamics/", "src/specification/",
                   "src/nn/")
# Folders of the public headers of the set classes, systems and specifications (CORA tag).
CLASS_HEADER_ROOTS = ("include/cora/cont_set/", "include/cora/matrix_set/",
                      "include/cora/cont_dynamics/", "include/cora/specification/",
                      "include/cora/nn/")
CORA_TAG = re.compile(r"^// (CORA: \S+\.m\b|CORA\.cpp only: \S)")
# Where code may ask which backend, device or dtype a tensor has: the folders named backend and
# tensor. Any other place says why on the line: // NOLINT(backend): <reason>.
BACKEND_FOLDER = re.compile(r"(^|/)(backend|tensor)/")

# The namespaces of the libraries the code calls: a camelCase name behind one is theirs, not ours.
FOREIGN_NAMESPACES = ("Eigen", "std", "py", "pybind11", "torch", "at", "c10", "omp")


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
    parts = r.split("/")
    return r.endswith(".cpp") and r.startswith(OPERATION_ROOTS) and "/private/" not in r \
        and len(parts) == 4 and os.path.splitext(parts[-1])[0] != parts[-2]


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
                one_line = code_lines[j].count("{") == code_lines[j].count("}")
                if code_lines[j].rstrip().endswith("}") and j == i and one_line:
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

def check_operation_docstring(path, lines, issues):
    """An operation has Syntax, Inputs, Outputs and See also blocks, and names its CORA file
    (a class header too); the layout of the docstring is checked for every file by layout.py."""
    stem = os.path.splitext(os.path.basename(path))[0]
    block = header_comment(lines)
    text = "\n".join(block)
    if is_operation(path):
        for label in ("Syntax:", "Inputs:", "Outputs:", "See also:"):
            if label not in text:
                issues.append(f"the {label[:-1]} block is missing from the docstring")
        syntax = re.search(r"Syntax:(.*?)(?=\n// [A-Z][a-z]+( [a-z]+)?:|\Z)", text, re.S)
        if syntax and not re.search(rf"\b{re.escape(stem)}\(", syntax.group(1)):
            issues.append(f"the Syntax block should show a call of {stem}")
    if needs_cora_tag(path) and not any(CORA_TAG.match(line) for line in block):
        issues.append("the docstring should name the CORA file, '// CORA: <path>.m', or say "
                      "'// CORA.cpp only: <reason>'")


def check_auxiliary_names(lines, issues):
    """The anonymous namespace of a library file is its auxiliary code: functions named aux_...,
    defined above MAIN (the markers themselves are checked by layout.py)."""
    at = layout.marker_lines(lines, "cpp")
    aux = at["AUXILIARY"][0] if at["AUXILIARY"] else None
    main = at["MAIN"][0] if at["MAIN"] else None
    in_anon = False
    for i, line in enumerate(lines):
        if line == "namespace {":
            in_anon = True
        elif line.rstrip() == "} // namespace" and in_anon:
            in_anon = False
        elif in_anon and line and not line.startswith((" ", "//", "}", "using", "struct", "class", "enum", "#")):
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
    if aux is not None and main is not None and not any(line == "namespace {" for line in lines):
        issues.append("an AUXILIARY marker without an anonymous namespace: auxiliary code of a library "
                      "file is in one")


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
            foreign_scope = scope in FOREIGN_NAMESPACES or (scope and scope[0].islower()
                                                            and scope not in ("cora", "detail"))
            if code[:m.start()].rstrip().endswith("template"):  # obj.template member<...>(): a member
                continue
            if re.match(r"^k[A-Z]", name) or foreign_scope:
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
    if not is_operation_file(r):
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
            if len(rest) == 2 and re.match(r"^return (detail::\w+|aux_\w+)\(.*\);$", rest[0]):
                continue  # forwards to a guarded entry (detail::dispatch) or to the aux_ that holds it
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
    if BACKEND_FOLDER.search(r):
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
    if BACKEND_FOLDER.search(rel(path)):
        return
    marked = set()
    for _, first, last in functions(code_lines):
        if code_lines[first].startswith(FROM_NUMBERS):
            marked.update(range(first, last + 1))
    for i, code in enumerate(code_lines):
        if FACTORY.search(code) and i not in marked:
            issues.append(f"line {i + 1}: a tensor from raw numbers on the current backend; make it "
                          f"like an operand (zeros_like, Tensor::like) or mark the function {FROM_NUMBERS}")


# A floating-point literal with a negative exponent (1e-8, 2.2e-16), or the machine epsilon.
SMALL_LITERAL = re.compile(r"(?<![\w.])\d+(?:\.\d*)?[eE]-(?:[4-9]|\d\d+)\b|\b(?:DBL|FLT)_EPSILON\b"
                           r"|numeric_limits\s*<[^>]*>\s*::\s*epsilon\s*\(")
# A tolerance or epsilon named thing set from an expression with a nonzero floating-point literal.
NAMED_LITERAL = re.compile(r"\b\w*(?:[Tt]ol|[Ee]ps)\w*\s*(?:=|\{)\s*([^;,]*)")
FLOAT_LITERAL = re.compile(r"(?<![\w.])(?:\d+\.\d*|\.\d+|\d+[eE][-+]?\d+)(?:[fF])?(?![\w.])")
# The Python default of a tol argument: py::arg("tol") = ..., "tol"_a = ..., def f(tol=...).
TOL_DEFAULT = re.compile(r"\"\w*tol\w*\"\s*(?:\)\s*|_a\s*)=\s*(py::none\(\)|[^,;)]+)"
                         r"|\bdef\b[^#]*?\b\w*tol\w*\s*=\s*([^,)]+)")


def is_zero_literal(text):
    return re.fullmatch(r"\s*-?0(?:\.0*)?\s*", text) is not None


def check_tolerance_literals(path, lines, issues):
    """No tolerance is a number of its own: the repository tolerance and the thresholds of the
    numerical methods live in tolerance.h (default_tol, machine_eps, numerics::k...), and a `tol`
    argument defaults to unset. Flagged outside it: a literal with a negative exponent, the machine
    epsilon, a tol or eps named thing set from a nonzero floating-point literal, and a tol argument
    in the bindings or the Python package that defaults to something but None (0 is exact)."""
    code = code_only("\n".join(lines)).split("\n") if path.endswith((".h", ".cpp")) else []
    for i, line in enumerate(code):
        if SMALL_LITERAL.search(line):
            issues.append(f"line {i + 1}: a tolerance literal or the machine epsilon; use "
                          "default_tol, machine_eps or a numerics:: threshold (tolerance.h)")
            continue
        for m in NAMED_LITERAL.finditer(line):
            if any(not is_zero_literal(f) for f in FLOAT_LITERAL.findall(m.group(1))):
                issues.append(f"line {i + 1}: a tolerance set from a number; take it from "
                              "tolerance.h (default_tol, a numerics:: threshold) or leave it unset")
                break
    for i, line in enumerate(lines):
        if line.lstrip().startswith(("//", "#")):
            continue
        for m in TOL_DEFAULT.finditer(line):
            value = (m.group(1) or m.group(2) or "").strip()
            if value not in ("py::none()", "None", "std::nullopt") and not is_zero_literal(value):
                issues.append(f"line {i + 1}: the tol argument defaults to {value}; it defaults "
                              "to unset (None), the repository tolerance")


def waiving(rule, check, path, lines, data, issues):
    """Runs check(path, data, found) and keeps the findings on lines that no
    `NOLINT(rule): reason` comment waives."""
    found = []
    check(path, data, found)
    skip = layout.waived(lines, rule)
    for issue in found:
        m = re.match(r"line (\d+):", issue)
        if not (m and int(m.group(1)) - 1 in skip):
            issues.append(issue)


def check_source(path):
    lines = read_lines(path)
    code_lines = code_only("\n".join(lines)).split("\n")
    issues = []
    issues.extend(layout.check_file(rel(path), lines))
    check_operation_docstring(path, lines, issues)
    check_auxiliary_names(lines, issues)
    check_sections(path, lines, issues)
    check_line_length(path, lines, issues)
    check_file_name(path, issues)
    waiving("naming", check_naming, path, lines, code_lines, issues)
    check_checks(path, code_lines, issues)
    check_options(path, lines, issues)
    waiving("backend", check_one_code_path, path, lines, code_lines, issues)
    waiving("backend", check_factories, path, lines, code_lines, issues)
    waiving("tolerance", check_tolerance_literals, path, lines, lines, issues)
    return issues


# --------------------------------------------------------------- tests and examples

def operation_tests(path):
    """The class, the operation and the test files of an operation file."""
    parts = rel(path).split("/")
    op, cls, area = os.path.splitext(parts[-1])[0], parts[-2], "/".join(parts[1:-2])
    pattern = f"tests/{area}/{cls}/test_{cls}_{op}*.cpp" if area else f"tests/{cls}/test_{cls}_{op}*.cpp"
    return pattern, files(pattern)


def unnamed(parameter):
    """Whether a parameter of a definition has no name (`const std::string &`, `Tol`): the
    function ignores it."""
    return bool(re.search(r"[&*>]\s*$", parameter)) or len(parameter.split()) <= 1


def takes_no_input(path):
    """Whether every definition of the operation in its file has no parameter, or only unnamed
    ones (`Zonotope Zonotope::abs() const`): it ignores its input, so none can be wrong."""
    op = os.path.splitext(os.path.basename(path))[0]
    code = code_only("\n".join(read_lines(path)))
    signatures = re.findall(rf"::{re.escape(op)}(?:_impl)?\s*\(([^)]*)\)\s*(?:const)?\s*(?:noexcept)?"
                            r"\s*(?:override)?\s*\{", code)
    return bool(signatures) and all(
        all(unnamed(p.strip()) for p in sig.split(",") if p.strip()) for sig in signatures)


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
        if not takes_no_input(path) and not re.search(r"\braises\(", text):
            issues.append(f"its tests ({pattern}) call it with no invalid input: "
                          "test::raises(f, Error::<identifier>)")
        if "ChecksScope" not in text:
            issues.append(f"its tests ({pattern}) never call it with the checks off: ChecksScope")
        if not issues:
            del issues_by_file[rel(path)]


def check_example(path):
    """An example named example_* is split into sections ('// Parameters ----...'); the markers
    and the docstring are checked for every file by layout.py."""
    lines = read_lines(path)
    issues = []
    comment = "//" if path.endswith(".cpp") else "#"
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
        check_line_length(path, lines, found, limit=110)
        check_file_name(path, found)
        check_naming(path, code_only("\n".join(lines)).split("\n"), found)
        if found:
            issues_by_file.setdefault(rel(path), []).extend(found)


def python_naming(path, lines):
    """Functions snake_case, classes PascalCase; a method of a class that derives from a library
    class (`unittest.TestCase`) keeps the library's names (setUp). `NOLINTBEGIN(naming): reason`
    waives a stretch (a bridge that keeps CORA's names)."""
    try:
        tree = ast.parse("\n".join(lines))
    except SyntaxError as error:
        return [f"line {error.lineno}: syntax error"]
    skip, found = layout.waived(lines, "naming"), []
    foreign = {id(item) for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
               and any(isinstance(b, ast.Attribute) for b in node.bases) for item in node.body}
    for node in ast.walk(tree):
        if node.__class__ not in (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef) \
                or node.lineno - 1 in skip or id(node) in foreign:
            continue
        pattern = r"^_?[A-Z][A-Za-z0-9]*$" if isinstance(node, ast.ClassDef) else r"^_?_?[a-z0-9_]+$"
        if not re.match(pattern, node.name):
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            found.append(f"line {node.lineno}: the {kind} '{node.name}' is not "
                         f"{'PascalCase' if kind == 'class' else 'snake_case'}")
    return found


def check_python_files(issues_by_file):
    """The Python package, its tests and examples: a docstring naming the file, line lengths,
    snake_case functions."""
    for path in (files("src/global/python/cora/*.py") + files("tests/python/test_*.py")
                 + files("examples/python/**/*.py") + files("scripts/**/*.py")):
        lines = read_lines(path)
        found = []
        stem = os.path.splitext(os.path.basename(path))[0]
        is_test = "/tests/" in path.replace("\\", "/")
        if "/src/" in path.replace("\\", "/"):
            check_line_length(path, lines, found)
            check_tolerance_literals(path, lines, found)
        elif is_test:
            check_line_length(path, lines, found, limit=110)
        if not stem.startswith("_"):
            check_file_name(path, found)
        found.extend(python_naming(path, lines))
        if found:
            issues_by_file.setdefault(rel(path), []).extend(found)


def check_python_tests_cover_the_api(issues_by_file):
    """Every class with a header of its own (include/cora/<area>/<class>/<class>.h) has a Python
    test file: tests/python/test_<class>*.py."""
    for header in [h for root in CLASS_HEADER_ROOTS for h in files(root + "*/*.h")]:
        folder = os.path.basename(os.path.dirname(header))
        if os.path.splitext(os.path.basename(header))[0] == folder and not files(f"tests/python/test_{folder}*.py"):
            issues_by_file.setdefault(rel(header), []).append(
                f"no Python test for {folder}: expected tests/python/test_{folder}*.py")


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

TOKEN = re.compile(r"[A-Za-z_]\w*|\d[\w.]*|::|->|[^\s\w]")


DECLARATION_START = re.compile(r"^\s+(?:static\s+|virtual\s+|explicit\s+)*[\w:<>,*&\s]+?[\w~]+\(")


def declaration_lines(code, begin, end):
    """The lines of a header's member function declarations that have no body: from a line that
    opens `type name(` to the line that ends the statement with `;`, with no `{` and no `return`
    on the way."""
    out, i = set(), begin + 1
    while i < end:
        if DECLARATION_START.match(code[i]):
            j = i
            while j < end and not code[j].rstrip().endswith((";", "{", "}")):
                j += 1
            block = code[i:j + 1]
            if j < end and code[j].rstrip().endswith(";") and not any("{" in l or "return" in l for l in block):
                out.update(range(i, j + 1))
                i = j + 1
                continue
        i += 1
    return out


def tokens(path):
    """The tokens of the code between BEGIN CODE and END OF CODE, literals abstracted, with the
    line of each. Preprocessor lines and, in a header, declarations without a body (a member
    function declared over one or more lines up to `;`) are left out: every class of an
    interface repeats the same declarations, which is not repeated logic."""
    lines = read_lines(path)
    begin = lines.index(BEGIN_MARKER) if BEGIN_MARKER in lines else 0
    end = lines.index(END_MARKER) if END_MARKER in lines else len(lines)
    code = code_only("\n".join(lines)).split("\n")
    skip = (declaration_lines(code, begin, end) if path.endswith(".h") else set()) | layout.waived(
        lines, "clones")
    out = []
    for i in range(begin + 1, end):
        if code[i].lstrip().startswith("#") or i in skip:
            continue
        for t in TOKEN.findall(code[i]):
            out.append(("0" if t[0].isdigit() else t, i + 1))
    return out


def check_clones(issues_by_file):
    """Code repeated in two places (token clones of CLONE_TOKENS tokens): merge it into one
    function. Lines under `NOLINTBEGIN(clones): <reason>` are left out (rows that repeat by design)."""
    seen = {}
    sources = [p for p in files("src/**/*.cpp") + files("src/**/*.h") + files("include/**/*.h")
               ]
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


def check_build_jobs(issues_by_file):
    """A build names its number of jobs: a bare `--parallel` (or `-j`) means unlimited with make,
    which runs a CI runner out of memory."""
    bare = re.compile(r"(--parallel|\s-j)(?=\s*(\\|$|&&|\||;|--\s))")
    for path in files("scripts/**/*.sh") + files(".github/workflows/*.yml") + files("cmake/**/*.cmake"):
        for i, line in enumerate(read_lines(path)):
            if not line.lstrip().startswith("#") and "cmake --build" in line and bare.search(line):
                issues_by_file.setdefault(rel(path), []).append(
                    f"line {i + 1}: a build without a number of jobs; pass --parallel <n>")


def check_nolint(issues_by_file):
    """A suppression names the checks and the reason: `// NOLINT(check): reason` (also NOLINTNEXTLINE,
    NOLINTBEGIN, in Python after the comment sign); a bare NOLINT silences every check without a word."""
    marker = re.compile(r"\bNOLINT(?:NEXTLINE|BEGIN)?\b(?!\([^)]+\):\s*\S)")
    paths = [p for pattern in ("src/**/*.cpp", "src/**/*.h", "include/**/*.h", "tests/**/*.cpp",
                               "tests/**/*.h", "examples/**/*.cpp", "benchmarks/**/*.cpp")
             for p in files(pattern)]
    paths += [p for folder in ("src", "tests", "scripts", "examples") for p in files(f"{folder}/**/*.py")]
    for path in paths:
        for i, line in enumerate(read_lines(path)):
            text = re.search(r"#.*", line) if path.endswith(".py") else re.search(r".*", line)
            if text and marker.search(text.group(0)):
                issues_by_file.setdefault(rel(path), []).append(
                    f"line {i + 1}: NOLINT without checks and a reason: // NOLINT(check): why")


def check_portable_types(issues_by_file):
    """C++ that builds with MSVC: `ssize_t` is POSIX (MSVC has none); use py::ssize_t in the
    bindings and std::ptrdiff_t elsewhere."""
    bare = re.compile(r"(?<![:\w])ssize_t\b")
    for path in files("src/**/*.cpp") + files("src/**/*.h") + files("include/**/*.h") + files("tests/**/*.cpp"):
        for i, line in enumerate(code_only("\n".join(read_lines(path))).split("\n")):
            if bare.search(line):
                issues_by_file.setdefault(rel(path), []).append(
                    f"line {i + 1}: ssize_t is POSIX; use py::ssize_t or std::ptrdiff_t")


def check_layout_of_the_rest(issues_by_file):
    """The layout of every C++ and Python file outside src/ and include/ (checked with their
    other rules): tests, examples, benchmarks, scripts."""
    for folder in ("tests", "examples", "benchmarks", "scripts"):
        for pattern in ("**/*.cpp", "**/*.h", "**/*.py"):
            for path in files(f"{folder}/{pattern}"):
                found = layout.check_file(rel(path), read_lines(path))
                if found:
                    issues_by_file.setdefault(rel(path), []).extend(found)


def main():
    issues_by_file = {}
    for path in files("src/**/*.cpp") + files("src/**/*.h") + files("include/**/*.h"):
        if rel(path).startswith("src/global/python/"):
            found = []
            lines = read_lines(path)
            found.extend(layout.check_file(rel(path), lines))
            check_file_name(path, found)
            code_lines = code_only("\n".join(lines)).split("\n")
            waiving("naming", check_naming, path, lines, code_lines, found)
            check_checks(path, code_lines, found)
            waiving("backend", check_one_code_path, path, lines, code_lines, found)
            waiving("backend", check_factories, path, lines, code_lines, found)
            waiving("tolerance", check_tolerance_literals, path, lines, lines, found)
        else:
            found = check_source(path)
        if found:
            issues_by_file[rel(path)] = found
    check_tests_exist(issues_by_file)
    check_test_files(issues_by_file)
    check_python_files(issues_by_file)
    check_python_tests_cover_the_api(issues_by_file)
    check_clones(issues_by_file)
    check_build_jobs(issues_by_file)
    check_portable_types(issues_by_file)
    check_nolint(issues_by_file)
    check_tensor_ops(issues_by_file)
    for path in files("examples/**/example_*.cpp") + files("examples/**/example_*.py"):
        found = check_example(path)
        if found:
            issues_by_file.setdefault(rel(path), []).extend(found)
    check_layout_of_the_rest(issues_by_file)

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
