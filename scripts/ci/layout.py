"""layout - the file layout of CORA.cpp: docstring, author block, markers, comment limits

Every C++ and Python file has the layout of the library files: a docstring that starts with the
file's name, the CORA tags, the includes or imports, then the code between marker lines. This module
reads that layout (shared by the conventions test and the layout converter) and checks it.

Syntax:   issues = check_file(rel_path, lines)       the problems of one file, as messages
          body, start = read_doc(lines, "cpp")        the docstring lines without comment signs
Outputs:  the messages, each saying what to do; an empty list for a file in the layout
See also: tests/global/test_coding_conventions.py, tests/global/test_layout.py, CONTRIBUTING.md

Authors:       Claude Sonnet 5.5
Written:       06-November-2026
Last update:   ---
Reviewed:      ---
"""
import ast
import io
import os
import re
import tokenize

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

BEGIN_MARKER = "// ----------------------------------------  BEGIN CODE  ---------------------------------------- //"
END_MARKER = "// ---------------------------------------  END OF CODE  ---------------------------------------- //"
AUX_MARKER = "// ----------------------------------------  AUXILIARY  ----------------------------------------- //"
MAIN_MARKER = "// ===========================================  MAIN  =========================================== //"
MAX_DESCRIPTION = 12  # lines of prose in a docstring at the top of a file
MAX_DOC_LINES = 3  # lines of a docstring ahead of a function
MAX_CODE_RUN = 20  # lines of code without a comment
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December")
DATE = re.compile(r"^(\d\d)-(" + "|".join(MONTHS) + r")-(\d{4})$")
MODEL = re.compile(r"^Claude (Opus|Sonnet|Fable|Haiku) \d+(\.\d+)?$")
PERSON = re.compile(r"^[A-Z][a-z]+( [A-Z][a-z]+)+$")
REVIEW = re.compile(r"^(\d\d-[A-Za-z]+-\d{4}) \((.+)\)$")
LABEL = re.compile(r"^([A-Z][A-Za-z.]*(?: [A-Za-z.]+){0,3}):(?:\s+(.*))?$")
ENTRY_LABELS = ("Syntax", "Inputs", "Outputs", "See also")  # one line per entry
TAG_LABELS = ("CORA", "CORA.cpp only", "PROPERTY", "ORACLE")  # the tags above the author block
AUTHOR_LABELS = ("CORA authors", "Authors", "Written", "Last update", "Reviewed")
FIELD_COLUMN = 15  # the author block's values start here, after the comment sign and a blank
SECTION = re.compile(r"^\s*(//|#) [A-Z][A-Za-z ]+ -{10,}$")
INCLUDE_LIKE = re.compile(r"^\s*(#|using\b|extern\b|namespace\s+\w+\s*=)")


# ============================================  MAIN  ============================================ #

def language(path):
    """'cpp' for .cpp and .h files, 'py' for .py files, else None."""
    return {".cpp": "cpp", ".h": "cpp", ".py": "py"}.get(os.path.splitext(path)[1])


def waived(lines, rule):
    """The 0-based lines that a comment waives from `rule`: `NOLINT(rule): reason` its own line,
    `NOLINTNEXTLINE(rule): reason` the next, `NOLINTBEGIN(rule): reason` to `NOLINTEND(rule)`
    the lines between (the clang-tidy forms; a waiver without a reason waives nothing)."""
    out, begin = set(), None
    for i, line in enumerate(lines):
        for kind, rules, reason in re.findall(r"NOLINT(NEXTLINE|BEGIN|END)?\(([^)]*)\)(:\s*\S)?", line):
            if rule not in [r.strip() for r in rules.split(",")]:
                continue
            if kind == "END" and begin is not None:
                out.update(range(begin, i + 1))
                begin = None
            elif kind == "BEGIN" and reason:
                begin = i
            elif kind == "NEXTLINE" and reason:
                out.add(i + 1)
            elif not kind and reason:
                out.add(i)
    if begin is not None:
        out.update(range(begin, len(lines)))
    return out


def marker(name, lang="cpp"):
    """The marker line of BEGIN, AUXILIARY, MAIN or END, in the comment style of the language."""
    line = {"BEGIN": BEGIN_MARKER, "AUXILIARY": AUX_MARKER, "MAIN": MAIN_MARKER, "END": END_MARKER}[name]
    if lang == "cpp":
        return line
    return ("# =" + line[3:-3] + "= #") if name == "MAIN" else ("# -" + line[3:-3] + "- #")


def field_line(label, value, lang="cpp"):
    """One line of the author block: the label, then the value in column 15."""
    sign = "// " if lang == "cpp" else ""
    return sign + (label + ":").ljust(FIELD_COLUMN if label != "CORA authors" else 14) + value


def parse_date(text):
    """The (day, month number, year) of DD-Month-YYYY, or None when it is not such a date."""
    m = DATE.match(text.strip())
    if not m:
        return None
    day, year = int(m.group(1)), int(m.group(3))
    return (year, MONTHS.index(m.group(2)) + 1, day) if 1 <= day <= 31 else None


def read_doc(lines, lang):
    """The docstring at the top of a file: (body, start), the lines without comment signs and the
    0-based file line of the first one; ([], 0) when there is none."""
    if lang == "cpp":
        body = []
        for line in lines:
            if not line.startswith("//") or line.startswith("///"):
                break
            body.append(line[3:] if line.startswith("// ") else line[2:])
        return body, 0
    if not lines or not lines[0].startswith('"""'):
        return [], 0
    text = lines[0][3:]
    if '"""' in text:
        return [text[:text.index('"""')]], 0
    body = [text]
    for line in lines[1:]:
        if '"""' in line:
            before = line[:line.index('"""')]
            if before.strip():
                body.append(before)
            break
        body.append(line)
    return body, 0


def scan(lines, lang):
    """Per line: (code text with strings and comments blanked, kind of its comment): 'line' for
    // and #, 'doc' for /// and docstrings, None for a line without a comment."""
    if lang == "py":
        return scan_python(lines)
    text = "\n".join(lines)
    out, kinds, i, n, line_no = [], [None] * len(lines), 0, len(text), 0
    while i < n:
        c = text[i]
        if c == "\n":
            line_no += 1
            out.append(c)
            i += 1
        elif text.startswith("//", i) or text.startswith("/*", i):
            line_comment = text[i + 1] == "/"
            end = text.find("\n" if line_comment else "*/", i + 2)
            end = n if end < 0 else (end if line_comment else end + 2)
            segment = text[i:end]
            for k in range(segment.count("\n") + 1):
                kinds[line_no + k] = "doc" if segment.startswith(("///", "/**")) else "line"
            out.append(re.sub(r"[^\n]", " ", segment))
            line_no += segment.count("\n")
            i = end
        elif c == '"' or (c == "'" and not (i > 0 and text[i - 1].isalnum())):
            j = i + 1
            while j < n and text[j] != c and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            out.append(c + " " * max(0, min(j, n) - i - 1) + (c if j < n else ""))
            i = j + 1
        else:
            out.append(c)
            i += 1
    return list(zip("".join(out).split("\n"), kinds))


def scan_python(lines):
    """The same for Python: a # comment is 'line'; a string that is a statement of its own (a
    docstring) is 'doc'."""
    code = list(lines)
    kinds = [None] * len(lines)
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO("\n".join(lines) + "\n").readline))
    except (tokenize.TokenError, IndentationError):
        return [(line, None) for line in lines]
    previous = None
    for tok in tokens:
        first, last = tok.start[0] - 1, tok.end[0] - 1
        if tok.type == tokenize.COMMENT:
            kinds[first] = "line"
            code[first] = code[first][:tok.start[1]]
        elif tok.type == tokenize.STRING and (previous is None or previous.type in (
                tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.NL)):
            for k in range(first, last + 1):
                kinds[k], code[k] = "doc", ""
        if tok.type not in (tokenize.NL, tokenize.COMMENT):
            previous = tok
    return list(zip(code, kinds))


def marker_lines(lines, lang):
    """{name: [line indices]} of the four marker lines."""
    return {name: [i for i, line in enumerate(lines) if line == marker(name, lang)]
            for name in ("BEGIN", "AUXILIARY", "MAIN", "END")}


# ---------------------------------------------------------------------------- the docstring

def split_blocks(body):
    """The blocks of a docstring body after its first line: [(label or None, first, last)], label
    None for prose. A block is a 'Label:' line and the indented lines that continue it."""
    blocks, current = [], None
    for k, line in enumerate(body):
        m = LABEL.match(line) if line and not line.startswith(" ") else None
        if k == 0:
            continue
        if m:
            current = [m.group(1), k, k]
            blocks.append(current)
        elif line.startswith("    ") and current:
            current[2] = k
        elif line.strip() and current and current[0] is None:
            current[2] = k
        elif line.strip():
            current = [None, k, k]
            blocks.append(current)
        else:
            current = None
    return [tuple(b) for b in blocks]


def check_entries(body, blocks, issues, base):
    """Syntax, Inputs, Outputs and See also take one line per entry."""
    entry = re.compile(r"^\S.*? - \S|^-$|^-\s")
    for label, first, last in blocks:
        if label not in ENTRY_LABELS:
            continue
        for k in range(first, last + 1):
            text = body[k][len(label) + 1:].strip() if k == first else body[k].strip()
            if label == "See also":
                bad = k > first
            elif label == "Syntax":
                bad = k > first and not re.search(r"[(=\[]", text) \
                    and text.split()[:1] != body[first][len(label) + 1:].split()[:1]
            else:
                bad = k > first and not entry.match(text)
            if bad:
                issues.append(f"line {base + k + 1}: '{label}' takes one line per entry; a continued "
                              "line belongs in the description or in a comment of the code")
            elif label in ("Inputs", "Outputs") and re.search(r"[;.]\s+[A-Za-z_]\w*(?:, [A-Za-z_]\w*)* - [a-z]", text):
                issues.append(f"line {base + k + 1}: two entries on one line; one line per "
                              f"{label.lower()[:-1]}put")


def check_author_block(body, blocks, base, issues, lang, has_cora_tag):
    """The last lines of the docstring: [CORA authors,] Authors, Written, Last update, Reviewed."""
    end = len(body)
    while end and not body[end - 1].strip():
        end -= 1
    labels = [(label, first, last) for label, first, last in blocks if label in AUTHOR_LABELS]
    names = [label for label, _, _ in labels]
    wanted = ["CORA authors"] if has_cora_tag else []
    if names != wanted + ["Authors", "Written", "Last update", "Reviewed"]:
        issues.append("the docstring should end with the author block"
                      + (", 'CORA authors:' above it (the Authors: of the CORA file)" if has_cora_tag else "")
                      + ": " + ", ".join(f"'{label}:'" for label in wanted + list(AUTHOR_LABELS[1:])))
        return
    if labels[-1][2] != end - 1 or any(labels[k][2] + 1 != labels[k + 1][1] for k in range(len(labels) - 1)):
        issues.append("the author block should be the last lines of the docstring, one field per line")
        return
    values = {}
    for label, first, last in labels:
        expected = field_line(label, "", lang)
        sign = len("// " if lang == "cpp" else "")
        line = body[first]
        if not line.startswith(expected[sign:]):
            issues.append(f"line {base + first + 1}: the author block reads '{expected.rstrip()}' and then "
                          f"the value, aligned to column {FIELD_COLUMN}")
            return
        values[label] = [line[len(expected) - sign:].strip()] + [body[k].strip() for k in range(first + 1, last + 1)]
    check_author_values(values, base, issues)


def check_author_values(values, base, issues):
    """Authors are models (or people), Written and Last update are dates, Reviewed is by humans."""
    for who in [w.strip() for w in values["Authors"][0].split(",")]:
        if not (MODEL.match(who) or PERSON.match(who)):
            issues.append(f"'Authors:' lists the models that wrote the file, comma-separated, each "
                          f"'Claude <Opus|Sonnet|Fable|Haiku> <version>', not '{who}'")
    written, updated = values["Written"][0], values["Last update"][0]
    if not parse_date(written):
        issues.append(f"'Written:' is a DD-Month-YYYY date, not '{written}'")
    if updated != "---" and not parse_date(updated):
        issues.append(f"'Last update:' is a DD-Month-YYYY date or '---', not '{updated}'")
    elif updated != "---" and parse_date(written) and parse_date(updated) < parse_date(written):
        issues.append("'Last update:' is earlier than 'Written:'")
    reviews = values["Reviewed"]
    if reviews != ["---"]:
        for text in reviews:
            m = REVIEW.match(text)
            if not m or not parse_date(m.group(1)):
                issues.append(f"'Reviewed:' is '---' or lines 'DD-Month-YYYY (Name)', not '{text}'")
            elif m.group(2).startswith("Claude"):
                issues.append(f"'Reviewed:' records human reviews; '{m.group(2)}' is a model")


def check_docstring(path, lines, lang, issues):
    """The name line, the prose limit, one line per entry and the author block."""
    body, start = read_doc(lines, lang)
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem == "__init__":  # a package is named by its folder
        stem = os.path.basename(os.path.dirname(path))
    if not body:
        issues.append("no docstring at the top" + (' (""")' if lang == "py" else ""))
        return
    if not re.match(rf"^{re.escape(stem)} - \S", body[0]):
        issues.append(f"the docstring should start with '{stem} - <what the file is>'")
    if len(body) > 1 and body[1].strip():
        issues.append("an empty line should follow the first docstring line")
    blocks = split_blocks(body)
    special = ENTRY_LABELS + TAG_LABELS + AUTHOR_LABELS
    prose = sum(last - first + 1 for label, first, last in blocks if label not in special)
    if prose > MAX_DESCRIPTION:
        issues.append(f"the description is {prose} lines of prose, at most {MAX_DESCRIPTION}: say less, "
                      "or put the detail into a comment of the code")
    check_entries(body, blocks, issues, start)
    check_author_block(body, blocks, start, issues, lang, has_cora_path(body, lines, lang))
    for label, first, last in blocks:
        if label in TAG_LABELS and last - first >= 3:
            issues.append(f"line {start + first + 1}: the {label} tag takes at most three lines")


def has_python_tag(lines, lang):
    """Whether a Python file names a CORA file in a '# CORA: <path>.m' line."""
    return lang == "py" and any(re.match(r"^# CORA: \S+\.m\b", line) for line in lines[:80])


def has_cora_path(body, lines, lang):
    """Whether the file names a CORA file ('CORA: <path>.m'), which gives it 'CORA authors:'."""
    if lang == "py":
        return has_python_tag(lines, lang)
    return any(re.match(r"^CORA: \S+\.m\b", line) for line in body)


# ------------------------------------------------------------------------------ the markers

def code_above_begin(lines, lang, begin):
    """The first line above BEGIN CODE that is neither docstring, comment, import nor include."""
    if lang == "cpp":
        for i, line in enumerate(lines[:begin]):
            if line.strip() and not line.startswith("//") and not INCLUDE_LIKE.match(line) \
                    and not line.startswith("  "):
                return i
        return None
    try:
        tree = ast.parse("\n".join(lines))
    except SyntaxError:
        return None
    for node in tree.body[1:] if tree.body and is_docstring(tree.body[0]) else tree.body:
        if node.lineno > begin:
            break
        if not is_import_statement(node):
            return node.lineno - 1
    return None


def is_docstring(node):
    return isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant) \
        and isinstance(node.value.value, str)


def is_import_statement(node):
    """An import, or a try/if whose branches only import, assign constants or raise."""
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return True
    if isinstance(node, (ast.Try, ast.If)):
        parts = list(node.body) + list(getattr(node, "orelse", [])) + [
            s for h in getattr(node, "handlers", []) for s in h.body]
        return all(is_import_statement(s) or isinstance(s, (ast.Pass, ast.Raise)) for s in parts)
    return False


def python_needs_main(lines):
    """Whether a Python file has anything to run or to export: a def, a class, an executable
    statement. A file of imports and constants has no MAIN."""
    try:
        tree = ast.parse("\n".join(lines))
    except SyntaxError:
        return True
    for node in tree.body:
        if is_docstring(node) or is_import_statement(node) or isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        return True
    return False


def has_executable(nodes):
    """Whether top-level statements run something: a call, a loop, a with, a branch that is not
    an import guard or `if __name__ == "__main__"`."""
    for node in nodes:
        if isinstance(node, (ast.For, ast.While, ast.With, ast.AugAssign)):
            return True
        if isinstance(node, ast.Expr) and not is_docstring(node):
            return True
        if isinstance(node, ast.If) and not is_import_statement(node) \
                and "__name__" not in ast.unparse(node.test):
            return True
    return False


def first_main_statement(lines, begin, end):
    """The 0-based line of the first top-level statement between BEGIN and END that belongs to
    MAIN in a Python file: the first test class, else `def main`, else (a script that runs
    statements) the first statement that is no import or definition, else the first public
    function or class."""
    try:
        tree = ast.parse("\n".join(lines))
    except SyntaxError:
        return None
    nodes = [n for n in tree.body if begin < n.lineno - 1 < end]
    definitions = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    tests = [n for n in nodes if isinstance(n, ast.ClassDef) and any(
        ast.unparse(b).endswith("TestCase") for b in n.bases)]
    mains = [n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == "main"]
    public = [n for n in nodes if isinstance(n, definitions) and not n.name.startswith("_")]
    script = [n for n in nodes if not isinstance(n, definitions) and not is_import_statement(n)
              and not is_docstring(n)]
    for group in (tests, mains, script if has_executable(nodes) else [], public):
        if group:
            first = group[0]
            return min([first.lineno] + [d.lineno for d in getattr(first, "decorator_list", [])]) - 1
    return None


def check_markers(path, lines, lang, issues):
    """BEGIN CODE ... END OF CODE around the code; AUXILIARY above MAIN iff there is auxiliary
    code; MAIN (not in a header) above the operation, the test cases or `main`."""
    at = marker_lines(lines, lang)
    is_header = path.endswith(".h")
    needs_main = (not is_header) if lang == "cpp" else python_needs_main(lines)
    required = ["BEGIN", "END"] + (["MAIN"] if needs_main else [])
    for name in required:
        if len(at[name]) != 1:
            issues.append(f"exactly one {name} marker line is expected: {marker(name, lang)}")
    for name in ("AUXILIARY", "MAIN"):
        if len(at[name]) > 1 or (name == "MAIN" and not needs_main and at[name]):
            issues.append(f"unexpected {name} marker line(s)")
    if any(len(at[name]) != 1 for name in required):
        return
    begin, end = at["BEGIN"][0], at["END"][0]
    aux = at["AUXILIARY"][0] if at["AUXILIARY"] else None
    main = at["MAIN"][0] if at["MAIN"] else None
    order = [x for x in (begin, aux, main, end) if x is not None]
    if order != sorted(order):
        issues.append("the markers should be in the order BEGIN CODE, AUXILIARY, MAIN, END OF CODE")
    if any(line.strip() for line in lines[end + 1:]):
        issues.append("nothing but empty lines should follow END OF CODE")
    bad = code_above_begin(lines, lang, begin)
    if bad is not None:
        issues.append(f"line {bad + 1}: code above BEGIN CODE; only the docstring, imports/includes "
                      "and using-declarations belong there")
    if main is not None and not (main > 2 and lines[main - 1] == "" and lines[main - 2] == ""
                                 and lines[main - 3] != ""):
        issues.append("the MAIN marker line should have exactly two empty lines above it")
    for i in [begin, end] + [x for x in (aux, main) if x is not None]:
        if (i > 0 and lines[i - 1] != "") or (i + 1 < len(lines) and lines[i + 1] != ""):
            issues.append("a marker line should have an empty line above and below")
            break
    check_auxiliary(lines, lang, begin, aux, main, end, issues)
    if main is not None and not is_header:
        check_main_position(lines, lang, main, begin, end, issues)


def check_auxiliary(lines, lang, begin, aux, main, end, issues):
    """AUXILIARY stands right under BEGIN CODE and exists iff code lies between it and MAIN."""
    stop = main if main is not None else end
    code = scan(lines, lang)
    inner = [i for i in range(begin + 1, stop) if code[i][0].strip() and lines[i] not in
             (marker("AUXILIARY", lang),) and not re.match(r"^namespace [\w:]+ \{$", lines[i])]
    if main is None:
        if aux is not None:
            issues.append("an AUXILIARY marker needs a MAIN marker below it; remove it")
        return
    if inner and aux is None:
        issues.append(f"line {inner[0] + 1}: code above MAIN is auxiliary and needs the AUXILIARY marker "
                      f"line right under BEGIN CODE: {marker('AUXILIARY', lang)}")
    elif aux is not None and not inner:
        issues.append("an AUXILIARY marker without auxiliary code; remove it")
    elif aux is not None and inner and inner[0] < aux:
        issues.append(f"line {inner[0] + 1}: the AUXILIARY marker comes before the auxiliary code")


def check_main_position(lines, lang, main, begin, end, issues):
    """MAIN stands right above what the file offers: `int main`, the test classes, `def main`."""
    code = scan(lines, lang)
    skipped = "#" if lang == "py" else "//"  # comments right under MAIN belong to what follows
    after = next((i for i in range(main + 1, end) if lines[i].strip() and not lines[i].startswith(skipped)), None)
    if after is None:
        issues.append("nothing follows the MAIN marker")
        return
    if lang == "py":
        expected = first_main_statement(lines, begin, end)
        if expected is not None and expected != after:
            issues.append(f"line {main + 1}: MAIN belongs right above line {expected + 1}, the first "
                          "test class, `main`, public function or class, or executable statement")
        return
    below = [i for i in range(main + 1, end) if code[i][0].strip()]
    main_def = next((i for i in below if re.match(r"\s*int\s+main\s*\(", code[i][0])), None)
    if main_def is not None and below[0] != main_def:
        issues.append(f"line {below[0] + 1}: the code between MAIN and `int main` is auxiliary; it "
                      "belongs above MAIN, under AUXILIARY")
    elif main_def is None and any(code[i][0].startswith("namespace {") for i in below):
        issues.append("an anonymous namespace is auxiliary code: it belongs between AUXILIARY and MAIN")


# ------------------------------------------------------------------- comments in the code

def check_comments(lines, lang, issues, begin, end):
    """At least one comment per MAX_CODE_RUN lines of code, and a block comment is one line."""
    code = scan(lines, lang)
    run, start, block = 0, 0, []
    for i in range(begin + 1, end):
        text, kind = code[i]
        if kind == "line" and not text.strip() and not SECTION.match(lines[i]):
            block.append(i)
        else:
            if len(block) > 1:
                issues.append(f"line {block[0] + 1}: a block comment of {len(block)} lines; say it in one "
                              "line, or move it into the docstring of the file")
            block = []
        if kind:
            run = 0
        elif text.strip():
            start = i if run == 0 else start
            run += 1
            if run == MAX_CODE_RUN + 1:
                issues.append(f"line {start + 1}: more than {MAX_CODE_RUN} lines of code without a "
                              "comment; add a line saying what the next step does")


def check_function_docs(lines, lang, issues):
    """A doc comment ahead of a function is at most MAX_DOC_LINES lines."""
    if lang == "cpp":
        run = 0
        for i, line in enumerate(lines + [""]):
            if line.strip().startswith("///"):
                run += 1
            else:
                if run > MAX_DOC_LINES:
                    issues.append(f"line {i - run + 1}: a doc comment of {run} lines, at most {MAX_DOC_LINES}: "
                                  "move the explanation into the docstring of the file or a comment")
                run = 0
        return
    try:
        tree = ast.parse("\n".join(lines))
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body \
                and is_docstring(node.body[0]):
            doc = node.body[0]
            size = doc.end_lineno - doc.lineno + 1
            if size > MAX_DOC_LINES:
                issues.append(f"line {doc.lineno}: a docstring of {size} lines ahead of {node.name}, at most "
                              f"{MAX_DOC_LINES}: move the explanation into the docstring of the file or a comment")


def check_file(path, lines):
    """All layout checks of one C++ or Python file (path relative to the repository)."""
    lang = language(path)
    issues = []
    check_docstring(path, lines, lang, issues)
    check_markers(path, lines, lang, issues)
    at = marker_lines(lines, lang)
    if len(at["BEGIN"]) == 1 and len(at["END"]) == 1:
        check_comments(lines, lang, issues, at["BEGIN"][0], at["END"][0])
    check_function_docs(lines, lang, issues)
    return issues

# ----------------------------------------  END OF CODE  ----------------------------------------- #
