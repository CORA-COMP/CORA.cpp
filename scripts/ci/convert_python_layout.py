"""convert_python_layout - move a Python file to the layout of issue #194, without changing its code

Python follows CORA's order: BEGIN CODE, the operation or the test cases right away, then AUXILIARY
with the helpers, then END OF CODE; there is no MAIN marker. A trailing `if __name__ == "__main__":`
block stays last, below the helpers. Files under examples/ import with `from cora import <names>`
(every `cora.X` becomes `X`); tests, scripts and library code keep `import cora`.

Syntax:   python scripts/ci/convert_python_layout.py [--check] [--force] [FILE ...]
Inputs:   FILE - the Python files to convert (default: every tracked .py file in the old layout)
Outputs:  the converted files, rewritten in place, and a line per file that was skipped and why
Checks:   after the move, the file's AST must equal the old one apart from the order of the
          top-level statements and the imports; a name that a helper defines and the operation
          needs while the module loads (a decorator, a default value, a base class) stops the file.
Options:  --check lists what would change and writes nothing; --force writes despite a skipped check.
Exit code: 0 if every file converted or already in the layout, 1 if one was skipped.

Authors:       Claude Opus 5.5
Written:       22-November-2026
Last update:   ---
Reviewed:      ---
"""
import argparse
import ast
import io
import os
import re
import subprocess
import sys
import tokenize


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("files", nargs="*")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    files = args.files or tracked_python_files()
    # a skipped file is left as it is, and the exit code says so
    skipped = 0
    for path in files:
        with open(path, encoding="utf-8") as f:
            old = f.read()
        try:
            new = convert(path, old)
        except Skip as why:
            print(f"{path}: skipped: {why}")
            skipped += 1
            if not (args.force and why.text is not None):
                continue
            new = why.text
        if new != old and not args.check:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new)
        if new != old:
            print(f"{path}: converted")
    return 1 if skipped else 0


# -----------------------------------------  AUXILIARY  ------------------------------------------ #

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import layout  # noqa: E402


MARKER_LIKE = re.compile(r"^# [-=]{5,}\s+(BEGIN CODE|AUXILIARY|MAIN|END OF CODE)\s+[-=]{5,} #$")


class Skip(Exception):
    """A file that the converter leaves alone; `text` is what it would have written."""

    def __init__(self, message, text=None):
        super().__init__(message)
        self.text = text


def tracked_python_files():
    out = subprocess.run(["git", "ls-files", "*.py"], capture_output=True, text=True, check=True).stdout
    return [p for p in out.split() if p != os.path.relpath(__file__)]


def is_example(path):
    return path.replace(os.sep, "/").startswith("examples/")


def convert(path, old):
    """The text of the file in the new layout (the same text if it is in it already)."""
    lines = old.split("\n")
    at = layout.marker_lines(lines, "py")
    if len(at["BEGIN"]) != 1 or len(at["END"]) != 1:
        raise Skip("no BEGIN CODE / END OF CODE markers")
    begin, end = at["BEGIN"][0], at["END"][0]
    aux = at["AUXILIARY"][0] if at["AUXILIARY"] else None
    main = at["MAIN"][0] if at["MAIN"] else None
    if main is None:
        new = lines  # no MAIN marker: the new layout, or a file without helpers
    elif aux is None:
        new = lines[:main] + lines[main + 1:]  # an operation without helpers: only the MAIN line goes
    else:
        new = move_blocks(lines, begin, aux, main, end)
    new = tidy("\n".join(new))
    if is_example(path):
        new = from_imports(path, new)
    verify(path, old, new)
    return new


def has_code(lines, first, last):
    return any(line.strip() and not line.startswith("#") for line in lines[first:last])


def move_blocks(lines, begin, aux, main, end):
    """BEGIN, the MAIN blocks, AUXILIARY, the helper blocks, the `__main__` guard, END."""
    tree = ast.parse("\n".join(lines))
    helpers = blocks(lines, tree, aux + 1, main)
    stray = [line for line in lines[begin + 1:aux] if line.strip() and not MARKER_LIKE.match(line)]
    if stray and helpers:  # a comment between BEGIN CODE and AUXILIARY belongs to the first helper
        text, node, blank = helpers[0]
        helpers[0] = (stray + [""] + text, node, blank)
    operation = blocks(lines, tree, main + 1, end)
    guard = [b for b in operation if is_guard(b[1])]
    operation = [b for b in operation if not is_guard(b[1])]
    imports, helpers = hoist_imports(helpers)
    operation, helpers = promote(operation, helpers, definitions(tree))
    head = lines[:begin]
    if imports:
        head = add_imports(head, tree, imports)
    out = head + [lines[begin]] + [""] + render(operation)
    if helpers:
        out += ["", "", layout.marker("AUXILIARY", "py"), ""] + render(helpers)
    if guard:
        out += ["", ""] + render(guard)
    return out + ["", lines[end]] + lines[end + 1:]


def operation_after(helpers, operation):
    """The helpers first, then the operation, no AUXILIARY: the old order without the MAIN marker."""
    return helpers + operation, []


def runs_at_load(nodes):
    """Whether top-level statements do more than define: a call, a loop, a with, a branch."""
    for node in nodes:
        if isinstance(node, (ast.For, ast.While, ast.With, ast.AugAssign, ast.Try, ast.Assert)):
            return True
        if isinstance(node, ast.Expr) and not layout.is_docstring(node):
            return True
        if isinstance(node, ast.If) and not layout.is_import_statement(node):
            return True
    return False


def hoist_imports(helpers):
    """The imports at the top of the helpers, up to the first statement with a side effect (a call
    such as sys.path.insert may decide what an import finds); the others stay where they are."""
    taken = 0
    for _, node, _ in helpers:
        if not layout.is_import_statement(node):
            break
        taken += 1
    return helpers[:taken], helpers[taken:]


def add_imports(head, tree, imports):
    """The header lines with the imports added after its last import."""
    last = max([n.end_lineno for n in tree.body if n.end_lineno <= len(head)
                and (layout.is_import_statement(n) or layout.is_docstring(n))] or [0])
    added = [line for text, _, _ in imports for line in text]
    return head[:last] + added + head[last:]


def promote(operation, helpers, known):
    """Moves the helpers that the operation reads while the module loads (a decorator, a default
    value, a base class, a statement of a script) above the first block that reads them, with
    the helpers they need and the statements with side effects that stood above them."""
    chosen = set()
    needed_by = {}  # helper index -> first operation index that reads it
    if runs_at_load([node for _, node, _ in operation]):
        # the module calls things while it loads, and a helper may register what they use
        return operation_after(helpers, operation)
    defines = [defined_names(node) for _, node, _ in helpers]
    for i, (_, node, _) in enumerate(helpers):
        if layout.is_import_statement(node):  # an import after a side effect keeps its place next to it
            chosen.add(i)
            needed_by[i] = 0
    for j, (_, node, _) in enumerate(operation):
        for name in reach(deftime_names(node), known):
            for i, names in enumerate(defines):
                if name in names:
                    chosen.add(i)
                    needed_by.setdefault(i, j)
    # a helper that a chosen one reads is chosen too, and so is any side effect above it
    changed = True
    while changed:
        changed = False
        for i in sorted(chosen):
            wanted = reach(deftime_names(helpers[i][1]), known)
            for k in range(i):
                side_effect = not isinstance(helpers[k][1], (ast.FunctionDef, ast.AsyncFunctionDef,
                                                              ast.ClassDef, ast.Assign, ast.AnnAssign,
                                                              ast.Import, ast.ImportFrom))
                if k not in chosen and (wanted & defines[k] or side_effect):
                    chosen.add(k)
                    needed_by[k] = needed_by.get(i, 0)
                    changed = True
    # the chosen helpers go above the first block that needs them, in their old order
    if not chosen:
        return operation, helpers
    first = min(needed_by[i] for i in chosen)
    moved = [helpers[i] for i in sorted(chosen)]
    rest = [h for i, h in enumerate(helpers) if i not in chosen]
    return operation[:first] + moved + operation[first:], rest


def blocks(lines, tree, first, last):
    """[(text lines, node, blank lines above)] of the top-level statements in lines[first:last]."""
    nodes = [n for n in tree.body if first <= n.lineno - 1 < last]
    result, cursor = [], first
    for index, node in enumerate(nodes):
        stop = node.end_lineno if index + 1 < len(nodes) else last
        if index + 1 == len(nodes):
            stop = max(node.end_lineno, last_content(lines, node.end_lineno, last))
        text = lines[cursor:stop]
        blank = 0
        while text and not text[0].strip():
            text, blank = text[1:], blank + 1
        result.append((text, node, min(blank, 2)))
        cursor = stop
    return result


def last_content(lines, after, last):
    """The end of the comment lines that follow the last statement of a region."""
    stop = after
    for i in range(after, last):
        if lines[i].strip():
            stop = i + 1
    return stop


def render(items):
    """The lines of the blocks; two empty lines around a def or class, else the original gap."""
    out = []
    for index, (text, node, blank) in enumerate(items):
        if index:
            big = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) or \
                isinstance(items[index - 1][1], (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            out += [""] * (2 if big else max(blank, 1))
        out += text
    return out


def is_guard(node):
    return isinstance(node, ast.If) and ast.unparse(node.test) in ("__name__ == '__main__'",
                                                                    '__name__ == "__main__"')


def defined_names(node):
    """The names that a top-level statement binds."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return {(a.asname or a.name).split(".")[0] for a in node.names}
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}


def definitions(tree):
    """{name: the names that its body reads} of the top-level functions and classes."""
    return {n.name: {m.id for m in ast.walk(n) if isinstance(m, ast.Name) and isinstance(m.ctx, ast.Load)}
            for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def reach(names, known):
    """The names with those that the functions among them read when they run (a call at load time)."""
    found, todo = set(names), list(names)
    while todo:
        for name in known.get(todo.pop(), ()):
            if name not in found:
                found.add(name)
                todo.append(name)
    return found


def deftime_names(node):
    """The names that a statement reads while the module loads: not the body of a function."""
    def reads(*parts):
        return {n.id for part in parts if part is not None for n in ast.walk(part)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        a = node.args
        notes = [x.annotation for x in a.posonlyargs + a.args + a.kwonlyargs + [a.vararg, a.kwarg]
                 if x is not None]
        return reads(*node.decorator_list, *a.defaults, *a.kw_defaults, *notes, node.returns)
    if isinstance(node, ast.ClassDef):
        found = reads(*node.decorator_list, *node.bases, *[k.value for k in node.keywords])
        for statement in node.body:
            found |= deftime_names(statement)
        return found
    return reads(node)


# ------------------------------------------------------------------------ whitespace and imports

def tidy(text):
    """One empty line under BEGIN CODE and above END OF CODE, no run of three empty lines."""
    for name, below in (("BEGIN", True), ("END", False)):
        line = layout.marker(name, "py")
        pattern = re.escape(line) + r"\n+" if below else r"\n+" + re.escape(line)
        text = re.sub(pattern, line + "\n\n" if below else "\n\n" + line, text)
    return re.sub(r"\n{4,}", "\n\n\n", text)


def from_imports(path, text):
    """`import cora` becomes `from cora import <names>`, every `cora.X` becomes `X`."""
    lines = text.split("\n")
    where = [i for i, line in enumerate(lines) if line == "import cora"]
    uses = {}
    tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    for i, tok in enumerate(tokens):
        if tok.type == tokenize.NAME and tok.string == "cora" and i + 2 < len(tokens) \
                and tokens[i + 1].string == "." and tokens[i + 2].type == tokenize.NAME \
                and not (i and tokens[i - 1].string == "."):
            uses[tok.start] = tokens[i + 2].string
    if not where:
        return text  # an import inside a function stays as it is
    names = sorted(set(uses.values()), key=lambda s: (s.lower(), s))
    if len(where) != 1 or not names or any(n.startswith("_") for n in names):
        raise Skip("cannot rewrite `import cora` (several, none used, or a private name)")
    bound = {name for name in names if name in rebound(text)}
    if bound:
        raise Skip(f"the file binds {', '.join(sorted(bound))} itself; rename it by hand")
    # rewrite from the back, so that earlier columns stay valid: delete `cora.`
    rows = text.split("\n")
    for (row, col), _ in sorted(uses.items(), reverse=True):
        rows[row - 1] = rows[row - 1][:col] + rows[row - 1][col + len("cora."):]
    line = "from cora import " + ", ".join(names)
    if len(line) > 100:
        line = "from cora import (\n" + "".join(f"    {n},\n" for n in names) + ")"
    start = rows.index("import cora")
    rows[start:start + 1] = line.split("\n")
    return "\n".join(rows)


def rebound(text):
    """The names that the file binds anywhere (assignment, def, class, argument, import, loop)."""
    found = set()
    for n in ast.walk(ast.parse(text)):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            found.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.add(n.name)
        elif isinstance(n, ast.arg):
            found.add(n.arg)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            found |= {(a.asname or a.name).split(".")[0] for a in n.names}
    return found


# ------------------------------------------------------------------------------ the AST check

class StripCora(ast.NodeTransformer):
    """`cora.X` to `X`: the form of the examples, to compare an example with its old self."""

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "cora":
            return ast.copy_location(ast.Name(id=node.attr, ctx=node.ctx), node)
        return node


def shape(text, example):
    """The sorted dumps of the top-level statements that are not imports."""
    tree = ast.parse(text)
    if example:
        tree = StripCora().visit(tree)
    keep = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    return sorted(ast.dump(n) for n in keep)


def comments(text):
    """The comments of the file except the marker lines, sorted, to see that none is lost."""
    found = [t.string for t in tokenize.generate_tokens(io.StringIO(text).readline)
             if t.type == tokenize.COMMENT and not MARKER_LIKE.match(t.string)]
    return sorted(found)


def verify(path, old, new):
    """The statements are the same set, apart from order and imports; the markers are in order."""
    example = is_example(path) and "from cora import" in new
    if shape(old, example) != shape(new, example):
        raise Skip("the AST changed (not only statement order and imports)", new)
    if comments(old) != comments(new):
        raise Skip("a comment was lost or added", new)
    if example and "import cora" in new.split("\n"):
        raise Skip("an example still has `import cora`", new)


if __name__ == "__main__":
    sys.exit(main())

# ----------------------------------------  END OF CODE  ----------------------------------------- #
