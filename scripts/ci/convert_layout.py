"""convert_layout - brings every C++ and Python file into the layout that layout.py checks

Adds what a file lacks and leaves the rest: the author block back-filled from the git history, the
CORA authors from the CORA files the tags name, the markers BEGIN CODE, AUXILIARY, MAIN and END OF
CODE, a blank line after a docstring's first line, and entries that share a line split in two. Runs
again without change, so a rebase can rerun it. Never touches `Last update:` of a file that has
the block, nor any `Reviewed:`.

Syntax:   python scripts/ci/convert_layout.py [--cora ~/CORA] [--check] [files ...]
Outputs:  the number of files changed; with --check, the files that would change
See also: scripts/ci/layout.py, CONTRIBUTING.md

Authors:       Claude Sonnet 5.5
Written:       06-November-2026
Last update:   ---
Reviewed:      ---
"""
import argparse
import ast
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import layout  # noqa: E402


WIDTH = 98  # the longest line of the CORA authors
SWEEP = 150  # a commit that changes this many files is a tree-wide sweep, not the file's own change
TAG = re.compile(r"^(?:// |# )(CORA|CORA\.cpp only|PROPERTY|ORACLE): ")
TRAILER = re.compile(r"^Co-authored-by: (.+?) <", re.I | re.M)
TWO_ENTRIES = re.compile(r"^(\S.*?);\s+([A-Za-z_]\w*(?:, [A-Za-z_]\w*)* - [a-z].*)$")
IGNORED_CHANGE = re.compile(r"^\s*($|//|#|/\*|\*|using\b|extern\b)|BEGIN CODE|END OF CODE|MAIN  =|AUXILIARY")


def git(*args):
    return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True, check=True).stdout


def history(path):
    """The commits that touched the file, oldest first: (date, author, [trailer names], files in
    the commit, meaningful change)."""
    out = git("log", "--follow", "--reverse", "-w", "-p", "--date=format:%d-%B-%Y",
              "--format=%x01%ad%x02%an%x02%H%x02%B%x03", "--", path)
    commits = []
    for record in out.split("\x01")[1:]:
        head, _, patch = record.partition("\x03")
        date, author, sha, body = head.split("\x02", 3)
        names = TRAILER.findall(body)
        names += re.findall(r"^Co-Authored-By: (Claude [^<]+?) <", body, re.M | re.I) if not names else []
        size = len(git("show", "--name-only", "--format=", sha).split())
        changed = [line[1:] for line in patch.split("\n") if line[:1] in "+-" and line[:3] not in ("+++", "---")]
        meaningful = any(not IGNORED_CHANGE.search(line) for line in changed)
        commits.append((date, author, [n.strip() for n in names] or [author], size, meaningful))
    return commits


def authors_from(commits):
    """(authors, written, last update) from the history of a file; a sweep adds nothing but its
    first commit, the creation."""
    if not commits:
        return ["Claude Sonnet 5.5"], "06-November-2026", "---"
    names, last = [], "---"
    for k, (date, _, who, size, meaningful) in enumerate(commits):
        if k and size >= SWEEP:
            continue
        for name in who:
            if name not in names:
                names.append(name)
        if k and meaningful:
            last = date
    written = commits[0][0]
    return names, written, "---" if last == written else last


def cora_authors(tags, cora_dir):
    """The union, in order, of the Authors: lines of the CORA files that the tags name."""
    names = []
    for tag in tags:
        path = os.path.join(cora_dir, re.split(r"[:\s]", tag)[0])
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                m = re.match(r"^%\s*Authors?:\s*(.*)$", line)
                if m:
                    names += [n.strip() for n in m.group(1).split(",") if n.strip() and n.strip() not in names]
                    break
    return names


def split_entries(body):
    """Entries that share a line of Inputs or Outputs go on lines of their own."""
    out, current, indent = [], None, ""
    for line in body:
        m = layout.LABEL.match(line) if line and not line.startswith(" ") else None
        if m:
            current = m.group(1)
            value = line[len(current) + 1:].lstrip()
            prefix = line[:len(line) - len(value)]
            indent = " " * len(prefix)
        elif line.startswith("    ") and current:
            value = line.lstrip()
            prefix = line[:len(line) - len(value)]
        else:
            current = current if line.strip() else None
            out.append(line)
            continue
        parts = split_value(value) if current in ("Inputs", "Outputs") else [value]
        out.append(prefix + parts[0])
        out += [indent + part for part in parts[1:]]
    return out


def split_value(value):
    """Splits `name - one;  other - two` at the semicolon into its entries."""
    parts = []
    while True:
        m = TWO_ENTRIES.match(value)
        if not m or " - " not in m.group(1):
            return parts + [value]
        parts.append(m.group(1).rstrip())
        value = m.group(2)


def wrap_names(sign, label, names, width):
    """The names in lines of at most `width` columns, the later lines under the first name."""
    lines, current = [], sign + label
    for k, name in enumerate(names):
        piece = name + ("," if k < len(names) - 1 else "")
        if len(current) + len(piece) > width and current.strip(" /#") != label.strip():
            lines.append(current.rstrip())
            current = sign + " " * len(label)
        current += piece + " "
    return lines + [current.rstrip()]


def author_lines(values, lang, cora):
    """The lines of the author block (with the CORA authors line above it)."""
    lines, sign = [], "// " if lang == "cpp" else ""
    if cora:
        lines += wrap_names(sign, "CORA authors: ", cora[0].split(", "), WIDTH)
    for label in ("Authors", "Written", "Last update"):
        lines.append(layout.field_line(label, values[label], lang))
    first, *rest = values["Reviewed"]
    lines.append(layout.field_line("Reviewed", first, lang))
    lines += [("// " if lang == "cpp" else "") + " " * layout.FIELD_COLUMN + r for r in rest]
    return lines


def existing_values(body):
    """The author fields a docstring body already has: {label: value} (Reviewed: a list)."""
    values, blocks = {}, layout.split_blocks(body)
    for label, first, last in blocks:
        if label in ("Authors", "Written", "Last update", "Reviewed"):
            lines = [body[first][len(label) + 1:].strip()] + [body[k].strip() for k in range(first + 1, last + 1)]
            values[label] = lines if label == "Reviewed" else lines[0]
        elif label == "CORA authors":
            values[label] = " ".join([body[first][len(label) + 1:].strip()]
                                     + [body[k].strip() for k in range(first + 1, last + 1)])
    return values


def strip_author_block(body):
    """The docstring body without its author lines, trailing blank lines removed."""
    drop = set()
    for label, first, last in layout.split_blocks(body):
        if label in layout.AUTHOR_LABELS:
            drop.update(range(first, last + 1))
    kept = [line for k, line in enumerate(body) if k not in drop]
    while kept and not kept[-1].strip():
        kept.pop()
    return kept


def new_body(body, path, lang, tags, cora_dir, commits):
    """The docstring body with the blank line after the first line, split entries and the author
    block at its end."""
    old = existing_values(body)
    names, written, updated = authors_from(commits())  if "Authors" not in old else (None, None, None)
    values = {"Authors": old.get("Authors", ", ".join(names or [])), "Written": old.get("Written", written),
              "Last update": old.get("Last update", updated), "Reviewed": old.get("Reviewed", ["---"])}
    cora = old.get("CORA authors") or ", ".join(cora_authors(tags, cora_dir))
    kept = strip_author_block(body)
    if len(kept) > 1 and kept[1].strip():
        kept.insert(1, "")
    kept = split_entries(kept)
    # the author block follows the tags directly, else a blank line
    tagged = bool(kept) and re.match(r"^(CORA|CORA\.cpp only|PROPERTY|ORACLE)\b", kept[-1])
    separator = [] if tagged or not kept or not kept[-1].strip() else [""]
    block = author_lines(values, lang, [cora] if cora and tags else [])
    block = [line[3:] if lang == "cpp" else line for line in block]
    return kept + separator + block


# ---------------------------------------------------------------------------------- C++ files

def comment_before(lines, k, sign):
    """The first line of the comment lines that directly precede line k (no blank between)."""
    while k > 0 and lines[k - 1].lstrip().startswith(sign) and not lines[k - 1].startswith(sign + " -") \
            and not lines[k - 1].startswith(sign + " ="):
        k -= 1
    return k


def comment_run(lines, k):
    """The same for C++ comments."""
    return comment_before(lines, k, "//")


def cpp_prelude_end(lines, start):
    """The line where the code starts: after the leading comments, includes, preprocessor lines and
    using-declarations that follow the docstring."""
    k, in_macro = start, False
    while k < len(lines):
        line = lines[k]
        if in_macro:
            in_macro = line.rstrip().endswith("\\")
        elif not line.strip() or layout.INCLUDE_LIKE.match(line) or line.startswith("//"):
            in_macro = line.startswith("#") and line.rstrip().endswith("\\")
        else:
            break
        k += 1
    return comment_run(lines, k)


def strip_markers(lines):
    """Removes every marker line (and the empty lines they leave in a row beyond two)."""
    names = {layout.marker(n, lang) for n in ("BEGIN", "AUXILIARY", "MAIN", "END") for lang in ("cpp", "py")}
    kept = [line for line in lines if line not in names]
    out = []
    for line in kept:
        if line.strip() or len(out) < 2 or out[-1].strip() or out[-2].strip():
            out.append(line)
    lines[:] = out


def valid_markers(lines, path, lang):
    """Whether the markers present are all right, or there are none (they are then added)."""
    if not any(layout.marker_lines(lines, lang)[n] for n in ("BEGIN", "AUXILIARY", "MAIN", "END")):
        return True
    issues = []
    layout.check_markers(path, lines, lang, issues)
    return not issues


def respace(lines, lang):
    """The empty lines around the markers: one, and two above MAIN (and above a Python BEGIN that
    follows imports, which ruff separates from what comes next by two)."""
    for name in ("END", "AUXILIARY", "BEGIN", "MAIN"):
        for at in reversed(layout.marker_lines(lines, lang)[name]):
            first = at
            while first > 0 and not lines[first - 1].strip():
                first -= 1
            last = at + 1
            while last < len(lines) and not lines[last].strip():
                last += 1
            after_code = lang == "py" and first > 0 and lines[first - 1].startswith(("import ", "from "))
            above = 2 if name == "MAIN" or (name == "BEGIN" and after_code) else 1
            lines[first:last] = [""] * above + [lines[at]] + [""]


def put(lines, at, name, lang, above=1, below=1):
    """Inserts a marker line at `at` with `above` empty lines before and `below` after it."""
    while at > 0 and not lines[at - 1].strip():
        at -= 1
    end = at
    while end < len(lines) and not lines[end].strip():
        end += 1
    lines[at:end] = [""] * above + [layout.marker(name, lang)] + [""] * below
    return at + above


def cpp_markers(lines, path, doc_end):
    """BEGIN, AUXILIARY, MAIN and END OF CODE where a file lacks them; markers that cannot be
    right (out of order, doubled, code above BEGIN) are removed first."""
    if not valid_markers(lines, path, "cpp"):
        strip_markers(lines)
    at = layout.marker_lines(lines, "cpp")
    if not at["BEGIN"]:
        put(lines, cpp_prelude_end(lines, doc_end), "BEGIN", "cpp")
    if not at["END"] and layout.marker_lines(lines, "cpp")["BEGIN"]:
        while lines and not lines[-1].strip():
            lines.pop()
        lines += ["", layout.marker("END", "cpp"), ""]
    at = layout.marker_lines(lines, "cpp")
    if path.endswith(".cpp") and at["BEGIN"] and at["END"] and not at["MAIN"]:
        place_main(lines, at["BEGIN"][0], at["END"][0])
    at = layout.marker_lines(lines, "cpp")
    if at["MAIN"] and at["BEGIN"] and not at["AUXILIARY"]:
        place_aux(lines, at["BEGIN"][0], at["MAIN"][0])
    respace(lines, "cpp")


def place_main(lines, begin, end):
    """MAIN above `int main`, else after the last anonymous namespace, else after the opening
    `namespace X {`, else right under BEGIN CODE."""
    code = layout.scan(lines, "cpp")
    main = next((i for i in range(begin + 1, end) if re.match(r"\s*int\s+main\s*\(", code[i][0])), None)
    if main is not None:
        put(lines, comment_run(lines, main), "MAIN", "cpp", above=2)
        return
    closes = [i for i in range(begin + 1, end) if re.match(r"^\}\s*//\s*namespace\s*$", lines[i])]
    if closes:
        at = closes[-1] + 1
    else:
        first = next((i for i in range(begin + 1, end) if code[i][0].strip()), end)
        at = first + 1 if re.match(r"^namespace [\w:]+ \{$", lines[first]) else first
        at = comment_run(lines, at) if at == first else at
    put(lines, at, "MAIN", "cpp", above=2)


def place_aux(lines, begin, main):
    """AUXILIARY right under BEGIN CODE (and the namespace it opens) when code lies above MAIN."""
    code = layout.scan(lines, "cpp")
    inner = [i for i in range(begin + 1, main) if code[i][0].strip()
             and not re.match(r"^namespace [\w:]+ \{$", lines[i])]
    if inner:
        at = inner[0]
        put(lines, comment_run(lines, at), "AUXILIARY", "cpp")


# ------------------------------------------------------------------------------ Python files

def python_prelude_end(lines):
    """The 0-based line after the docstring, its comment tags and the leading imports."""
    tree = ast.parse("\n".join(lines))
    body = tree.body
    end = 0
    if body and layout.is_docstring(body[0]):
        end = body[0].end_lineno
        body = body[1:]
    for node in body:
        if not layout.is_import_statement(node):
            break
        end = node.end_lineno
    while end < len(lines) and lines[end].startswith("# CORA"):
        end += 1
    return end


def python_markers(lines, path):
    """BEGIN, AUXILIARY, MAIN and END OF CODE where a file lacks them; markers that cannot be
    right are removed first."""
    if not valid_markers(lines, path, "py"):
        strip_markers(lines)
    at = layout.marker_lines(lines, "py")
    if not at["BEGIN"]:
        end = python_prelude_end(lines)
        if any(line.strip() for line in lines[end:]):
            put(lines, end, "BEGIN", "py")
    if not layout.marker_lines(lines, "py")["END"] and layout.marker_lines(lines, "py")["BEGIN"]:
        while lines and not lines[-1].strip():
            lines.pop()
        lines += ["", layout.marker("END", "py"), ""]
    at = layout.marker_lines(lines, "py")
    if at["BEGIN"] and at["END"] and not at["MAIN"] and layout.python_needs_main(lines):
        main = layout.first_main_statement(lines, at["BEGIN"][0], at["END"][0])
        if main is not None:
            put(lines, comment_before(lines, main, "#"), "MAIN", "py", above=2)
    at = layout.marker_lines(lines, "py")
    if at["MAIN"] and at["BEGIN"] and not at["AUXILIARY"]:
        stop = at["MAIN"][0]
        code = layout.scan(lines, "py")
        # code between BEGIN and MAIN needs an AUXILIARY marker
        inner = [i for i in range(at["BEGIN"][0] + 1, stop) if code[i][0].strip()]
        if inner:
            put(lines, comment_before(lines, inner[0], "#"), "AUXILIARY", "py")
    respace(lines, "py")


def convert_python(path, text, cora_dir, commits):
    lines = text.split("\n")
    body, _ = layout.read_doc(lines, "py")
    if not body or not lines[0].startswith('"""'):
        return text
    close = 0 if lines[0].count('"""') >= 2 else next(k for k in range(1, len(lines)) if '"""' in lines[k])
    tags = [m.group(1) for m in (re.match(r"^# CORA: (\S+)", line) for line in lines[:80]) if m]
    new = new_body(body, path, "py", tags, cora_dir, commits)
    lines[:close + 1] = ['"""' + new[0]] + new[1:] + ['"""']
    python_markers(lines, path)
    return "\n".join(lines)


# --------------------------------------------------------------------------------- the driver

def convert_cpp(path, text, cora_dir, commits):
    lines = text.split("\n")
    body, _ = layout.read_doc(lines, "cpp")
    if not body or layout.read_doc(lines, "cpp")[0][0].strip() == "":
        return text
    tags = [m.group(1) for m in (re.match(r"^// CORA: (\S+)", line) for line in lines[:len(body)]) if m]
    new = new_body(body, path, "cpp", tags, cora_dir, commits)
    header = ["// " + line if line else "//" for line in new]
    lines[:len(body)] = header
    if len(lines) > len(header) and lines[len(header)].strip():
        lines.insert(len(header), "")  # a blank line ends the docstring
    cpp_markers(lines, path, len(header))
    return "\n".join(lines)


def convert(path, cora_dir):
    with open(os.path.join(ROOT, path), encoding="utf-8", newline="") as handle:
        raw = handle.read()
    text = raw.replace("\r\n", "\n")
    cache = []

    def commits():
        if not cache:
            cache.append(history(path))
        return cache[0]

    new = convert_cpp(path, text, cora_dir, commits) if layout.language(path) == "cpp" \
        else convert_python(path, text, cora_dir, commits)
    if not new.endswith("\n"):
        new += "\n"
    return raw, new.replace("\n", "\r\n") if "\r\n" in raw else new


def tracked(paths):
    files = paths or git("ls-files").split("\n")
    return [f for f in files if f and layout.language(f) and os.path.isfile(os.path.join(ROOT, f))]


# ============================================  MAIN  ============================================ #

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cora", default=os.path.expanduser("~/CORA"), help="a CORA checkout")
    parser.add_argument("--check", action="store_true", help="only list the files that would change")
    parser.add_argument("files", nargs="*", help="files relative to the repository (default: all)")
    args = parser.parse_args()
    paths = tracked(args.files)
    # the history of every file, in parallel: git is the slow part
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda p: (p, *convert(p, args.cora)), paths))
    changed = [(p, new) for p, raw, new in results if raw != new]
    for path, new in changed:
        print(path)
        if not args.check:
            with open(os.path.join(ROOT, path), "w", encoding="utf-8", newline="") as handle:
                handle.write(new)
    print(f"{len(changed)} of {len(paths)} files {'would change' if args.check else 'converted'}")
    return 1 if args.check and changed else 0


if __name__ == "__main__":
    sys.exit(main())

# ----------------------------------------  END OF CODE  ----------------------------------------- #
