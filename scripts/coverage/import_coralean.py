"""import_coralean - what CORALean says about each CORA method, as one JSON file

Syntax:   python scripts/coverage/import_coralean.py <CORALean checkout>
Reads:    references/coverage/<group>/<class>.md of the checkout, rows `| `method` | state | links | note |`.
Writes:   coverage/coralean-status.json: {"coralean_commit": <hash>, "rows": {"group/class/method":
          {"status": "proved" | "float" | "bound" | "out-of-scope", "decls": [<Lean declaration>]}}}.
          A row CORALean has not done (`—`) is absent. `decls` are the declarations the row links, in
          the order of its table. CORALean can publish the same file with its releases; this script
          runs on demand, not in CI.
Exit code: 0, or 1 for a checkout without the tables.

Authors:       Tobias Ladner
Written:       02-November-2026
Last update:   ---
Reviewed:      ---
"""
import json
import re
import subprocess
import sys
from pathlib import Path


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "coverage" / "coralean-status.json"
STATES = ("proved", "float", "bound", "out-of-scope")
LINK = re.compile(r"\[`([^`]+)`\]\([^)\s]+\.lean\)")


def cells_of(line):
    """The cells of a markdown table row (`\\|` is a pipe inside a cell)."""
    return [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]


def rows_of(table, group, cls):
    """`{key: {status, decls}}` of one class table; `—` rows are left out."""
    out = {}
    for line in table.read_text(encoding="utf-8").splitlines():
        cells = cells_of(line) if line.startswith("| `") else []
        if len(cells) != 4 or cells[1] not in STATES:
            continue
        decls = list(dict.fromkeys(LINK.findall(cells[2])))
        out[f"{group}/{cls}/{cells[0].strip('`')}"] = {"status": cells[1], "decls": decls}
    return out


# ============================================  MAIN  ============================================ #

def main():
    if len(sys.argv) != 2 or not (Path(sys.argv[1]) / "references" / "coverage").is_dir():
        print(__doc__)
        return 1
    lean = Path(sys.argv[1])
    rows = {}
    for table in sorted((lean / "references" / "coverage").glob("*/*.md"), key=lambda p: p.as_posix()):
        rows.update(rows_of(table, table.parent.name, table.stem))
    commit = subprocess.run(["git", "-C", str(lean), "rev-parse", "HEAD"], capture_output=True, text=True)
    text = json.dumps({"coralean_commit": commit.stdout.strip() or "unknown",
                       "rows": dict(sorted(rows.items()))}, indent=1, ensure_ascii=False)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(text + "\n", encoding="utf-8")
    print(f"import_coralean: wrote {OUT.relative_to(ROOT)} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# ----------------------------------------  END OF CODE  ----------------------------------------- #
