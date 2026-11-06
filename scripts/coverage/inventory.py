"""inventory - the rows of CORA: every method file of its classes, the denominator of the coverage

Syntax:   python scripts/coverage/inventory.py [--cora <checkout>] [--check] [--against <file>]
Writes:   coverage/cora-inventory.json: {"cora_commit": <hash>, "groups": {group: {class: [method]}}},
          the rules of cora_rows.py (the same as CORALean's references/coverage/cora-inventory.json).
Checks:   --check compares the committed file with the checkout (CORA_DIR or ~/CORA) and is skipped,
          not failed, without one; --against <file> compares `groups` with a file in that format.
Exit code: 0, or 1 if a comparison differs.

Authors:       Tobias Ladner
Written:       02-November-2026
Last update:   ---
Reviewed:      ---
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cora_rows  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "coverage" / "cora-inventory.json"


def cora_checkout(given=None):
    """The CORA checkout: the argument, CORA_DIR or ~/CORA, if it is one."""
    for candidate in (given, os.environ.get("CORA_DIR"), str(Path.home() / "CORA")):
        if candidate and (Path(candidate) / "contSet").is_dir():
            return Path(candidate)
    return None


def cora_commit(cora):
    """The commit a CORA checkout is at."""
    out = subprocess.run(["git", "-C", str(cora), "rev-parse", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def render(inventory):
    """The text of the file: the groups as CORALean writes them."""
    return json.dumps(inventory, indent=1, ensure_ascii=False) + "\n"


def load(path=INVENTORY):
    """`{group: {class: [method]}}` of a file in this format, {} if there is none."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))["groups"]
    except (OSError, ValueError, KeyError):
        return {}


def differences(a, b):
    """The classes (`group/class`) in which two inventories differ."""
    return sorted(f"{g}/{c}" for g in set(a) | set(b)
                  for c in set(a.get(g, {})) | set(b.get(g, {}))
                  if a.get(g, {}).get(c) != b.get(g, {}).get(c))


# ============================================  MAIN  ============================================ #

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cora", help="a CORA checkout (default: CORA_DIR, ~/CORA)")
    parser.add_argument("--check", action="store_true", help="compare the committed file with CORA")
    parser.add_argument("--against", help="an inventory file whose groups must equal the output")
    args = parser.parse_args()
    cora = cora_checkout(args.cora)
    if cora is None:
        print("inventory: no CORA checkout (set CORA_DIR); skipped")
        return 0 if args.check else 1
    groups = cora_rows.inventory_of(cora)
    status = 0
    if args.against:
        other = json.loads(Path(args.against).read_text(encoding="utf-8"))
        other = other.get("groups", other)
        wrong = differences(groups, other)
        print(f"inventory: equals {args.against}" if not wrong else
              f"inventory: differs from {args.against} in {', '.join(wrong)}")
        status |= bool(wrong)
    if args.check:
        wrong = differences(groups, load())
        print("inventory: the committed file is current" if not wrong else
              f"inventory: the committed file differs from {cora} in {', '.join(wrong)}; "
              "rerun scripts/coverage/inventory.py")
        status |= bool(wrong)
    elif not args.against:
        INVENTORY.parent.mkdir(exist_ok=True)
        INVENTORY.write_text(render({"cora_commit": cora_commit(cora), "groups": groups}),
                             encoding="utf-8")
        print(f"inventory: wrote {INVENTORY.relative_to(ROOT)} ({sum(map(len, groups.values()))} classes)")
    return status


if __name__ == "__main__":
    sys.exit(main())

# ----------------------------------------  END OF CODE  ----------------------------------------- #
