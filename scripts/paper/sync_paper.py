"""sync_paper - copy the generated paper files into the paper repository, keeping its captions

The generators write the data of the figures and tables (paper/generated/); the captions are the
paper's own. Each file goes to its path in the paper; a table whose paper version has a caption
keeps it: the caption is put back before the table's label.

Syntax:   python scripts/paper/sync_paper.py PAPER [FILE ...]
Reads:    the files given (default: every file of paper/generated/ in PLACES), and their versions in
          the paper repository PAPER
Writes:   PAPER/<the path of PLACES> for each file
Exit code: 0 if every file is copied, else 1 with the files that have no place in the paper.

Authors:       Claude Opus 5.5
Written:       26-November-2026
Last update:   ---
Reviewed:      ---
"""
import argparse
import sys
from pathlib import Path


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("paper", help="the paper repository")
    parser.add_argument("files", nargs="*", help="the generated files (default: all of paper/generated/)")
    args = parser.parse_args()
    files = [Path(f) for f in args.files] or [GENERATED / name for name in PLACES]
    unknown = [f.name for f in files if f.name not in PLACES]
    if unknown:
        print("sync_paper: no place in the paper for " + ", ".join(unknown))
        return 1
    # each file to its place, with the caption that the paper has there
    for f in files:
        target = Path(args.paper) / PLACES[f.name]
        old = target.read_text(encoding="utf-8") if target.exists() else ""
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(keep_caption(f.read_text(encoding="utf-8"), old), encoding="utf-8")
        print(f"sync_paper: {f.name} -> {PLACES[f.name]}")
    return 0


# -----------------------------------------  AUXILIARY  ------------------------------------------ #

REPO = Path(__file__).resolve().parents[2]
GENERATED = REPO / "paper" / "generated"
# the path of each generated file in the paper repository
PLACES = {
    "status.tikz": "figures/status/status.tikz",
    **{f"status-classes-{s}.tikz": f"figures/status/status-classes-{s}.tikz"
       for s in ("sets", "contdynamics", "others", "specification", "misc")},
    "arch-comp-table.tex": "sections/data/arch-comp-table.tex",
    "arch-comp-summary.tex": "sections/data/arch-comp-summary.tex",
    "vnn-comp-table.tex": "sections/data/vnn-comp-table.tex",
}


def caption_of(text):
    """The paper's `\\caption{...}` (braces balanced), or None."""
    start = text.find("\\caption{")
    if start < 0:
        return None
    depth = 0
    for i in range(start + len("\\caption"), len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return text[start:i + 1]
    return None


def keep_caption(new, old):
    """The generated text with the old version's caption put back before its first label."""
    caption = caption_of(old)
    at = new.find("\\label{")
    if caption is None or at < 0 or "\\caption{" in new:
        return new
    return new[:at] + caption + "\n" + new[at:]


if __name__ == "__main__":
    sys.exit(main())

# ----------------------------------------  END OF CODE  ----------------------------------------- #
