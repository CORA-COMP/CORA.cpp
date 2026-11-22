"""cora_references - the CORA files that cite papers, for the check that CORA.cpp ports them

Syntax:   python3 scripts/ci/cora_references.py <path of CORA> > scripts/ci/cora_references.txt
Inputs:   CORA - the root folder of the CORA checkout
Outputs:  the paths (relative to CORA) of the .m files with a '% References:' block that has entries
See also: scripts/ci/layout.py

Authors:       Claude Opus 5.5
Written:       22-November-2026
Last update:   ---
Reviewed:      ---
"""
import os
import re
import sys


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

def main(cora):
    """Print the paths, sorted, one per line."""
    found = []
    # every .m file of CORA except the unit tests, kept when its block has entries
    for folder, _, files in os.walk(cora):
        for name in files:
            full = os.path.join(folder, name)
            rel = os.path.relpath(full, cora)
            if name.endswith(".m") and "unitTests" not in rel and references_of(full):
                found.append(rel)
    print("\n".join(sorted(found)))


# -----------------------------------------  AUXILIARY  ------------------------------------------ #

def references_of(path):
    """The entry lines of the '% References:' block of a .m file; [] if it has none or only '-'."""
    with open(path, errors="replace") as handle:
        lines = handle.read().split("\n")
    for i, line in enumerate(lines):
        if re.match(r"^%\s*References:\s*$", line):
            block = []
            # the block ends at the first line that is not a comment or is an empty one
            for text in lines[i + 1:]:
                if not text.startswith("%") or not text[1:].strip():
                    break
                block.append(text[1:].strip())
            return [t for t in block if t not in ("-", "---")]
    return []


if __name__ == "__main__":
    main(sys.argv[1])

# ----------------------------------------  END OF CODE  ----------------------------------------- #
