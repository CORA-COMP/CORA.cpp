"""cora_rows - which CORA method a CORA file is: the one place that knows the inventory rules

A row is `group/class/method`. It is one `.m` file of a class of `contSet/`, `contDynamics/`,
`hybridDynamics/`, `specification/`, `nn/` or of the group `others` (`matrixSet/`, `discrDynamics/`,
`converter/`); `private/` and `global/` have none. The inventory (inventory.py) lists the rows of a
CORA checkout and the tags (coverage.py) name rows, both through `row_of_path`.
"""
import re

GROUPS = ("contSet", "contDynamics", "hybridDynamics", "nn", "specification", "others")
OTHERS_DIRS = ("matrixSet", "discrDynamics", "converter")
# Example inputs of discrDynamics/ProbOccupancyPrediction: scripts and models, no rows.
PROB_EXAMPLE_DIRS = ("scenarios", "velocityProfiles", "vehicleModels")
PROB = "ProbOccupancyPrediction"
ROOT_DIRS = ("contSet", "contDynamics", "hybridDynamics", "nn", "specification") + OTHERS_DIRS


def at_class(directory):
    """The class of a folder `@class`, else None."""
    return directory[1:] if directory.startswith("@") else None


def row_of_path(path):
    """`(group, class, method)` of a CORA file, method None for a single-file class (its methods
    are in the file, see `classdef_methods`); None for a file without a row (`private/`,
    `global/`); ValueError for a path that no row can have."""
    parts = path.split("/")
    if not path.endswith(".m"):
        raise ValueError(f"`{path}` is no .m file")
    if parts[0] == "global" or "private" in parts[:-1]:
        return None
    name, dirs = parts[-1][:-2], parts[1:-1]
    if parts[0] not in ROOT_DIRS:
        raise ValueError(f"`{path}` is in no CORA folder of the inventory")
    cls = at_class(dirs[0]) if dirs else None
    found = None
    if parts[0] in ("contSet", "contDynamics", "hybridDynamics") and len(dirs) == 1:
        found = (parts[0], cls, name)
    elif parts[0] == "nn" and dirs[:1] == ["layers"]:
        found = ("nn", "layers", name)
    elif parts[0] == "nn" and dirs == ["@neuralNetwork"]:
        found = ("nn", cls, name)
    elif parts[0] == "specification":
        dirs = dirs[1:] if dirs[:1] == ["signals"] else dirs
        cls = at_class(dirs[0]) if dirs else None
        found = ("specification", cls, name) if len(dirs) == 1 and cls else None
        found = found or (("specification", name, None) if not dirs else None)
    elif parts[0] in OTHERS_DIRS:
        found = other_row(parts[0], dirs, cls, name)
    if found is None or found[1] is None:
        raise ValueError(f"`{path}` is no method file of an inventory class")
    return found


def other_row(root, dirs, cls, name):
    """The row of a file under `matrixSet/`, `discrDynamics/` or `converter/`."""
    if root == "converter":
        return ("others", dirs[0], name) if dirs else None
    if len(dirs) == 1 and cls:
        return ("others", cls, name)
    if dirs[:1] != [PROB]:
        return None
    rest = dirs[1:]
    if len(rest) == 1 and at_class(rest[0]):
        return ("others", at_class(rest[0]), name)
    if any(d.startswith("@") or d in PROB_EXAMPLE_DIRS for d in rest):
        return None
    return ("others", PROB, name)


METHODS_HEADER = re.compile(r"^methods\b(.*)$")
METHOD_FN = re.compile(r"^    function\s+(?:[^=\n(]*=\s*)?(\w+)")
ABSTRACT_DECL = re.compile(r"^    (?:[\w\[\], ]+?=\s*)?(\w+)\s*(?:\(|%|$)")


def classdef_methods(path):
    """The public methods of a single-file MATLAB class, constructor included: the `function`s of
    the `methods` blocks that are not `Access = private` or `protected`, and the declarations of
    `Abstract` blocks. Local and nested functions have no row."""
    names, public, abstract = set(), False, False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        header = METHODS_HEADER.match(line)
        if header:
            public = not re.search(r"Access\s*=\s*(private|protected)", header.group(1))
            abstract = "Abstract" in header.group(1)
        elif line.startswith("end"):
            public, abstract = False, False
        elif public and not abstract and METHOD_FN.match(line):
            names.add(METHOD_FN.match(line).group(1))
        elif public and abstract and not line.lstrip().startswith("%") and ABSTRACT_DECL.match(line):
            names.add(ABSTRACT_DECL.match(line).group(1))
    return sorted(names)


def inventory_of(cora):
    """`{group: {class: [method]}}` of a CORA checkout (a Path), by the rules above."""
    out = {g: {} for g in GROUPS}
    for root in ROOT_DIRS:
        for file in sorted((cora / root).rglob("*.m")):
            rel = file.relative_to(cora).as_posix()
            try:
                row = row_of_path(rel)
            except ValueError:
                continue
            if row is None:
                continue
            group, cls, method = row
            methods = out[group].setdefault(cls, [])
            methods.extend([method] if method else classdef_methods(file))
    return {g: {c: sorted(m) for c, m in sorted(classes.items())} for g, classes in out.items()}


def key_of(group, cls, method):
    """The name of a row in the JSON files and the ledger."""
    return f"{group}/{cls}/{method}"
