"""stubs - the type stubs (.pyi) of the compiled module `cora._cora`, written from the bindings

pybind11 prints every signature in the docstring of the function; those are parsed here, with the
standard library only. `Array` is a numpy float64 array or a torch tensor, `object` is `Any`, an
int is any `SupportsInt` (written `int`), a default that is no literal is `...`.

Syntax:   python scripts/stubs.py --out <package dir>      writes the stubs and py.typed there
          python scripts/stubs.py --check <package dir>    exit 1 if the stubs there differ from fresh ones
Inputs:   the built package (its build directory on PYTHONPATH)
Outputs:  one stub per compiled module (the submodule cora._cora.lean has its own) and py.typed
See also: CONTRIBUTING.md (Python stubs), tests/python/test_stubs.py

Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
Written:       04-November-2026
Last update:   ---
Reviewed:      ---
"""
import argparse
import importlib
import inspect
import os
import re
import sys
import types

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

ROOT_MODULE = "cora._cora"

HEADER = '''"""{doc}"""
from __future__ import annotations

import collections.abc
import typing
from typing import Any, overload

import numpy
import numpy.typing
import torch
{imports}#: A numpy float64 array (Eigen backend) or a torch tensor (torch backend): what the package takes and returns.
Array = numpy.typing.NDArray[numpy.float64] | torch.Tensor
'''

LITERAL = re.compile(r"^(None|True|False|\[\]|-?[\d.]+(e[+-]?\d+)?|'[^']*'|\"[^\"]*\")$")
SIGNATURE = re.compile(r"^(?:\d+\. )?(\w*)\((.*)\) -> (.*)$")
INTERNAL = {"__module__", "__doc__", "__dict__", "__weakref__", "__qualname__", "__init_subclass__"}


# --- types -------------------------------------------------------------------------------------------

def split_top(text, separator=","):
    """Splits at the separators outside brackets."""
    parts, depth, start = [], 0, 0
    for i, c in enumerate(text):
        depth += c in "([{"
        depth -= c in ")]}"
        if c == separator and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    if text[start:].strip():
        parts.append(text[start:].strip())
    return parts


def class_name(name, module):
    """A class of pybind11's signature as this stub writes it: bare in its own module, `lean.X` from
    the package's top module, `_cora.X` from a submodule."""
    name = name.replace("cora::lean::", f"{ROOT_MODULE}.lean.").replace("cora::", f"{ROOT_MODULE}.")
    if name.startswith(module + ".") and "." not in name[len(module) + 1:]:
        return name[len(module) + 1:]
    if name.startswith(ROOT_MODULE + "."):
        rest = name[len(ROOT_MODULE) + 1:]
        return rest if module == ROOT_MODULE else f"_cora.{rest}"
    return name


HEADS = {"List": "list", "Dict": "dict", "Tuple": "tuple", "Set": "set", "FrozenSet": "frozenset",
         "Sequence": "collections.abc.Sequence", "Iterable": "collections.abc.Iterable",
         "Iterator": "collections.abc.Iterator", "Callable": "collections.abc.Callable"}
SIMPLE = {"SupportsInt": "int", "SupportsFloat": "float", "object": "Any", "list": "list[Any]",
          "tuple": "tuple[Any, ...]", "Callable": "collections.abc.Callable[..., Any]",
          "collections.abc.Callable": "collections.abc.Callable[..., Any]"}


def annotation(text, module):
    """The annotation `text` of a pybind11 signature, as Python writes it: the spellings of older
    pybind11 (`Optional[int]`, `List[int]`) and newer ones (`int | None`, `list[int]`) alike."""
    text = re.sub(r"^typing\.", "", text.strip())
    if re.match(r"^(numpy\.ndarray|numpy\.typing\.NDArray|torch\.Tensor)\b", text):
        return "Array"
    members = split_top(text, "|")
    if len(members) > 1:
        names = [re.sub(r"^typing\.", "", m) for m in members]
        members = [m for m, n in zip(members, names, strict=True) if not (n == "SupportsIndex" and
                                                              {"SupportsInt", "SupportsFloat"} & set(names))]
        shown = [annotation(m, module) for m in members]
        return "Any" if "Any" in shown else " | ".join(dict.fromkeys(shown))
    match = re.match(r"^([\w.:]+)\[(.*)\]$", text)
    if match:
        head, args = match.groups()
        head = re.sub(r"^typing\.", "", head)
        parts = split_top(args)
        if head == "Annotated":
            return annotation(parts[0], module)
        if head == "Optional":
            return annotation(parts[0], module) + " | None"
        if head == "Union":
            return annotation(" | ".join(parts), module)
        shown = [("[" + ", ".join(annotation(a, module) for a in split_top(p[1:-1])) + "]") if p.startswith("[")
                 else annotation(p, module) for p in parts]
        return f"{HEADS.get(head, class_name(head, module))}[{', '.join(shown)}]"
    return SIMPLE.get(text, class_name(text, module))


# --- signatures --------------------------------------------------------------------------------------

def overloads(doc):
    """[(parameters, return type, text)] of a pybind11 docstring, one entry per overload."""
    lines = (doc or "").split("\n")
    found = [(i, SIGNATURE.match(line)) for i, line in enumerate(lines)]
    found = [(i, m) for i, m in found if m]
    out = []
    for k, (i, m) in enumerate(found):
        end = found[k + 1][0] if k + 1 < len(found) else len(lines)
        out.append((m.group(2), m.group(3), "\n".join(lines[i + 1:end]).strip()))
    return out


def argument_type(text, module):
    """The type of a parameter: a list argument takes any sequence."""
    shown = annotation(text, module)
    return "collections.abc.Sequence[" + shown[len("list["):] if shown.startswith("list[") else shown


def parameter(text, module, drop_annotation):
    """One parameter of a signature as `name: type = default`; `self` has no annotation."""
    if text.startswith("*"):
        return text if ":" in text else text + ": Any"
    match = re.match(r"^(\w+)(?:: (.*?))?(?: = (.*))?$", text)
    if not match:
        return text
    name, ann, default = match.groups()
    if drop_annotation:
        return "self"
    out = f"{name}: {argument_type(ann, module) if ann else 'Any'}"
    if default is not None:
        out += " = " + (default if LITERAL.match(default) else "...")
    return out


def docstring(text, pad):
    """A docstring literal of the lines of `text`, indented by `pad`."""
    text = text.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    return pad + '"""' + text.replace("\n", "\n" + pad) + '"""'


def functions(name, doc, module, indent, kind, text=None):
    """The stub of a function, method (kind `method`, `static`, `property`) with all its overloads;
    `text` is the docstring where pybind11 keeps it apart from the signature (a property)."""
    pad = "    " * indent
    found = overloads(doc)
    if text is not None:
        found = [(params, ret, text) for params, ret, _ in found]
    out = []
    for params, ret, text in found:
        parts = split_top(params)
        shown = [parameter(p, module, kind in ("method", "property") and i == 0) for i, p in enumerate(parts)]
        if len(found) > 1:
            out.append(pad + "@overload")
        if kind in ("static", "property"):
            out.append(pad + "@" + ("staticmethod" if kind == "static" else "property"))
        returns = "None" if name == "__init__" else annotation(ret, module)
        out.append(f"{pad}def {name}({', '.join(shown)}) -> {returns}:")
        out.append(docstring(text, pad + "    ") if text else pad + "    ...")
    return out


# --- modules -----------------------------------------------------------------------------------------

def class_stub(cls, module):
    bases = [b.__name__ for b in cls.__bases__ if b.__module__ == module]
    out = [f"class {cls.__name__}({', '.join(bases)}):" if bases else f"class {cls.__name__}:"]
    if cls.__doc__:
        out.append(docstring(inspect.cleandoc(cls.__doc__), "    "))
    for name, member in vars(cls).items():
        if name in INTERNAL or name.startswith("_pybind11"):
            continue
        if isinstance(member, property):
            out += functions(name, member.fget.__doc__, module, 1, "property", member.__doc__ or "")
            if member.fset is not None:
                (params, _, _), = overloads(member.fset.__doc__)
                value = parameter(split_top(params)[1], module, False).replace("arg0", "value", 1)
                out += [f"    @{name}.setter", f"    def {name}(self, {value}) -> None: ..."]
        elif isinstance(member, staticmethod):
            out += functions(name, member.__func__.__doc__, module, 1, "static")
        elif callable(member):
            out += functions(name, member.__doc__, module, 1, "method")
    return out + ([] if len(out) > 1 else ["    ..."])


def submodules(module):
    return sorted(n for n, v in vars(module).items()
                  if isinstance(v, types.ModuleType) and v.__name__.startswith(module.__name__ + "."))


def module_stub(module):
    """The text of the stub of a compiled module."""
    subs = submodules(module)
    imports = "from cora import _cora\n" if module.__name__ != ROOT_MODULE else ""
    imports += "from . import " + ", ".join(subs) + "\n" if subs else ""
    out = [HEADER.format(doc=inspect.cleandoc(module.__doc__ or module.__name__), imports=imports)]
    for name, member in sorted(vars(module).items()):
        if name.startswith("__") or isinstance(member, types.ModuleType):
            continue
        if inspect.isclass(member) and member.__module__ == module.__name__:
            out += ["", ""] + class_stub(member, module.__name__)
        elif callable(member) and not inspect.isclass(member):
            out += [""] + functions(name, member.__doc__, module.__name__, 0, "function")
        elif isinstance(member, (bool, int, float, str)):
            out += ["", f"{name}: {type(member).__name__}"]
    return "\n".join(out) + "\n"


def is_private(name):
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def lacking(module=None, prefix=""):
    """What a user of the stubs misses: the public functions and methods (not private names, not
    the dunder methods, which the class docstring covers) of the compiled modules that have no
    docstring, and the parameters that pybind11 called `arg0`, `arg1`, ... because the binding gave
    no `py::arg`."""
    module = module or importlib.import_module(ROOT_MODULE)
    found = []

    def check(label, doc, text=None, getter=False):
        for params, _, own in overloads(doc):
            shown = own if text is None else text
            if not shown:
                found.append(f"{label}: no docstring")
            if re.search(r"\barg\d+\b", ",".join(split_top(params)[getter:])):
                found.append(f"{label}: unnamed parameter in ({params})")

    for name, member in vars(module).items():
        if is_private(name) or name.startswith("__") or name.startswith("_pybind11"):
            continue
        if isinstance(member, types.ModuleType) and member.__name__.startswith(module.__name__ + "."):
            found += lacking(member, prefix + name + ".")
        elif inspect.isclass(member) and member.__module__ == module.__name__:
            for attribute, item in vars(member).items():
                if is_private(attribute) or attribute in INTERNAL or attribute.startswith("_pybind11"):
                    continue
                dunder = attribute.startswith("__")
                function = item.fget if isinstance(item, property) else getattr(item, "__func__", item)
                if callable(function) and not dunder:
                    label = f"{prefix}{name}.{attribute}"
                    check(label, function.__doc__, item.__doc__ if isinstance(item, property) else None,
                          isinstance(item, property))
        elif callable(member) and not inspect.isclass(member):
            check(prefix + name, member.__doc__)
    return found


def generate():
    """{path in the package: text} of every stub and py.typed."""
    files = {"py.typed": ""}

    def walk(module, path):
        files[path] = module_stub(module)
        for sub in submodules(module):
            walk(getattr(module, sub), os.path.join("_cora", sub + ".pyi"))
    walk(importlib.import_module(ROOT_MODULE), os.path.join("_cora", "__init__.pyi"))
    return files


# ============================================  MAIN  ============================================ #

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", help="the package directory the stubs are written to")
    parser.add_argument("--lacking", action="store_true", help="list the undocumented functions and unnamed parameters")
    parser.add_argument("--check", help="a package directory whose stubs must equal fresh ones")
    args = parser.parse_args()
    if args.lacking:
        print("\n".join(lacking()))
    files = generate()
    for path, text in files.items():
        if args.out:
            target = os.path.join(args.out, path)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                f.write(text)
    if args.check:
        stale = [path for path, text in files.items() if not os.path.exists(os.path.join(args.check, path))
                 or open(os.path.join(args.check, path), encoding="utf-8").read() != text]
        if stale:
            print("stale stubs: " + ", ".join(stale), file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
