"""test_stubs - the type stubs of the compiled module: shipped, fresh, documented, and they type-check"""
import ast
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest

import cora

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
PACKAGE = os.path.dirname(os.path.abspath(cora.__file__))

spec = importlib.util.spec_from_file_location("stubs", os.path.join(ROOT, "scripts", "stubs.py"))
stubs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stubs)


def mypy(folder, *files):
    """The errors of `mypy --strict` on the files and the stubs of the package. A missing mypy fails
    in CI, which installs it (scripts/requirements-python.txt), and skips elsewhere."""
    try:
        import mypy  # noqa: F401
    except ImportError:
        if os.environ.get("CI"):
            raise AssertionError("mypy is not installed: the CI python job must install it") from None
        raise unittest.SkipTest("mypy is not installed (pip install mypy)") from None
    env = dict(os.environ, MYPYPATH=os.path.dirname(PACKAGE))
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", "--allow-untyped-calls", "--follow-imports=silent",
         "--ignore-missing-imports", "--cache-dir", os.path.join(folder, "cache"), *files]
        + [os.path.join(PACKAGE, "_cora", name) for name in ("__init__.pyi", "lean.pyi")],
        capture_output=True, text=True, env=env)
    return [line for line in result.stdout.splitlines() if ": error:" in line]

# one file that must fail the check with exactly these messages: the stubs say what a call takes
WRONG = """import cora
Z = cora.Zonotope(cora.Tensor([0.0]), cora.Tensor([[1.0]]))
Z.reduce("two")
Z.nonexistent()
x: int = Z.center
"""


def names_defined(path):
    """The classes, functions and assignments at the top level of a Python or stub file."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


class Stubs(unittest.TestCase):
    def test_the_package_ships_stubs_and_py_typed(self):
        for name in ("py.typed", os.path.join("_cora", "__init__.pyi"), os.path.join("_cora", "lean.pyi")):
            self.assertTrue(os.path.exists(os.path.join(PACKAGE, name)), f"{name} is not in {PACKAGE}")

    def test_the_shipped_stubs_are_the_fresh_ones(self):
        for path, text in stubs.generate().items():
            with open(os.path.join(PACKAGE, path), encoding="utf-8") as f:
                self.assertEqual(f.read(), text, f"{path} is stale: rebuild the target `python`")

    def test_every_bound_function_has_named_arguments_and_a_docstring(self):
        self.assertEqual(stubs.lacking(), [])

    def test_every_public_name_has_a_definition(self):
        compiled = names_defined(os.path.join(PACKAGE, "_cora", "__init__.pyi"))
        written = set()
        for name in os.listdir(PACKAGE):
            if name.endswith(".py"):
                written |= names_defined(os.path.join(PACKAGE, name))
        for name in cora.__all__:
            module = [os.path.join(PACKAGE, name + ".py"), os.path.join(PACKAGE, "_cora", name + ".pyi")]
            self.assertTrue(name in compiled or name in written or any(map(os.path.exists, module)),
                            f"cora.{name} has no class or function in the stubs or the package")

    def test_the_stubs_are_valid_python(self):
        for path in stubs.generate():
            if path.endswith(".pyi"):
                with open(os.path.join(PACKAGE, path), encoding="utf-8") as f:
                    ast.parse(f.read(), path)

    def test_mypy_accepts_the_example_and_refuses_wrong_calls(self):
        example = os.path.join(ROOT, "examples", "python", "example_linear_reach_09_prototype.py")
        with tempfile.TemporaryDirectory() as folder:
            wrong = os.path.join(folder, "wrong_calls.py")
            with open(wrong, "w", encoding="utf-8") as f:
                f.write(WRONG)
            errors = mypy(folder, example, wrong)
        self.assertEqual([line.split(":")[0] for line in errors], [wrong] * 3, "\n".join(errors))
        for code in ("arg-type", "attr-defined", "assignment"):
            self.assertEqual(sum(f"[{code}]" in line for line in errors), 1, "\n".join(errors))

    def test_every_set_class_accepts_what_contset_accepts(self):
        # A method of a ContSet subclass takes every argument of ContSet's (mypy's override check),
        # so code written against ContSet works on each class.
        with tempfile.TemporaryDirectory() as folder:
            errors = mypy(folder)
        self.assertEqual(errors, [], "\n".join(errors))


if __name__ == "__main__":
    unittest.main()
