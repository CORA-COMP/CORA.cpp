"""setup_vscode - writes the machine-specific VS Code settings of this checkout

Syntax:   scripts/with_env.sh python scripts/setup_vscode.py     (scripts/setup_local.sh runs it)
Writes:   .vscode/settings.json          the paths tasks.json and launch.json read (coracpp.*), the
                                         interpreter, and the terminal environment
          .vscode/c_cpp_properties.json  IntelliSense: include paths of Eigen and libtorch
Both are git-ignored: they hold this machine's paths. tasks.json and launch.json are shared.
"""
import json
import os
import shutil
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
VSCODE = os.path.join(ROOT, ".vscode")
prefix = sys.prefix
build = os.environ.get("CORACPP_BUILD", os.path.join(ROOT, "build"))


def load(path):
    """The JSON of `path`, or {} if there is none; a file that is not plain JSON is kept as .bak."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except ValueError:
        shutil.move(path, path + ".bak")
        print(f"[setup_vscode] {os.path.basename(path)} is not plain JSON: kept as {path}.bak")
        return {}


def write(name, data):
    os.makedirs(VSCODE, exist_ok=True)
    with open(os.path.join(VSCODE, name), "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=4)
        f.write("\n")
    print(f"[setup_vscode] wrote .vscode/{name}")


env = {"PYTHONPATH": build, "LD_LIBRARY_PATH": os.path.join(prefix, "lib")}
settings = load(os.path.join(VSCODE, "settings.json"))
settings.update({
    "coracpp.envPrefix": prefix,
    "coracpp.buildDir": build,
    "python.defaultInterpreterPath": os.path.join(prefix, "bin", "python"),
    "python.analysis.extraPaths": [build],
    "python.testing.unittestEnabled": True,
    "python.testing.unittestArgs": ["-v", "-s", "tests/python", "-p", "test_*.py"],
    "terminal.integrated.env.linux": env,
    "C_Cpp.clang_format_path": os.path.join(prefix, "bin", "clang-format"),
})
write("settings.json", settings)

include = ["${workspaceFolder}/src", "${workspaceFolder}/tests",
           os.path.join(prefix, "include", "eigen3"), os.path.join(prefix, "include")]
defines = []
try:
    import torch
    torch_dir = os.path.dirname(torch.__file__)
    include += [os.path.join(torch_dir, "include"),
                os.path.join(torch_dir, "include", "torch", "csrc", "api", "include")]
    defines.append("CORACPP_TORCH")
except ImportError:
    print("[setup_vscode] no torch in this environment: IntelliSense without libtorch")
compiler = os.path.join(prefix, "bin", "x86_64-conda-linux-gnu-c++")
write("c_cpp_properties.json", {
    "version": 4,
    "configurations": [{
        "name": "coracpp",
        "compilerPath": compiler if os.path.exists(compiler) else shutil.which("g++") or "g++",
        "cppStandard": "c++20",
        "intelliSenseMode": "linux-gcc-x64",
        "defines": defines,
        "includePath": include,
    }],
})
