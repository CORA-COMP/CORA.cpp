"""setup_vscode - writes the machine-specific VS Code settings of this checkout

Syntax:   scripts/with_env.sh python scripts/setup_vscode.py     (scripts/setup_local.sh runs it)
          Windows: .venv\Scripts\python scripts\setup_vscode.py  (the .venv of the msvc-cuda preset)
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


WINDOWS = os.name == "nt"
if WINDOWS:
    # the build folder of the msvc-cuda preset, and the venv Python that preset created
    build = os.environ.get("CORACPP_BUILD", os.path.join(ROOT, "build", "python-torch-cuda"))
    python = sys.executable
    libraries = ""
    try:
        import torch
        libraries = os.path.join(os.path.dirname(torch.__file__), "lib")
    except ImportError:
        print("[setup_vscode] no torch in this Python: run the msvc-cuda preset first")
else:
    python = os.path.join(prefix, "bin", "python")
settings = load(os.path.join(VSCODE, "settings.json"))
settings.update({
    "coracpp.envPrefix": prefix,
    "coracpp.buildDir": build,
    "coracpp.python": python,
    "python.defaultInterpreterPath": python,
    "python.analysis.extraPaths": [build],
    "python.testing.unittestEnabled": True,
    "python.testing.unittestArgs": ["-v", "-s", "tests/python", "-p", "test_*.py"],
})
if WINDOWS:
    settings.update({
        "coracpp.libraryPath": libraries,
        "terminal.integrated.env.windows": {
            "PYTHONPATH": build, "CORACPP_BUILD": build, "PATH": libraries + ";${env:PATH}"},
        "python-envs.defaultEnvManager": "ms-python.python:venv",
        "python-envs.defaultPackageManager": "ms-python.python:pip",
        "cmake.useCMakePresets": "always",
    })
else:
    settings.update({
        "terminal.integrated.env.linux": {
            "PYTHONPATH": build, "LD_LIBRARY_PATH": os.path.join(prefix, "lib")},
        "C_Cpp.clang_format_path": os.path.join(prefix, "bin", "clang-format"),
    })
write("settings.json", settings)

if WINDOWS:
    include_root = []
else:
    include_root = [os.path.join(prefix, "include", "eigen3"), os.path.join(prefix, "include")]
include = ["${workspaceFolder}/src", "${workspaceFolder}/tests"] + include_root
defines = []
try:
    import torch
    torch_dir = os.path.dirname(torch.__file__)
    include += [os.path.join(torch_dir, "include"),
                os.path.join(torch_dir, "include", "torch", "csrc", "api", "include")]
    defines.append("CORACPP_TORCH")
except ImportError:
    print("[setup_vscode] no torch in this environment: IntelliSense without libtorch")
if WINDOWS:
    # Eigen is the one the build found or fetched (its CMake config folder is in the cache)
    try:
        with open(os.path.join(build, "CMakeCache.txt"), encoding="utf-8") as f:
            cached = [line.split("=", 1)[1].strip() for line in f if line.startswith("Eigen3_DIR:")]
    except OSError:
        cached = []
    for folder in cached:
        fetched = os.path.join(os.path.dirname(folder), "eigen3-src")
        installed = os.path.normpath(os.path.join(folder, "..", "..", "..", "include", "eigen3"))
        include += [path for path in (fetched, installed) if os.path.isdir(path)]
    compiler, mode = "cl.exe", "windows-msvc-x64"
else:
    compiler = os.path.join(prefix, "bin", "x86_64-conda-linux-gnu-c++")
    compiler = compiler if os.path.exists(compiler) else shutil.which("g++") or "g++"
    mode = "linux-gcc-x64"
write("c_cpp_properties.json", {
    "version": 4,
    "configurations": [{
        "name": "coracpp",
        "compilerPath": compiler,
        "cppStandard": "c++20",
        "intelliSenseMode": mode,
        "defines": defines,
        "includePath": include,
    }],
})
