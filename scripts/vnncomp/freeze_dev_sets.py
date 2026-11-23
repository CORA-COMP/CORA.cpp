"""freeze_dev_sets - writes and verifies the MANIFEST.json of a release of the VNN-COMP dev sets

A release is DIR/s<seed>/<benchmark>/ as make_dev_set writes them, each with its manifest.json.
MANIFEST.json records per benchmark and seed the instances, those identical to an official one
(left out), the generator's source and commit, the method (generator, sampled or sub-boxes), the
known nondeterminism and the sha256 of every onnx, vnnlib and csv file. A subset (safenlp_2024_sub)
is listed with its rule and its relative links. Writing and verifying both fail on an absolute,
dangling or escaping symlink, a work/ folder, a log or any other stray file; --verify also fails
on a file whose hash, or a set of files, differs from the manifest.

Syntax:   python scripts/vnncomp/freeze_dev_sets.py DIR (--write --release v1 | --verify)
Inputs:   --jobs N hashes in N processes
See also: scripts/vnncomp/make_dev_set.py, docs/nn_verification.md

Authors:       Claude Opus 5.5
Written:       23-November-2026
Last update:   ---
Reviewed:      ---
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dir")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--release", default="v1")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    # the tree first: nothing is hashed while it has a problem
    if args.write == args.verify:
        raise SystemExit("freeze_dev_sets: give one of --write and --verify")
    root = os.path.realpath(args.dir)
    problems = check_tree(root)
    if problems:
        print("\n".join(problems))
        raise SystemExit(f"freeze_dev_sets: {len(problems)} problems in {root}")
    manifest = build_manifest(root, args.release, args.jobs)
    path = os.path.join(root, "MANIFEST.json")
    if args.write:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=1, sort_keys=True)
            f.write("\n")
        print(f"freeze_dev_sets: wrote {path}: {summary(manifest)}")
        return 0
    # verify: the same benchmarks, files and hashes as written
    with open(path, encoding="utf-8") as f:
        written = json.load(f)
    differences = compare(written, manifest)
    if differences:
        print("\n".join(differences))
        raise SystemExit(f"freeze_dev_sets: {len(differences)} differences to {path}")
    print(f"freeze_dev_sets: {path} holds: {summary(manifest)}")
    return 0


# -----------------------------------------  AUXILIARY  ------------------------------------------ #

# The release's rules -----------------------------------------------------------------------------

# the subsets: the benchmark they take from and the rule (rows of 2.0/instances.csv)
SUBSETS = {
    "safenlp_2024_sub": ("safenlp_2024", "every fourth instance: rows 1, 5, 9, ... of 2.0/instances.csv"),
}

# the generators known to give other instances for the same seed (these files are the reference)
NONDETERMINISM = {
    "cifar100_2024": "the generator shuffles the test set with Python's random, which it never seeds",
    "tinyimagenet_2024": "the generator shuffles the test set with Python's random, which it never seeds",
}

# the files a benchmark folder may hold
SUFFIXES = (".onnx", ".vnnlib", ".csv")
TOP_FILES = ("MANIFEST.json",)


def check_tree(root):
    """The problems of the tree: absolute, dangling or escaping links, work/ folders, stray files."""
    problems = []
    for folder, dirs, files in os.walk(root):
        rel = os.path.relpath(folder, root)
        # links (to folders or files) must be relative and resolve inside the release
        for name in dirs + files:
            path = os.path.join(folder, name)
            if not os.path.islink(path):
                continue
            target = os.readlink(path)
            resolved = os.path.realpath(path)
            if os.path.isabs(target):
                problems.append(f"absolute link {os.path.relpath(path, root)} -> {target}")
            elif not os.path.exists(resolved):
                problems.append(f"dangling link {os.path.relpath(path, root)} -> {target}")
            elif os.path.commonpath([resolved, root]) != root:
                problems.append(f"link out of the release {os.path.relpath(path, root)} -> {target}")
        if "work" in dirs:
            problems.append(f"scratch folder {os.path.join(rel, 'work')}")
        # the files: MANIFEST.json at the top, manifest.json and the data below
        for name in files:
            depth = 0 if rel == "." else rel.count(os.sep) + 1
            allowed = (depth == 0 and name in TOP_FILES) or (depth == 2 and name == "manifest.json")
            if not allowed and not name.endswith(SUFFIXES):
                problems.append(f"stray file {os.path.join(rel, name)}")
    return problems


# The manifest ------------------------------------------------------------------------------------

def build_manifest(root, release, jobs):
    """MANIFEST.json's content: every seed's benchmarks with their files' hashes."""
    entries, paths = {}, []
    for seed_dir in sorted(d for d in os.listdir(root) if d.startswith("s") and d[1:].isdigit()):
        for bench in sorted(os.listdir(os.path.join(root, seed_dir))):
            entry, files = aux_benchmark(root, seed_dir, bench)
            entries.setdefault(bench, {})[seed_dir[1:]] = entry
            paths += files
    # the hashes of all files at once (a file under a linked folder is hashed where it lies)
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        sums = pool.map(aux_sha256, [os.path.join(root, p) for p in paths], chunksize=16)
        hashes = dict(zip(paths, sums, strict=True))
    for bench in entries.values():
        for entry in bench.values():
            entry["files"] = {p: hashes[os.path.join(entry["folder"], p)] for p in entry["files"]}
    return {
        "release": release,
        "written": datetime.date.today().isoformat(),
        "seeds": sorted({s for bench in entries.values() for s in bench}),
        "benchmarks": entries,
    }


def aux_benchmark(root, seed_dir, bench):
    """One benchmark of one seed: its manifest's facts and its files (relative to the release)."""
    folder = os.path.join(seed_dir, bench)
    absolute = os.path.join(root, folder)
    # a subset's own files only (its links are listed), a benchmark's all
    files = aux_files(absolute, follow=bench not in SUBSETS)
    rows = aux_rows(os.path.join(absolute, "2.0", "instances.csv"))
    # a subset: its rule and its links instead of a generator
    if bench in SUBSETS:
        parent, rule = SUBSETS[bench]
        # the rule holds: every fourth row of the parent's instances
        mine = aux_lines(os.path.join(absolute, "2.0", "instances.csv"))
        theirs = aux_lines(os.path.join(root, seed_dir, parent, "2.0", "instances.csv"))
        if mine != theirs[::4]:
            raise SystemExit(f"freeze_dev_sets: {folder} is not {rule} of {parent}")
        links = {
            os.path.relpath(os.path.join(f, n), absolute): os.readlink(os.path.join(f, n))
            for f, dirs, names in os.walk(absolute)
            for n in dirs + names
            if os.path.islink(os.path.join(f, n))
        }
        entry = {"folder": folder, "method": "subset", "subset_of": parent, "rule": rule,
                 "instances": rows, "links": links, "files": files}
        return entry, [os.path.join(folder, p) for p in files]
    # a generated benchmark: its manifest.json
    with open(os.path.join(absolute, "manifest.json"), encoding="utf-8") as f:
        made = json.load(f)
    text = made["method"]
    kind = "sub-boxes" if "sub-boxes" in text else text.split(":")[0]
    entry = {
        "folder": folder,
        "method": kind,
        "method_detail": text,
        "instances": rows,
        "instances_in_manifest": made.get("instances"),
        "official_identical_left_out": made.get("identical_to_official"),
        "source": made.get("source"),
        "missing_networks": made.get("missing_networks", []),
        "nondeterminism": NONDETERMINISM.get(bench, "none known"),
        "files": files,
    }
    return entry, [os.path.join(folder, p) for p in files]


def aux_files(folder, follow):
    """The data files below the folder (through its links with follow), relative to it, sorted."""
    out = []
    for f, _, names in os.walk(folder, followlinks=follow):
        out += [os.path.relpath(os.path.join(f, n), folder) for n in names if n.endswith(SUFFIXES)]
    return sorted(out)


def aux_rows(path):
    """The instances of an instances.csv (its non-empty rows)."""
    if not os.path.exists(path):
        return 0
    with open(path, newline="", encoding="utf-8") as f:
        return sum(1 for row in csv.reader(f) if row)


def aux_lines(path):
    """The lines of a text file."""
    with open(path, encoding="utf-8") as f:
        return f.read().splitlines()


def aux_sha256(path):
    """The sha256 of a file, read in 4 MiB blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


# Verifying ---------------------------------------------------------------------------------------

def compare(written, now):
    """The differences between the written manifest and the tree now (benchmarks, files, hashes)."""
    out = []
    a, b = written["benchmarks"], now["benchmarks"]
    for bench in sorted(set(a) | set(b)):
        for seed in sorted(set(a.get(bench, {})) | set(b.get(bench, {}))):
            x, y = a.get(bench, {}).get(seed), b.get(bench, {}).get(seed)
            if x is None or y is None:
                out.append(f"{bench} seed {seed}: {'new' if x is None else 'missing'}")
                continue
            # the files of both, and their hashes
            for p in sorted(set(x["files"]) | set(y["files"])):
                if x["files"].get(p) != y["files"].get(p):
                    out.append(f"{bench} seed {seed}: {p} differs")
            for key in ("instances", "method", "links"):
                if x.get(key) != y.get(key):
                    out.append(f"{bench} seed {seed}: {key} {x.get(key)} -> {y.get(key)}")
    return out


def summary(manifest):
    """Benchmarks, seeds, instances and files of a manifest, one line."""
    entries = [e for bench in manifest["benchmarks"].values() for e in bench.values()]
    files = sum(len(e["files"]) for e in entries)
    instances = sum(e["instances"] for e in entries if e["method"] != "subset")
    return (f"{len(manifest['benchmarks'])} benchmarks, seeds {', '.join(manifest['seeds'])}, "
            f"{instances} instances (subsets not counted), {files} files")


if __name__ == "__main__":
    sys.exit(main())

# ----------------------------------------  END OF CODE  ----------------------------------------- #
