"""make_dev_set - a development instance set of a VNN-COMP benchmark, from our own seed

VNN-COMP draws its instances at random. For development and tuning we draw our own: the
benchmark's own generator (generate_properties.py of the source repository and commit that the
2026 benchmark repository names) runs with our seed, in a copy of the source tree. Where the
generator has no randomness (acasxu) or is not public (dist_shift), the instances are drawn from
the official ones' structure: the same networks and output properties, random input boxes.
The set is written in VNN-COMP's layout, VNN-LIB 1.0 and its 2.0 form, with a manifest (seed,
source commit, the instances identical to official ones, which are left out unless
--keep-official) and, where the generator knows it, the expected verdicts. The same seed gives
the same instances.

Syntax:   python scripts/vnncomp/make_dev_set.py <benchmark> --seed S --out DIR
Inputs:   --sources, --official, --python, --cache: clones, benchmarks, Python, datasets
Outputs:  DIR/<benchmark>/{1.0,2.0}/{instances.csv,onnx/,vnnlib/}, manifest.json, expected.csv
See also: scripts/vnncomp/run_dev_set.py, docs/nn_verification.md

Authors:       Claude Opus 5.5
Written:       21-November-2026
Last update:   23-November-2026
Reviewed:      ---
"""

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("benchmark", choices=sorted(SOURCES))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sources", default=os.path.join(HOME, "vnncomp", "sources"))
    parser.add_argument(
        "--official", default=os.path.join(HOME, "vnncomp", "vnncomp2026_benchmarks")
    )
    parser.add_argument(
        "--python", default=os.path.join(HOME, "vnncomp", "genvenv", "bin", "python")
    )
    parser.add_argument("--cache", default=os.path.join(HOME, "vnncomp", "cache"))
    parser.add_argument("--keep-official", action="store_true")
    args = parser.parse_args()
    work = os.path.join(args.out, args.benchmark, "work")
    # the benchmark's own generator, else a sampler of the official instances' structure
    if args.benchmark == "acasxu_2023":
        args.method = (
            "sampled: random sub-boxes of the official properties (the generator is deterministic)"
        )
        os.makedirs(work, exist_ok=True)
        rows = sampled_acasxu(args, work)
    elif args.benchmark == "dist_shift_2023":
        args.method = "sampled: random MNIST test images (the generator is not public)"
        os.makedirs(work, exist_ok=True)
        rows = sampled_dist_shift(args, work)
    else:
        args.method = "generator: generate_properties.py with the seed"
        run_generator(args, work)
        rows = generated_rows(args, work)
    manifest = write_set(args, work, rows)
    shutil.rmtree(work)
    print(json.dumps(manifest))
    return 0


# -----------------------------------------  AUXILIARY  ------------------------------------------ #

HOME = os.path.expanduser("~")

# the source repositories and commits of the 2026 benchmark repository's READMEs
SOURCES = {
    "acasxu_2023": (
        "https://github.com/kkaulen/acasxu_2023.git",
        "fb1e1899079f2d25a270c5523798e9fa1fca7750",
    ),
    "cersyve": (
        "https://github.com/intelligent-control-lab/Cersyve.jl.git",
        "d1a8e46e7de59c77ab1cd6852f853eeef7e3ce8a",
    ),
    # cgan's generator downloads its networks with gdown
    "cgan2026": (
        "https://github.com/stanleybak/cgan_benchmark2026",
        "8d2cecd10ab95c0e15524c4b81571cbd6113fee7",
    ),
    "cifar100_2024": (
        "https://github.com/huanzhang12/vnncomp2024_cifar100_benchmark.git",
        "2f7b2d202ed58f260c61500b70883c7273030462",
    ),
    "collins_rul_cnn_2022": (
        "https://github.com/loonwerks/vnncomp2022.git",
        "4422a5f1a472eb7947970a2f0404e5a9e80937a3",
    ),
    "cora_2024": (
        "https://github.com/kollerlukas/cora-vnncomp2024-benchmark.git",
        "d8d9e2e3d0712f18983ec02bd63514660212c648",
    ),
    # dist_shift's repository is not public (sampled_dist_shift)
    "dist_shift_2023": (
        "https://github.com/toladnertum/dist-shift-vnn-comp.git",
        "5946000bd3e1ba89963e2d3ae341d17b89c8659b",
    ),
    # linearizenn's generator reads its models with h5py
    "linearizenn_2024": (
        "https://github.com/aliabigdeli/LinearizeNN_benchmark2024.git",
        "d695230914a0894da859c31d9f8b0dff141406e3",
    ),
    # lsnc_relu: a Lyapunov ring c1 < V(x) < c2 of a quadrotor's neural controller and dynamics
    "lsnc_relu": (
        "https://github.com/xiangruzh/LSNC-ReLU.git",
        "3f1cb43f393cb2ef160815cf2ae1e87675499db7",
    ),
    "malbeware": (
        "https://github.com/pkrobinette/MalBeWare.git",
        "fc044fe46f4a6ed7ad7f22341e59bbf99e616c5f",
    ),
    # metaroom: the commit of the 2026 README, not the repository's head
    "metaroom_2023": (
        "https://github.com/HanjiangHu/metaroom_vnn_comp2023",
        "80233c82c13ae795d0de7a9f87875dcca13e61ff",
    ),
    # the text benchmark, the SAT encodings and the hidden counterexamples
    "safenlp_2024": (
        "https://github.com/ANTONIONLP/safeNLP.git",
        "a88c5784bf71a9aa3689ae2bd409f356e8af163b",
    ),
    "sat_relu": ("https://github.com/emanino/neurocodebench", "ae1a535"),
    "soundnessbench_2026": (
        "https://github.com/Kevinyang555/SoundnessBench_vnncomp2026",
        "3ac608157866bc7409e37d89fcc564757037bc40",
    ),
    "tinyimagenet_2024": (
        "https://github.com/huanzhang12/vnncomp2024_tinyimagenet_benchmark.git",
        "771377743f16ee30ecb00029268ba62ec7e9e78c",
    ),
}


def sha256(path):
    """The hash of a file's content (gunzipped)."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def read_rows(path):
    """The rows (onnx, vnnlib, timeout) of an instances.csv, paths without a leading ./"""
    with open(path, newline="") as stream:
        return [
            (r[0].strip().removeprefix("./"), r[1].strip().removeprefix("./"), r[2].strip())
            for r in csv.reader(stream)
            if r
        ]


def run_generator(args, work):
    """Copy the source tree to work and run its generate_properties.py with the seed there."""
    source = os.path.join(args.sources, args.benchmark)
    if not os.path.isdir(source):
        raise SystemExit(
            f"make_dev_set: clone {SOURCES[args.benchmark][0]} at "
            f"{SOURCES[args.benchmark][1]} to {source}"
        )
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(source, work, ignore=shutil.ignore_patterns(".git"))
    # the generator writes its own vnnlib/; datasets go to one shared cache
    if args.benchmark == "cora_2024":
        shutil.rmtree(os.path.join(work, "vnnlib"), ignore_errors=True)
        os.makedirs(os.path.join(work, "vnnlib"))
        os.makedirs(os.path.join(args.cache, "datasets"), exist_ok=True)
        os.symlink(os.path.join(args.cache, "datasets"), os.path.join(work, "tmp"))
    env = dict(os.environ, PYTHONHASHSEED="0", OMP_NUM_THREADS="2")
    subprocess.run(
        [args.python, "generate_properties.py", str(args.seed)], cwd=work, env=env, check=True
    )


def generated_rows(args, work):
    """The instances the generator wrote, as (onnx path, vnnlib path, timeout) in work."""
    # the generators that write bare file names: the networks' and properties' directories
    folders = {
        "collins_rul_cnn_2022": "vnnlib",  # the seed changes the files' content, not their names
        "cifar100_2024": "generated_vnnlib",
        "tinyimagenet_2024": "generated_vnnlib",
    }
    if args.benchmark in folders:
        return [
            (f"onnx/{o}", f"{folders[args.benchmark]}/{v}", t)
            for o, v, t in read_rows(os.path.join(work, "instances.csv"))
        ]
    return read_rows(os.path.join(work, "instances.csv"))


def sampled_acasxu(args, work):
    """acasxu: the generator is deterministic, so every official (network, property) gets a random
    sub-box of the property's input box: per dimension a width of U(0.2, 1) of the original and a
    uniform position (the property 6 cases each get one)."""
    import cora.vnnlib

    rng = np.random.default_rng(args.seed)
    official = os.path.join(args.official, "benchmarks", args.benchmark, "1.0")
    os.makedirs(os.path.join(work, "vnnlib"), exist_ok=True)
    rows = []
    for k, (onnx, prop, timeout) in enumerate(read_rows(os.path.join(official, "instances.csv"))):
        text = cora.vnnlib.read_text(os.path.join(official, prop + ".gz"))
        name = f"vnnlib/{os.path.basename(prop)[:-7]}_dev{k}.vnnlib"
        with open(os.path.join(work, name), "w") as out:
            out.write(_resample_boxes(text, rng))
        rows.append((onnx, name, timeout))
        _copy_official(official, onnx, work)
    return rows


def _resample_boxes(text, rng):
    """A 1.0 text whose input bounds (<= X_i u) and (>= X_i l) are replaced by a random sub-box."""
    import re

    pairs = {}
    for op, i, v in re.findall(r"\((<=|>=) X_(\d+) (\S+?)\)", text):
        pairs.setdefault(i, {})[op] = float(v)
    boxes = {}
    for i, b in pairs.items():
        lo, hi = b[">="], b["<="]
        width = (hi - lo) * rng.uniform(0.2, 1.0)
        start = lo + rng.uniform(0.0, 1.0) * (hi - lo - width)
        boxes[i] = (round(start, 9), round(min(hi, start + width), 9))

    # every occurrence of a dimension gets the same new bounds (property 6 has two boxes)
    def replace(m):
        op, i = m.group(1), m.group(2)
        return f"({op} X_{i} {boxes[i][0] if op == '>=' else boxes[i][1]})"

    return re.sub(r"\((<=|>=) X_(\d+) (\S+?)\)", replace, text)


def sampled_dist_shift(args, work):
    """dist_shift, whose generator is not public: 72 random MNIST test images that the classifier
    labels correctly, each with the prior's latent box (reconstructed from the official files)."""
    import onnxruntime as ort

    # an image (784 fixed inputs), latent mean +- 0.13 std (8), unsafe: runner-up >= top class

    official = os.path.join(args.official, "benchmarks", args.benchmark, "1.0")
    prior = ort.InferenceSession(_onnx_bytes(os.path.join(official, "onnx/mnist_prior.onnx")))
    classifier = ort.InferenceSession(_onnx_bytes(os.path.join(official, "onnx/mnist_concat.onnx")))
    images, labels = _mnist_test(args)
    rng = np.random.default_rng(args.seed)
    os.makedirs(os.path.join(work, "vnnlib"), exist_ok=True)
    rows = []
    for idx in rng.permutation(len(images)):
        img = images[idx].reshape(1, 1, 28, 28)
        mu, logvar = prior.run(None, {"input": img})
        x = np.concatenate([img.reshape(-1), mu[0]]).reshape(1, 792).astype(np.float32)
        y = classifier.run(None, {classifier.get_inputs()[0].name: x})[0][0]
        top, second = np.argsort(-y)[:2]
        if top != labels[idx]:
            continue
        # the box of the latent inputs, the property of the two classes
        std = np.exp(0.5 * logvar[0].astype(np.float64))
        lo = np.concatenate([img.reshape(-1).astype(np.float64), mu[0] - 0.13 * std])
        hi = np.concatenate([img.reshape(-1).astype(np.float64), mu[0] + 0.13 * std])
        name = f"vnnlib/index{idx}_delta0.13.vnnlib"
        _write_box_property(
            os.path.join(work, name), lo, hi, 10, f"(assert (>= Y_{second} Y_{top}))"
        )
        rows.append(("onnx/mnist_concat.onnx", name, "300"))
        if len(rows) == 72:
            break
    _copy_official(official, "onnx/mnist_concat.onnx", work)
    return rows


def _mnist_test(args):
    """The MNIST test images in [0, 1] (float32, flat) and labels, via torchvision in the cache."""
    out = os.path.join(args.cache, "mnist_test.npz")
    if not os.path.exists(out):
        code = (
            "import numpy as np, torchvision, sys; d = torchvision.datasets.MNIST(sys.argv[1], "
            "train=False, download=True); np.savez(sys.argv[2], x=d.data.numpy(), y=d.targets.numpy())"
        )
        subprocess.run(
            [args.python, "-c", code, os.path.join(args.cache, "datasets"), out], check=True
        )
    data = np.load(out)
    return (data["x"].reshape(-1, 784) / np.float32(255)).astype(np.float32), data["y"]


def _write_box_property(path, lo, hi, outputs, assertion):
    """A 1.0 property of an input box and one output assertion."""
    with open(path, "w") as out:
        out.writelines(f"(declare-const X_{i} Real)\n" for i in range(len(lo)))
        out.writelines(f"(declare-const Y_{j} Real)\n" for j in range(outputs))
        for i, (a, b) in enumerate(zip(lo, hi, strict=True)):
            out.write(
                f"(assert (<= X_{i} {repr(float(b))}))\n(assert (>= X_{i} {repr(float(a))}))\n"
            )
        out.write(assertion + "\n")


def _onnx_bytes(path):
    """The model of path or path.gz."""
    if os.path.exists(path):
        with open(path, "rb") as stream:
            return stream.read()
    with gzip.open(path + ".gz", "rb") as stream:
        return stream.read()


def _copy_official(official, onnx, work):
    """The official network into work (unzipped)."""
    target = os.path.join(work, onnx)
    if not os.path.exists(target):
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as out:
            out.write(_onnx_bytes(os.path.join(official, onnx)))


def write_set(args, work, rows):
    """The set in VNN-COMP's layout, 1.0 and 2.0, the manifest and the expected verdicts."""
    import cora.onnx_network
    import cora.vnnlib

    base = os.path.join(args.out, args.benchmark)
    official = os.path.join(args.official, "benchmarks", args.benchmark, "1.0")
    official_props = set()
    if os.path.isdir(official):
        for onnx, prop, _ in read_rows(os.path.join(official, "instances.csv")):
            net = os.path.join(official, onnx)
            net = net if os.path.exists(net) else net + ".gz"
            if not os.path.exists(net):  # a network the checkout left out matches nothing
                continue
            official_props.add((sha256(net), sha256(os.path.join(official, prop + ".gz"))))
    shapes, overlap, expected, kept = {}, 0, [], []
    for version in ("1.0", "2.0"):
        shutil.rmtree(os.path.join(base, version), ignore_errors=True)
    for onnx, prop, timeout in rows:
        if onnx not in shapes:
            net = cora.onnx_network.OnnxNetwork.load(os.path.join(work, onnx))
            shapes[onnx] = (net.input_shape, net.output_shape)
        # the property in 1.0 and 2.0, the network beside it
        text = cora.vnnlib.read_text(os.path.join(work, prop))
        key = (sha256(os.path.join(work, onnx)), hashlib.sha256(text.encode()).hexdigest())
        if key in official_props:
            overlap += 1
            if not args.keep_official:  # an official instance is no dev instance
                continue
        kept.append((onnx, prop, timeout))
        for version, body in (("1.0", text), ("2.0", cora.vnnlib.to_vnnlib2(text, *shapes[onnx]))):
            for rel, data in ((onnx, None), (prop, body)):
                target = os.path.join(base, version, rel)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                if data is None and not os.path.exists(target):
                    shutil.copyfile(os.path.join(work, rel), target)
                elif data is not None:
                    with open(target, "w") as out:
                        out.write(data)
        if args.benchmark == "sat_relu":  # the file names carry the ground truth
            expected.append(
                (onnx, prop, "sat" if os.path.basename(onnx).startswith("sat_") else "unsat")
            )
    # instances.csv per version, the manifest, the expected verdicts
    for version in ("1.0", "2.0"):
        with open(os.path.join(base, version, "instances.csv"), "w", newline="") as out:
            csv.writer(out, lineterminator="\n").writerows(kept)
    if expected:
        with open(os.path.join(base, "expected.csv"), "w", newline="") as out:
            csv.writer(out, lineterminator="\n").writerows(expected)
    # the manifest: the seed, the source and a hash of the instances
    manifest = {
        "benchmark": args.benchmark,
        "seed": args.seed,
        "source": SOURCES[args.benchmark],
        "method": args.method,
        "instances": len(kept),
        "identical_to_official": overlap,
        "official_kept": bool(args.keep_official),
        "python": args.python,
        "instances_sha256": sha256(os.path.join(base, "1.0", "instances.csv")),
    }
    with open(os.path.join(base, "manifest.json"), "w") as out:
        json.dump(manifest, out, indent=2)
    return manifest


if __name__ == "__main__":
    sys.exit(main())

# ----------------------------------------  END OF CODE  ----------------------------------------- #
