"""test_benchmarkData - the ARCH-COMP AFF instance files of benchmarks/data: format and consistency."""
import glob
import json
import os
import struct
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "python"))
import benchmarkData  # noqa: E402

DATA = os.path.join(ROOT, "benchmarks", "data")
INSTANCES = ["ISSC01_ISS02", "ISSC01_ISU02", "ISSF01_ISS01", "ISSF01_ISU01", "CBC01", "CBC02",
             "CBC03", "CBF01", "CBF02", "CBF03", "HEAT01", "HEAT02", "RAND01", "RAND02"]
NAMES = ["A", "B", "C", "R0c", "R0G", "Uc", "UG"]


def write(folder, name, matrices, specs):
    """Writes an instance the way exportBenchmarkData.m does: coordinates if sparse, else dense."""
    entries, blob = {}, b""
    for key, M in matrices.items():
        i, j = np.nonzero(M)
        entry = {"shape": list(M.shape), "offset": len(blob)}
        if len(i) * 16 < M.size * 8:
            entry.update(storage="coo", nnz=len(i))
            blob += i.astype("<i4").tobytes() + j.astype("<i4").tobytes() + M[i, j].astype("<f8").tobytes()
        else:
            entry.update(storage="dense", nnz=M.size)
            blob += M.astype("<f8").tobytes()
        entries[key] = entry
    meta = {"format": "cora-benchmark-1", "benchmark": "Test", "instance": name, "label": name,
            "verifyAlg": "reachavoid:supportFunc", "tFinal": 1.5, "data": name + ".bin",
            "matrices": entries, "specs": specs, "expected": None}
    with open(os.path.join(folder, name + ".bin"), "wb") as f:
        f.write(blob)
    with open(os.path.join(folder, name + ".json"), "w") as f:
        json.dump(meta, f)


class Format(unittest.TestCase):
    def test_dense_and_coordinate_matrices_round_trip(self):
        rng = np.random.default_rng(0)
        sparse = np.zeros((6, 7))
        sparse[[0, 3, 5], [1, 6, 0]] = [0.1, -2.5, 1e-300]
        matrices = {"A": rng.standard_normal((3, 3)), "S": sparse, "E": np.zeros((4, 0))}
        with tempfile.TemporaryDirectory() as folder:
            write(folder, "t", matrices, [])
            loaded = benchmarkData.loadInstance("t", folder)["matrices"]
        for key, M in matrices.items():
            np.testing.assert_array_equal(loaded[key], M)

    def test_unknown_format_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            write(folder, "t", {"A": np.eye(2)}, [])
            path = os.path.join(folder, "t.json")
            with open(path) as f:
                meta = json.load(f)
            meta["format"] = "other"
            with open(path, "w") as f:
                json.dump(meta, f)
            with self.assertRaises(ValueError):
                benchmarkData.loadInstance("t", folder)

    def test_doubles_are_stored_exactly(self):
        value = struct.unpack("<d", struct.pack("<Q", 0x3FB999999999999A))[0]
        with tempfile.TemporaryDirectory() as folder:
            write(folder, "t", {"A": np.array([[value]])}, [])
            loaded = benchmarkData.loadInstance("t", folder)["matrices"]["A"]
        self.assertEqual(loaded[0, 0].hex(), value.hex())


class Instances(unittest.TestCase):
    def test_every_instance_is_there(self):
        found = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(DATA, "*.json")))
        self.assertTrue(set(found) <= set(INSTANCES), found)
        if not found:
            self.skipTest("benchmarks/data is empty: run benchmarks/matlab/exportBenchmarkData.m")

    def test_the_schema_and_the_dimensions_agree(self):
        for path in glob.glob(os.path.join(DATA, "*.json")):
            name = os.path.basename(path)[:-5]
            with self.subTest(name):
                inst = benchmarkData.loadInstance(name)
                mat = inst["matrices"]
                self.assertEqual(sorted(mat), sorted(NAMES))
                n = mat["A"].shape[0]
                self.assertEqual(mat["A"].shape, (n, n))
                self.assertEqual(mat["B"].shape[0], n)
                self.assertEqual(mat["C"].shape[1], n)
                self.assertEqual(mat["R0c"].shape, (n, 1))
                self.assertEqual(mat["R0G"].shape[0], n)
                self.assertEqual(mat["Uc"].shape, (mat["B"].shape[1], 1))
                self.assertEqual(mat["UG"].shape[0], mat["B"].shape[1])
                self.assertGreater(inst["tFinal"], 0)
                self.assertIn(inst["verifyAlg"], ["reachavoid:supportFunc", "reachavoid:zonotope"])
                self.assertEqual(inst["instance"], name)
                self.assertTrue(inst["specs"])
                for spec in inst["specs"]:
                    self.assertIn(spec["type"], ["safeSet", "unsafeSet"])
                    self.assertEqual(np.array(spec["A"]).shape, (len(spec["b"]), mat["C"].shape[0]))
                if inst["expected"]:
                    self.assertIn(inst["expected"]["verified"], [0, 1])


if __name__ == "__main__":
    unittest.main()
