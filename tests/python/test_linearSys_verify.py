"""test_linearSys_verify - the verify API in Python: VerifyParams, VerifyAlg, VerifyResult,
Falsification and LinearSys.verify (the algorithms themselves are tested in C++)."""
import unittest

import numpy as np

import cora


class Verify(unittest.TestCase):
    def setUp(self):
        self.addCleanup(cora.setBackend, cora.backend())
        cora.setBackend("eigen")
        self.sys = cora.LinearSys(np.array([[-1.0]]), np.array([[1.0]]))
        self.params = cora.VerifyParams(cora.Zonotope(np.array([1.0]), np.array([[0.1]])),
                                        cora.Zonotope(np.array([0.0]), np.array([[0.1]])), 1.0)
        self.specs = [cora.Specification.safeSet(np.array([1.0]), 5.0)]

    def test_the_parameters_are_read_and_written(self):
        self.assertEqual(self.params.tFinal, 1.0)
        self.params.tFinal = 2.0
        self.assertEqual(self.params.tFinal, 2.0)
        np.testing.assert_array_equal(self.params.R0.c, [1.0])
        self.params.U = cora.Zonotope(np.array([1.0]), np.array([[0.0]]))
        np.testing.assert_array_equal(self.params.U.c, [1.0])

    def test_the_algorithms_are_named_as_in_cora(self):
        self.assertNotEqual(cora.VerifyAlg.SupportFunc, cora.VerifyAlg.Zonotope)

    def test_an_unknown_algorithm_is_refused(self):
        with self.assertRaisesRegex(ValueError, "reachavoid:zonotope"):
            self.sys.verify(self.params, "reachavoid:unknown", self.specs)

    def test_verify_takes_an_enum_or_its_name_and_gives_a_result(self):
        for alg in (cora.VerifyAlg.SupportFunc, "reachavoid:supportFunc"):
            res = self.sys.verify(self.params, alg, self.specs)
            self.assertIsInstance(res, cora.VerifyResult)
            self.assertIsInstance(res.verified, bool)
            self.assertTrue(res.fals is None or isinstance(res.fals, cora.Falsification))

    def test_the_zonotope_algorithm_is_not_there_yet(self):
        with self.assertRaisesRegex(RuntimeError, "not implemented"):
            self.sys.verify(self.params, cora.VerifyAlg.Zonotope, self.specs)


if __name__ == "__main__":
    unittest.main()
