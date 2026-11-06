"""test_linear_sys_verify - the verify API in Python: keywords, the algorithm names, VerifyResult

and Falsification (the algorithms themselves are tested in C++).

CORA authors: Tobias Ladner
Authors:       Tobias Ladner
Written:       02-November-2026
Last update:   ---
Reviewed:      ---
"""
# CORA: contDynamics/@linearSys/verify.m

import unittest

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #


# ============================================  MAIN  ============================================ #

class Verify(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("eigen"))
        self.sys = cora.LinearSys(np.array([[-1.0]]), np.array([[1.0]]))
        self.problem = dict(initial_set=cora.Zonotope(np.array([1.0]), np.array([[0.1]])),
                            input_set=cora.Zonotope(np.array([0.0]), np.array([[0.1]])), t_final=1.0)
        self.specs = [cora.Specification.safe_set(np.array([1.0]), 5.0)]

    def test_an_unknown_algorithm_is_refused_with_the_known_names(self):
        with self.assertRaisesRegex(ValueError, "reach_avoid_zonotope"):
            self.sys.verify(**self.problem, specs=self.specs, alg="reach_avoid_unknown")

    def test_an_unknown_keyword_is_refused_with_the_closest_name(self):
        with self.assertRaisesRegex(cora.errors.RedundantNameValuePair, "t_final"):
            self.sys.verify(**self.problem, specs=self.specs, t_finl=2.0)

    def test_verify_gives_a_result(self):
        for alg in ("reach_avoid_support_function", "reach_avoid_zonotope"):
            res = self.sys.verify(**self.problem, specs=self.specs, alg=alg)
            self.assertIsInstance(res, cora.VerifyResult)
            self.assertIsInstance(res.verified, bool)
            self.assertTrue(res.falsification is None or isinstance(res.falsification, cora.Falsification))

    def test_a_missing_specification_list_is_described(self):
        # verify needs the specifications
        with self.assertRaisesRegex(ValueError, "specs"):
            self.sys.verify(**self.problem)


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
