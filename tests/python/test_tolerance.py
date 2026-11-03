"""test_tolerance - cora.tolerance, set_default_tolerance, default_tol and tol=None in Python."""
import threading
import unittest

import numpy as np

import cora


def box(lo, hi, dtype="float64"):
    return cora.Interval(np.array(lo, dtype=dtype), np.array(hi, dtype=dtype))


class ToleranceTest(unittest.TestCase):
    def tearDown(self):
        cora.set_default_tolerance(None)

    def test_default_follows_the_dtype(self):
        self.assertEqual(cora.default_tol(), 100 * np.finfo(np.float64).eps)
        self.assertEqual(cora.default_tol("float32"), 100 * float(np.finfo(np.float32).eps))
        self.assertEqual(cora.default_tol(np.float32), cora.default_tol("float32"))
        self.assertGreater(cora.default_tol("float32"), 1e6 * cora.default_tol("float64"))
        with self.assertRaises(ValueError):
            cora.default_tol("int8")

    def test_order_of_resolution(self):
        I, p = box([0.0], [1.0]), np.array([[1.0 + 1e-10]])
        self.assertFalse(I.contains(p))  # the dtype's default
        self.assertTrue(I.contains(p, tol=1e-9))  # an argument
        self.assertTrue(I.contains(p, 1e-9) and not I.contains(p, tol=None))
        with cora.tolerance(1e-9):
            self.assertEqual(cora.current_tolerance(), 1e-9)
            self.assertTrue(I.contains(p))  # a scope
            self.assertFalse(I.contains(p, tol=0.0))  # an argument beats it
            with cora.tolerance(1e-12):
                self.assertFalse(I.contains(p))  # the innermost scope
            self.assertTrue(I.contains(p))
        self.assertIsNone(cora.current_tolerance())
        cora.set_default_tolerance(1e-9)  # the process default
        self.assertTrue(I.contains(p))
        with cora.tolerance(1e-12):
            self.assertFalse(I.contains(p))  # a scope beats it
        cora.set_default_tolerance(None)
        self.assertFalse(I.contains(p))

    def test_float32_has_its_own_default(self):
        I = box([0.0], [1.0], "float32")
        self.assertTrue(I.contains(np.array([[1.0 + 1e-6]], dtype="float32")))
        self.assertFalse(box([0.0], [1.0]).contains(np.array([[1.0 + 1e-6]])))

    def test_functions_and_methods_take_it(self):
        I, J = box([0.0, 0.0], [1.0, 1.0]), box([1.0 + 1e-10, 0.0], [2.0, 1.0])
        self.assertFalse(cora.is_intersecting(I, J) or I.is_intersecting(J))
        self.assertTrue(cora.is_intersecting(I, J, tol=1e-9) and I.is_intersecting(J, tol=1e-9))
        K = box([0.0, 0.0], [1.0 + 1e-10, 1.0])
        self.assertFalse(I.isequal(K))
        self.assertTrue(I.isequal(K, tol=1e-9))
        Z = cora.Zonotope(np.zeros(2), np.array([[1.0, 0.2, 1e-12], [0.5, 1.0, 0.0]]))
        self.assertEqual(Z.compact().generators.shape[-1], 3)
        with cora.tolerance(1e-10):
            self.assertEqual(Z.compact().generators.shape[-1], 2)
            with cora.tolerance(1e-8):
                self.assertTrue(cora.contains(Z, Z))
        self.assertTrue(Z.contains(Z))
        with self.assertRaises(ValueError):
            I.contains(np.array([[0.5]]), tol=-1.0)
        with cora.checks(False):
            self.assertTrue(I.contains(np.array([[0.5], [0.5]]), tol=1e-9))

    def test_a_python_predicate_gets_a_number(self):
        class Ball(cora.ContSet):
            def _dim(self):
                return 1

            def _center(self):
                return np.zeros(1)

        seen = []

        def contains_ball(S1, S2, tol):
            seen.append(tol)
            return True
        # the registered predicate always receives the resolved tolerance
        cora.register_binary("contains", Ball, Ball, contains_ball)
        a = b = Ball()
        with cora.tolerance(1e-7):
            cora.contains(a, b)
        cora.contains(a, b)
        cora.contains(a, b, tol=1e-3)
        self.assertEqual(seen, [1e-7, cora.default_tol(), 1e-3])

    def test_threads_do_not_share_scopes(self):
        seen = []
        with cora.tolerance(1e-6):
            thread = threading.Thread(target=lambda: seen.append(cora.current_tolerance()))
            thread.start()
            thread.join()
            self.assertEqual(cora.current_tolerance(), 1e-6)
        self.assertEqual(seen, [None])
        cora.set_default_tolerance(1e-4)
        thread = threading.Thread(target=lambda: seen.append(cora.current_tolerance()))
        thread.start()
        thread.join()
        self.assertEqual(seen, [None, 1e-4])


if __name__ == "__main__":
    unittest.main()
