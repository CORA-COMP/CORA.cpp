"""test_tensor - cora.Tensor: constructors and elementwise functions as static methods."""
import math
import unittest

import numpy as np
try:
    import torch
except ImportError:  # numpy-only build: test_numpy.py covers it
    raise unittest.SkipTest("needs torch")

import cora


class Constructors(unittest.TestCase):
    def tearDown(self):
        cora.setBackend("torch")

    def test_a_tensor_is_made_from_data_and_stays_callable(self):
        t = cora.Tensor([[1, 2], [3, 4]])
        self.assertIsInstance(t, torch.Tensor)
        self.assertEqual(t.dtype, torch.float64)
        self.assertEqual(cora.Tensor([1, 2], dtype="float32").dtype, torch.float32)

    def test_ones_zeros_eye_and_randn_are_static_methods_and_functions(self):
        torch.testing.assert_close(cora.Tensor.ones(2, 3), torch.ones(2, 3, dtype=torch.float64))
        torch.testing.assert_close(cora.Tensor.zeros((2, 2)), torch.zeros(2, 2, dtype=torch.float64))
        torch.testing.assert_close(cora.Tensor.eye(3), torch.eye(3, dtype=torch.float64))
        self.assertEqual(tuple(cora.Tensor.randn(4, 2, seed=1).shape), (4, 2))
        torch.testing.assert_close(cora.ones(2), cora.Tensor.ones(2))
        torch.testing.assert_close(cora.eye(2), cora.Tensor.eye(2))

    def test_the_constructors_follow_the_backend(self):
        cora.setBackend("eigen")
        self.assertIsInstance(cora.Tensor.ones(2, 2), np.ndarray)
        self.assertIsInstance(cora.Tensor.eye(2), np.ndarray)
        self.assertIsInstance(cora.Tensor([1.0, 2.0]), np.ndarray)

    def test_a_wrong_dtype_is_described(self):
        with self.assertRaises(ValueError) as caught:
            cora.Tensor.ones(2, dtype="int8")
        self.assertIn("unknown dtype", str(caught.exception))


class Functions(unittest.TestCase):
    def tearDown(self):
        cora.setBackend("torch")

    def test_functions_of_numbers_and_arrays_need_no_math_or_numpy(self):
        self.assertAlmostEqual(float(cora.Tensor.cos(0.2)), math.cos(0.2))
        self.assertAlmostEqual(float(cora.Tensor.sin(0.2)), math.sin(0.2))
        x = [0.5, 1.0, 2.0]
        for name, f in (("tan", math.tan), ("exp", math.exp), ("log", math.log), ("sqrt", math.sqrt)):
            np.testing.assert_allclose(getattr(cora.Tensor, name)(x).numpy(), [f(v) for v in x])

    def test_a_rotation_matrix_is_built_from_them(self):
        phi = 0.2
        A = cora.Tensor([[cora.Tensor.cos(phi), cora.Tensor.sin(phi)],
                         [-cora.Tensor.sin(phi), cora.Tensor.cos(phi)]])
        np.testing.assert_allclose(A.numpy(), [[math.cos(phi), math.sin(phi)],
                                               [-math.sin(phi), math.cos(phi)]])

    def test_functions_of_the_package_also_take_symbolic_states(self):
        x = cora.Expr.var(0)
        self.assertAlmostEqual(cora.sin(x).eval([0.3]), math.sin(0.3))
        self.assertAlmostEqual(float(cora.sin(0.3)), math.sin(0.3))
        with self.assertRaises(TypeError):
            cora.tan(x)

    def test_functions_follow_the_backend(self):
        cora.setBackend("eigen")
        self.assertIsInstance(cora.Tensor.cos([0.0, 1.0]), np.ndarray)


if __name__ == "__main__":
    unittest.main()
