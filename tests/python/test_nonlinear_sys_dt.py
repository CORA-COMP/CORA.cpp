"""test_nonlinear_sys_dt - NonlinearSysDT reach, post, output_set and simulate in Python.

CORA authors: Matthias Althoff, Niklas Kochdumper
Authors:       Claude Sonnet 5.5
Written:       24-November-2026
Last update:   ---
Reviewed:      ---
"""
# CORA: contDynamics/@nonlinearSysDT/nonlinearSysDT.m
# CORA: contDynamics/@nonlinearSysDT/reach.m
# CORA: contDynamics/@nonlinearSysDT/post.m
# CORA: contDynamics/@nonlinearSysDT/linReach.m
# CORA: contDynamics/@nonlinearSysDT/simulate.m
# CORA: contDynamics/@contDynamics/outputSet.m
# PROPERTY: contDynamics/@nonlinearSysDT/reach.m

import unittest

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

def pendulum():
    """x+ = (x0 + 0.1 x1 + u, x1 - 0.1 sin x0), y = x0 x1 + x0."""
    # f and g are written on symbolic states and inputs
    return cora.NonlinearSysDT(
        lambda x, u: [x[0] + 0.1 * x[1] + u[0], x[1] - 0.1 * cora.sin(x[0])],
        2, inputs=1, time_step=0.1,
        g=lambda x, u: [x[0] * x[1] + x[0]], outputs=1,
    )


def start():
    # the initial set: a box around (1, 0.5)
    return cora.Zonotope(cora.Tensor([1.0, 0.5]), 0.05 * cora.eye(2))


class Reach(unittest.TestCase):
    def test_a_linear_system_is_exact(self):
        sys = cora.NonlinearSysDT(lambda x, u: [0.5 * x[0] + u[0]], 1, inputs=1, time_step=0.25)
        X0 = cora.Zonotope(cora.Tensor([1.0]), 0.2 * cora.eye(1))
        U = cora.Zonotope(cora.Tensor([0.0]), 0.1 * cora.eye(1))
        R = sys.reach(initial_set=X0, input_set=U, t_final=0.5)
        self.assertEqual(len(R.states), 3)
        box = R.states[2].interval()
        np.testing.assert_allclose(np.asarray(box.inf).ravel(), [0.05], atol=1e-12)
        np.testing.assert_allclose(np.asarray(box.sup).ravel(), [0.45], atol=1e-12)

    def test_trajectories_stay_in_the_sets(self):
        # sampled constant inputs and start points, simulated, lie in the hulls of the sets
        sys = pendulum()
        U = cora.Zonotope(cora.Tensor([0.0]), 0.01 * cora.eye(1))
        rng = np.random.default_rng(3)
        for order in (2, 3):
            # both tensor orders, of the dynamics and of the output equation
            R = sys.reach(initial_set=start(), input_set=U, t_final=0.5, tensor_order=order,
                          tensor_order_output=order)
            x0 = np.array([[1.0, 0.5]]) + 0.05 * rng.uniform(-1, 1, (20, 2))
            u = 0.01 * rng.uniform(-1, 1, (20, 1))
            T = sys.simulate(x0, 0.5, input=u, with_output=True)
            # the hull of each set bounds the trajectory at its time
            for k in range(len(R.states)):
                for sets, data in ((R.states, np.asarray(T["x"])), (R.outputs, np.asarray(T["y"]))):
                    box = sets[k].interval()
                    self.assertTrue(np.all(data[k] >= np.asarray(box.inf).ravel() - 1e-9))
                    self.assertTrue(np.all(data[k] <= np.asarray(box.sup).ravel() + 1e-9))

    def test_post_is_the_first_step(self):
        # post and output_set are the steps reach runs
        sys = pendulum()
        U = cora.Zonotope(cora.Tensor([0.0]), 0.01 * cora.eye(1))
        R = sys.reach(initial_set=start(), input_set=U, t_final=0.1)
        step = sys.post(start(), U)
        np.testing.assert_array_equal(np.asarray(step.set.center), np.asarray(R.states[1].center))
        Y = sys.output_set(start(), U)
        np.testing.assert_array_equal(np.asarray(Y.set.center), np.asarray(R.outputs[0].center))

    def test_wrong_arguments_are_described(self):
        # a tensor order of 4 and a sampling time of 0
        with self.assertRaises(ValueError):
            pendulum().reach(initial_set=start(), t_final=0.1, tensor_order=4)
        with self.assertRaises(ValueError):
            cora.NonlinearSysDT(lambda x: [x[0]], 1, time_step=0.0)


class Simulate(unittest.TestCase):
    def test_the_recursion(self):
        # x(1) = (0.5, 0.5 cos 0), x(2) = (1, 0.5 + 0.5 cos 0.5)
        sys = cora.NonlinearSysDT(lambda x, u: [x[0] + u[0], x[1] + u[0] * cora.cos(x[0])], 2,
                                  inputs=1, time_step=0.25)
        T = sys.simulate(np.array([[0.0, 0.0]]), 0.5, input=np.array([0.5]))
        x = np.asarray(T["x"])
        self.assertEqual(x.shape, (3, 1, 2))
        np.testing.assert_allclose(x[1].ravel(), [0.5, 0.5], atol=1e-12)
        np.testing.assert_allclose(x[2].ravel(), [1.0, 0.5 + 0.5 * np.cos(0.5)], atol=1e-12)

    def test_a_signal_of_the_wrong_length_is_refused(self):
        sys = pendulum()
        with self.assertRaises(ValueError):
            sys.simulate(np.array([[0.0, 0.0]]), 0.5, input_signal=[np.array([0.1])])


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
