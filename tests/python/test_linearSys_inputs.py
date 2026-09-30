"""test_linearSys_inputs - LinearSys(A, B, C) in Python: reach with an input set U against MATLAB CORA
(R2025b), outputSet, correctionMatrixInput, simulate with inputs, and the errors."""
import unittest

import numpy as np

import cora

# The system of the MATLAB call linearSys(A, B, [], C), reach with params.U, step 0.2, order 6.
A = np.array([[-0.3, 1, 0], [-1, -0.2, 0.4], [0, -0.5, -0.1]])
B = np.array([[1, 0], [0.5, -1], [0, 0.3]])
C = np.array([[1, 0, -1], [0, 1, 0.5]])
X0 = cora.Zonotope(np.array([1, -0.5, 0.2]), np.array([[0.2, 0.1, 0], [0, 0.1, 0.3], [0.1, 0, 0.1]]))
U = cora.Zonotope(np.array([0.4, -0.2]), np.array([[0.1, 0.05, 0], [0, 0.1, 0.02]]))
DIRS_X = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 1], [1, -2, 0.5]])
DIRS_Y = np.array([[1, 0], [0, 1], [1, -1]])
G_INF = np.array([[-2.5459749078231472e-05, -0.0050016028597035271, -0.00020530569255280311],
            [-0.00028841773260259622, -2.6485581539672779e-05, -0.0020005375348072889],
            [-0.00025663211569100232, -9.4778484769581458e-05, -2.7997240239361324e-06]])
G_SUP = np.array([[0.0019670832372059455, 0.00028841773260260175, 8.102211796562452e-06],
            [0.0050016028597035219, 0.0015953922973354145, 7.582278781567017e-05],
            [1.0127764745701503e-05, 0.002500671918509105, 0.00059751905118117195]])

# Support values [step][direction] of the states (intervals, time points) and of the outputs.
REFERENCE = {
    "standard": {
        "stateTi": np.array([
            [1.4094432951328197, -0.031777261599073237, 1.6694419529637321, 3.2080338181438424],
            [1.3956856613747277, -0.087042110599158895, 1.5134238319873508, 3.2113782880369852],
            [1.3532149504229656, -0.14293319207735772, 1.3721796966330588, 3.233587186134407],
            [1.2916924369356173, -0.19161031612531348, 1.2366933713628587, 3.2331022188421397],
            [1.2161962332094858, -0.1837509040310461, 1.1130145129518556, 3.1643940406039142]]),
        "stateTp": np.array([
            [1.3, -0.099999999999999978, 1.6000000000000001, 3],
            [1.3015322020045024, -0.14663010615312744, 1.465184794511178, 3.0612173984069702],
            [1.2763671474625489, -0.1927722564089196, 1.32948236836851, 3.0716163772543403],
            [1.2290023031838353, -0.23556951670869042, 1.2007715500077052, 3.09229466699589],
            [1.1638538958537934, -0.26344108861257948, 1.084007195218873, 3.043550523367847],
            [1.0908853632952158, -0.23791342985575714, 0.9836676762064267, 2.9344425990577068]]),
        "outputTi": np.array([
            [1.1612650790591732, 0.13585329905924443, 1.8222029622505813],
            [1.0320496228946543, 0.019212344807178428, 1.7130653288335793],
            [0.97949378077489602, -0.05572877250552738, 1.5490204701228452],
            [0.90275433222823898, -0.084176785332299775, 1.3389840425638226],
            [0.82574151843015775, -0.054897635242623311, 1.2027565933136524]]),
        "outputTp": np.array([
            [1.1000000000000001, 0.099999999999999978, 1.7000000000000002],
            [0.92384208489560371, -0.011754051532732512, 1.6166383106066933],
            [0.8637103361935603, -0.098897078083272416, 1.4695145350062109],
            [0.80136207709767637, -0.14091899864226765, 1.2724213500308612],
            [0.71990537413718736, -0.14656385222330875, 1.0464205669513251],
            [0.6767148687346064, -0.11779158883553048, 0.99128301428451782]]),
    },
    "wrapping-free": {
        "stateTi": np.array([
            [1.4094432951328197, -0.031777261599073237, 1.6694419529637321, 3.2080338181438424],
            [1.3956856613747277, -0.087042110599158895, 1.5834812100751865, 3.2810471685405966],
            [1.3532149504229656, -0.14293319207735772, 1.4747804198319274, 3.314545503711277],
            [1.2916924369356173, -0.19161031612531343, 1.3744168607473641, 3.3213759835423322],
            [1.2161962332094858, -0.18375090403104599, 1.2942429244317604, 3.2657323465191608]]),
        "stateTp": np.array([
            [1.3, -0.099999999999999978, 1.6000000000000001, 3],
            [1.3015322020045024, -0.14663010615312744, 1.465184794511178, 3.0612173984069702],
            [1.2763671474625486, -0.19277225640891954, 1.3995397464563459, 3.1412852577579518],
            [1.2290023031838353, -0.23556951670869036, 1.3033722732065738, 3.1732529845727595],
            [1.1638538958537934, -0.26344108861257937, 1.2217306846033782, 3.13182428806804],
            [1.0908853632952158, -0.23791342985575703, 1.1648960876863312, 3.0357809049729538]]),
        "outputTi": np.array([
            [1.1612650790591732, 0.13585329905924443, 1.8222029622505813],
            [1.0570476518921785, 0.036945042890052848, 1.7955469523559866],
            [1.0107642832528281, -0.026229034352287184, 1.6680171313108119],
            [0.93411368230700842, -0.042326528979977074, 1.493774128014727],
            [0.85710102985629377, 0.00078215257832464635, 1.3896288960416454]]),
        "outputTp": np.array([
            [1.1000000000000001, 0.099999999999999978, 1.7000000000000002],
            [0.92384208489560371, -0.011754051532732512, 1.6166383106066933],
            [0.88870836519108454, -0.08116438000039794, 1.5519961585286177],
            [0.83263257957560843, -0.11141926048902745, 1.3914180112188281],
            [0.75126472421595669, -0.10471359587098605, 1.2012106524022297],
            [0.70807438016074231, -0.062111801014582524, 1.1781553170125107]]),
    },
}


def supports(sets, dirs):
    """The support function of every zonotope along every row of dirs: (sets, directions)."""
    return np.array([[d @ Z.c + np.abs(Z.G.T @ d).sum() for d in dirs] for Z in sets])


class ReachWithInputs(unittest.TestCase):
    def setUp(self):
        self.addCleanup(cora.setBackend, cora.backend())
        cora.setBackend("eigen")

    def test_matches_matlab_cora(self):
        sys = cora.LinearSys(A, B, C)
        for linAlg, ref in REFERENCE.items():
            R = sys.reach(X0, timeStep=0.2, tFinal=1.0, taylorTerms=6, linAlg=linAlg, U=U)
            self.assertEqual((len(R.timeInt), len(R.timePoint)), (5, 6))
            np.testing.assert_allclose(supports(R.timeInt, DIRS_X), ref["stateTi"], atol=1e-10)
            np.testing.assert_allclose(supports(R.timePoint, DIRS_X), ref["stateTp"], atol=1e-10)
            Y = sys.outputSet(R)
            np.testing.assert_allclose(supports(Y.timeInt, DIRS_Y), ref["outputTi"], atol=1e-10)
            np.testing.assert_allclose(supports(Y.timePoint, DIRS_Y), ref["outputTp"], atol=1e-10)

    def test_the_correction_matrix_of_the_input_matches_matlab_cora(self):
        G = cora.LinearSys(A).correctionMatrixInput(0.2, 6)
        np.testing.assert_allclose(G.inf, G_INF, atol=1e-12)
        np.testing.assert_allclose(G.sup, G_SUP, atol=1e-12)

    def test_trajectories_with_piecewise_constant_inputs_stay_inside(self):
        sys, rng = cora.LinearSys(A, B), cora.Rng(2)
        for linAlg in ("standard", "wrapping-free"):
            R = sys.reach(X0, 0.2, 1.0, 6, linAlg, U=U, zonotopeOrder=2)
            x = X0.randPoint(30, rng, "extreme")
            for k in range(5):
                x = sys.simulate(x, 0.2, 0.2, u=U.randPoint(30, rng, "extreme"))[-1]
                for d in np.random.default_rng(k).normal(size=(10, 3)):
                    self.assertLessEqual((d @ x).max(), R.timePoint[k + 1].supportFunc(d) + 1e-9)

    def test_outputs_and_properties(self):
        sys = cora.LinearSys(A, B, C)
        np.testing.assert_array_equal(sys.B, B)
        np.testing.assert_array_equal(sys.C, C)
        self.assertIsNone(cora.LinearSys(A, B).C)
        R = sys.reach(X0, 0.2, 0.6, 6, U=U)
        Y = sys.outputSet(R)
        np.testing.assert_allclose(Y.timeInt[1].c, C @ R.timeInt[1].c)
        self.assertEqual(Y.timePoint[0].c.shape, (2,))
        # without C the output is the state
        np.testing.assert_array_equal(cora.LinearSys(A, B).outputSet(R).timeInt[2].G, R.timeInt[2].G)

    def test_simulate_with_inputs_follows_the_closed_form(self):
        x0 = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]])
        u = np.array([[0.4, -1.0], [-0.2, 0.5]])
        x = cora.LinearSys(A, B).simulate(x0, 0.15, 1.0, u=u)
        self.assertEqual(x.shape, (8, 3, 2))
        for k in range(8):
            E = expm(A * 0.15 * k)
            np.testing.assert_allclose(x[k], E @ x0 + np.linalg.inv(A) @ (E - np.eye(3)) @ B @ u,
                                       atol=1e-12)
        one = cora.LinearSys(A, B).simulate(x0, 0.15, 1.0, u=np.array([0.4, -0.2]))
        both = cora.LinearSys(A, B).simulate(x0, 0.15, 1.0, u=np.tile([[0.4], [-0.2]], (1, 2)))
        np.testing.assert_allclose(one, both, atol=1e-14)

    def test_simulate_random_draws_the_inputs_from_the_set(self):
        x = cora.LinearSys(A, B).simulateRandom(X0, 6, 0.1, 0.5, cora.Rng(4), U=U)
        self.assertEqual(x.shape, (6, 3, 6))

    def test_errors(self):
        with self.assertRaises(Exception):
            cora.LinearSys(A).reach(X0, 0.2, 1.0, 6, U=U)            # U needs B
        with self.assertRaises(Exception):
            cora.LinearSys(A, B).reach(X0, 0.2, 1.0, 6, U=X0)        # U of the wrong dimension
        with self.assertRaises(ValueError):
            cora.LinearSys(A, B).reach(X0, 0.2, 1.0, 6, "unknown", U=U)
        with self.assertRaises(Exception):
            cora.LinearSys(A).simulate(np.zeros((3, 2)), 0.1, 0.5, u=np.zeros((2, 2)))


def expm(M):
    w, V = np.linalg.eig(M)
    return (V @ np.diag(np.exp(w)) @ np.linalg.inv(V)).real


if __name__ == "__main__":
    unittest.main()
