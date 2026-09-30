// test_linearSys_reach_inputs - linearSys reach with an input set U, against MATLAB CORA, against
// simulations with constant and piecewise-constant inputs, and for the errors it reports

#include "contDynamics/linearSys/linearSysTesting.h"
#include "contDynamics/linearSys/matlabReferenceInputs.h"
#include "global/rng.h"
#include "testing.h"

#include <algorithm>
#include <cmath>

using namespace cora;
using test::check;
using test::close;
using test::throws;
using namespace test::lin;
using matlab_reference::InputSystem;

namespace {

struct Setup {
    const InputSystem &s;
    Tensor A, B, C;
    Zonotope X0, U;
    explicit Setup(const InputSystem &s)
        : s(s), A(Tensor::fromData(s.A, {3, 3})), B(Tensor::fromData(s.B, {3, 2})),
          C(Tensor::fromData(s.C, {2, 3})),
          X0(Tensor::fromData(s.c0, {3, 1}), Tensor::fromData(s.G0, {3, 3})),
          U(Tensor::fromData(s.cU, {2, 1}), Tensor::fromData(s.GU, {2, 3})) {}
    double tFinal() const { return s.timeStep * s.steps; }
};

/// The largest distance between the support values of the sets and the reference.
double distance(const std::vector<Zonotope> &sets, const std::vector<double> &dirs, int cols,
                const std::vector<std::vector<double>> &reference) {
    double worst = 0.0;
    for (std::size_t k = 0; k < reference.size(); ++k)
        for (int i = 0; i < int(dirs.size()) / cols; ++i) {
            const Eigen::VectorXd d = host_matrix(dirs, dirs.size() / cols, cols).row(i);
            worst = std::max(worst, std::abs(support(sets[k], d) - reference[k][i]));
        }
    return worst;
}

void matches_matlab_cora(const std::string &b) {
    for (const Algorithm algorithm : kAlgorithms) {
        const InputSystem &s = algorithm == Algorithm::Standard ? matlab_reference::standardInputs()
                                                                : matlab_reference::wrappingFreeInputs();
        const Setup in(s);
        const std::string what = b + ": " + name(algorithm);
        const LinearSys sys(in.A, in.B, in.C);
        const Reach R = sys.reach(in.X0, in.U, s.timeStep, in.tFinal(), s.taylorTerms, algorithm);
        check(R.timeInt.size() == 5 && R.timePoint.size() == 6, what + ": the number of steps");

        check(distance(R.timeInt, s.dirsX, 3, s.stateTi) < 1e-10, what + ": state enclosures");
        check(distance(R.timePoint, s.dirsX, 3, s.stateTp) < 1e-10, what + ": state time points");
        const Reach Y = sys.outputSet(R);
        check(distance(Y.timeInt, s.dirsY, 2, s.outputTi) < 1e-10, what + ": output enclosures");
        check(distance(Y.timePoint, s.dirsY, 2, s.outputTp) < 1e-10, what + ": output time points");
    }
}

/// Every trajectory of x' = A x + B u(t) with u(t) in U lies in the sets: the points at the time
/// points in the time-point sets, those in between in the enclosures. The input is piecewise
/// constant with extreme values of U, which is the demanding case.
void simulations_stay_in_the_reachable_set(const std::string &b) {
    cora::Rng rng(5);
    const InputSystem &s = matlab_reference::standardInputs();
    const Setup in(s);
    const LinearSys sys(in.A, in.B);
    const int N = 40, refine = 4;
    Eigen::MatrixXd dirs(3, 30);
    rng.normal(dirs.data(), dirs.size(), 1.0);

    for (const Algorithm algorithm : kAlgorithms)
        for (const int zonotopeOrder : {50, 2}) {
            const Reach R = sys.reach(in.X0, in.U, s.timeStep, in.tFinal(), 6, algorithm, zonotopeOrder);
            Tensor x = in.X0.randPoint(N, rng, "extreme");
            double worst = -1e9;
            for (int k = 0; k < s.steps; ++k) {
                const Tensor u = in.U.randPoint(N, rng, "extreme");
                for (int r = 1; r <= refine; ++r) {
                    // a point inside the step: the same input from the step's start
                    const double t = s.timeStep * r / refine;
                    const Tensor xr = sys.simulate(x, u, t, t).back();
                    const Eigen::MatrixXd P = host_of(xr);
                    for (int i = 0; i < dirs.cols(); ++i)
                        for (int j = 0; j < N; ++j)
                            worst = std::max(worst, dirs.col(i).dot(P.col(j)) -
                                                        support(R.timeInt[k], dirs.col(i)));
                }
                x = sys.simulate(x, u, s.timeStep, s.timeStep).back();
                const Eigen::MatrixXd P = host_of(x);
                for (int i = 0; i < dirs.cols(); ++i)
                    for (int j = 0; j < N; ++j)
                        worst = std::max(worst, dirs.col(i).dot(P.col(j)) -
                                                    support(R.timePoint[k + 1], dirs.col(i)));
            }
            check(worst <= 1e-9, b + ": " + name(algorithm) + " order " + std::to_string(zonotopeOrder) +
                                     ": a trajectory leaves the sets by " + std::to_string(worst));
        }
}

/// With U = {0} the input adds nothing: wrapping-free gives the sets of the system without input.
void a_zero_input_changes_nothing(const std::string &b) {
    const Setup in(matlab_reference::standardInputs());
    const Zonotope U0(Tensor::zeros({2, 1}), Tensor::zeros({2, 1}));
    const Reach with = LinearSys(in.A, in.B).reach(in.X0, U0, 0.2, 1.0, 6, Algorithm::WrappingFree);
    const Reach without = LinearSys(in.A).reach(in.X0, 0.2, 1.0, 6, Algorithm::WrappingFree);
    double worst = 0.0;
    for (std::size_t k = 0; k < with.timeInt.size(); ++k)
        for (const Eigen::VectorXd &d : {Eigen::VectorXd(Eigen::Vector3d(1, 0, 0)),
                                         Eigen::VectorXd(Eigen::Vector3d(1, -1, 2))})
            worst = std::max(worst, std::abs(support(with.timeInt[k], d) - support(without.timeInt[k], d)));
    check(worst < 1e-12, b + ": U = 0 changed the sets by " + std::to_string(worst));
}

void reports_errors(const std::string &b) {
    const Setup in(matlab_reference::standardInputs());
    const Zonotope wrong(Tensor::zeros({3, 1}), Tensor::eye(3));
    check(throws([&] { LinearSys(in.A).reach(in.X0, in.U, 0.2, 1.0, 6); }),
          b + ": an input set without B is refused");
    check(throws([&] { LinearSys(in.A, in.B).reach(in.X0, wrong, 0.2, 1.0, 6); }),
          b + ": an input set of the wrong dimension is refused");
    check(throws([&] { LinearSys(in.A, in.B).reach(in.X0, in.U, 0.2, 1.0, 6, static_cast<Algorithm>(7)); }),
          b + ": an unknown linAlg is refused");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        matches_matlab_cora(b);
        simulations_stay_in_the_reachable_set(b);
        a_zero_input_changes_nothing(b);
        reports_errors(b);
    });
    return test::finish("linearSys reach (inputs)");
}
