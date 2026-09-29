// test_linearSys_reach_steps - linearSys reach: the time points are exact, the number of steps, and
// a system that does not move

#include "contDynamics/linearSys/linearSysTesting.h"
#include "global/rng.h"
#include "testing.h"

using namespace cora::ct;
using test::check;
using test::close;
using namespace test::lin;
using matlab_reference::System;

namespace {

/// The time-point sets are `e^{A k Δt} X0` exactly, so their support along `d` is the
/// initial set's along `e^{A k Δt}ᵀ d`, computed here on the host.
void time_points_are_exact(const std::string &b) {
    const System &s = matlab_reference::three_dimensional();
    const Eigen::MatrixXd A = host_matrix(s.A, 3, 3), G = host_matrix(s.G, 3, 3);
    const Eigen::VectorXd c = Eigen::Map<const Eigen::VectorXd>(s.c.data(), 3);
    const Eigen::VectorXd d = (Eigen::VectorXd(3) << 0.3, -1.0, 0.7).finished();
    const Reach r = LinearSys(system_matrix(s)).reach(set_of(s), 0.2, 1.0, 6);
    for (std::size_t k = 0; k < r.timePoint.size(); ++k) {
        const Eigen::MatrixXd flow = (A * 0.2 * double(k)).exp();
        const Eigen::VectorXd back = flow.transpose() * d;
        const double want = back.dot(c) + (back.transpose() * G).cwiseAbs().sum();
        check(close(support(r.timePoint[k], d), want, 1e-10),
              b + ": time point " + std::to_string(k));
    }
}

/// Without dynamics nothing moves, so every enclosure is the initial set itself.
void a_zero_system_stays_put(const std::string &b) {
    cora::Rng rng(5);
    const Zonotope X0 = Zonotope::generateRandom(2, 3, rng);
    const Eigen::VectorXd d = Eigen::VectorXd::Ones(2);
    for (const Algorithm algorithm : kAlgorithms) {
        const Reach r = LinearSys(Tensor::zeros({2, 2})).reach(X0, 0.5, 2.0, 5, algorithm);
        for (const Zonotope &Z : r.timeInt)
            check(close(support(Z, d), support(X0, d), 1e-12),
                  b + ": a zero system's enclosure grew");
    }
}

/// The last step is a full one even when `tFinal` is not a multiple of the step.
void counts_steps_up(const std::string &b) {
    const System &s = matlab_reference::scalar();
    const LinearSys sys(system_matrix(s));
    check(sys.reach(set_of(s), 0.1, 0.25, 6).timeInt.size() == 3, b + ": ceil(0.25 / 0.1)");
    check(sys.reach(set_of(s), 0.1, 0.3, 6).timeInt.size() == 3, b + ": 0.3 / 0.1 stays 3");
    check(sys.reach(set_of(s), 0.1, 0.05, 6).timePoint.size() == 2, b + ": one step");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        time_points_are_exact(b);
        a_zero_system_stays_put(b);
        counts_steps_up(b);
    });
    return test::finish("linearSys reach (steps)");
}
