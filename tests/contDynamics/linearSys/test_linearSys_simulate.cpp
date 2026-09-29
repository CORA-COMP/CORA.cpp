// test_linearSys_simulate - linearSys simulate: exact against the matrix exponential, and inside
// the reachable sets

#include "contDynamics/linearSys/linearSysTesting.h"
#include "global/rng.h"
#include "testing.h"

#include <algorithm>
#include <cmath>

using namespace cora::ct;
using test::check;
using test::close;
using namespace test::lin;
using matlab_reference::System;

namespace {

void follows_the_matrix_exponential(const std::string &b) {
    const Eigen::MatrixXd A = oscillator3();
    Eigen::MatrixXd x0(3, 4);
    x0 << 1, 0, -1, 0.5, 0, 1, 0.2, -0.5, 0.3, -0.2, 1, 0;
    const double dt = 0.15;
    const std::vector<Tensor> x = LinearSys(tensor_of(A)).simulate(tensor_of(x0), dt, 1.0);

    check(x.size() == 8, b + ": ceil(1.0 / 0.15) + 1 time points");
    check(close(x[0], tensor_of(x0)), b + ": the start is the start");
    double worst = 0.0;
    for (std::size_t k = 0; k < x.size(); ++k)
        worst = std::max(worst,
                         (host_of(x[k]) - (A * dt * double(k)).exp() * x0).cwiseAbs().maxCoeff());
    check(worst < 1e-12, b + ": a trajectory differs from e^{A t} x0 by " + std::to_string(worst));
}

/// A stable linear system contracts: the norm of every trajectory falls.
void a_stable_system_decays(const std::string &b) {
    const Tensor A({{-1.0, 0.0}, {0.0, -2.0}});
    const std::vector<Tensor> x = LinearSys(A).simulate(Tensor({{1.0}, {1.0}}), 0.1, 3.0);
    const Eigen::MatrixXd last = host_of(x.back());
    check(std::abs(last(0, 0) - std::exp(-3.0)) < 1e-12 &&
              std::abs(last(1, 0) - std::exp(-6.0)) < 1e-12,
          b + ": x' = -x and x' = -2x decay exactly");
}

/// Simulated points lie in the reachable sets: at every time point in the time-point set, and
/// at every point in between in the enclosure of that step. Starting from extreme points of the
/// initial set is the demanding case: the boundary is where a set that is too small shows it.
void simulations_stay_in_the_reachable_set(const std::string &b) {
    cora::Rng rng(11);
    const int n = 3, N = 60, refine = 5;
    const double dt = 0.2, tFinal = 2.0;
    const Tensor A = tensor_of(oscillator3());
    const Zonotope X0 = Zonotope::generateRandom(n, 4, rng);
    const LinearSys sys(A);

    Eigen::MatrixXd dirs(n, 40);
    rng.normal(dirs.data(), dirs.size(), 1.0);

    for (const Algorithm algorithm : {Algorithm::Standard, Algorithm::WrappingFree}) {
        const Reach R = sys.reach(X0, dt, tFinal, 8, algorithm);
        for (const bool extreme : {false, true}) {
            const Tensor start = X0.randPoint(N, rng, extreme ? "extreme" : "standard");
            const std::vector<Tensor> x = sys.simulate(start, dt / refine, tFinal);
            double worst = 1e9;
            for (std::size_t j = 0; j < x.size(); ++j) {
                const Eigen::MatrixXd points = host_of(x[j]);
                const std::size_t step = std::min(j / refine, R.timeInt.size() - 1);
                for (int i = 0; i < 40; ++i) {
                    const Eigen::VectorXd d = dirs.col(i);
                    const double bound = support(R.timeInt[step], d);
                    worst =
                        std::min(worst, (bound - (d.transpose() * points).maxCoeff()) / d.norm());
                    if (j % refine == 0) {
                        const double at = support(R.timePoint[j / refine], d);
                        worst =
                            std::min(worst, (at - (d.transpose() * points).maxCoeff()) / d.norm());
                    }
                }
            }
            check(worst >= -1e-9,
                  b + ": " + (algorithm == Algorithm::Standard ? "standard" : "wrapping-free") +
                      (extreme ? ", extreme" : ", random") +
                      ": a trajectory left the reachable set by " + std::to_string(-worst));
        }
    }
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        follows_the_matrix_exponential(b);
        a_stable_system_decays(b);
        simulations_stay_in_the_reachable_set(b);
    });
    return test::finish("linearSys simulate");
}
