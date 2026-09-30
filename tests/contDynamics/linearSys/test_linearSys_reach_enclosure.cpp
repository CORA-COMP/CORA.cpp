// test_linearSys_reach_enclosure - linearSys reach encloses: the true trajectories, both time
// points of a step, and a known 1-D flow tightly

#include "contDynamics/linearSys/linearSysTesting.h"
#include "global/rng.h"
#include "testing.h"

#include <algorithm>
#include <cmath>

using namespace cora;
using test::check;
using test::close;
using namespace test::lin;
using matlab_reference::System;

namespace {

struct Case {
    Eigen::MatrixXd A;
    double dt;
    std::string what;
};

/// Points of the initial set pushed along the flow, at times inside every step, must lie
/// under the support function of the step's enclosure in every sampled direction. The fast
/// rotation turns a long way per step, so the flow bulges far out of the chord between the
/// time points and only the curvature enlargement keeps it inside.
void encloses_the_trajectories(const std::string &b) {
    cora::Rng rng(3);
    const int n = 3, m = 4;
    Eigen::MatrixXd gentle(n, n), rotation = Eigen::MatrixXd::Zero(n, n);
    gentle << -0.3, 1.0, 0.0, -1.0, -0.2, 0.4, 0.0, -0.5, -0.1;
    rotation(0, 1) = 3.0;
    rotation(1, 0) = -3.0;
    rotation(2, 2) = -0.5;

    for (const Case &tc : {Case{gentle, 0.1, "gentle"}, Case{rotation, 0.3, "fast rotation"}}) {
        const Zonotope X0 = Zonotope::generateRandom(n, m, rng);
        const Eigen::MatrixXd c0 = host_matrix(X0.c.data(), n, 1),
                              g0 = host_matrix(X0.G.data(), n, m);
        const Tensor A = tensor_of(tc.A);
        for (const Algorithm algorithm : kAlgorithms) {
            const Reach r = LinearSys(A).reach(X0, tc.dt, 2.0, 10, algorithm);
            const std::string what = b + ": " + tc.what + ": " + name(algorithm);

            const int samples = 300;
            Eigen::MatrixXd beta(m, samples), dirs(n, samples), times(1, samples);
            rng.uniform(beta.data(), beta.size(), -1.0, 1.0);
            rng.normal(dirs.data(), dirs.size(), 1.0);
            rng.uniform(times.data(), times.size(), 0.0, 1.0);
            double worst = 1e9;
            for (std::size_t k = 0; k < r.timeInt.size(); ++k)
                for (int i = 0; i < samples; ++i) {
                    const Eigen::MatrixXd flow = (tc.A * (double(k) + times(i)) * tc.dt).exp();
                    const Eigen::VectorXd x = flow * (c0 + g0 * beta.col(i));
                    const Eigen::VectorXd d = dirs.col(i);
                    worst = std::min(worst, (support(r.timeInt[k], d) - d.dot(x)) / d.norm());
                }
            check(worst >= -1e-9,
                  what + ": a trajectory left its enclosure by " + std::to_string(-worst));
        }
    }
}

/// Each enclosure contains the time point at either end of its step.
void encloses_both_time_points(const std::string &b) {
    cora::Rng rng(7);
    const System &s = matlab_reference::three_dimensional();
    const Zonotope X0 = Zonotope::generateRandom(3, 4, rng);
    Eigen::MatrixXd dirs(3, 40);
    rng.normal(dirs.data(), dirs.size(), 1.0);
    for (const Algorithm algorithm : kAlgorithms) {
        const Reach r = LinearSys(system_matrix(s)).reach(X0, 0.2, 1.0, 6, algorithm);
        double worst = 1e9;
        for (std::size_t k = 0; k < r.timeInt.size(); ++k)
            for (int i = 0; i < 40; ++i) {
                const Eigen::VectorXd d = dirs.col(i);
                const double hull = support(r.timeInt[k], d);
                worst = std::min({worst, hull - support(r.timePoint[k], d),
                                  hull - support(r.timePoint[k + 1], d)});
            }
        check(worst >= -1e-9, b + ": " + name(algorithm) + ": a time point left its enclosure");
    }
}

/// `x' = -x` on `[1, 3]`: the flow moves the set down, so the upper bound of a step is the
/// start's `3 e^{-t}`, tight to `dt²`. The lower bound is `e^{-t-dt}` at best; `linComb`
/// builds a symmetric zonotope and gives away up to `dt` on that far side.
void is_tight_in_one_dimension(const std::string &b) {
    const System &s = matlab_reference::scalar();
    const double dt = s.timeStep;
    for (const Algorithm algorithm : kAlgorithms) {
        const Reach r = LinearSys(system_matrix(s)).reach(set_of(s), dt, 1.0, 8, algorithm);
        for (std::size_t k = 0; k < r.timeInt.size(); ++k) {
            const double t0 = double(k) * dt;
            const double hi = support(r.timeInt[k], Eigen::VectorXd::Ones(1));
            const double lo = -support(r.timeInt[k], -Eigen::VectorXd::Ones(1));
            const std::string what = b + ": " + name(algorithm) + ": step " + std::to_string(k);
            check(hi >= 3 * std::exp(-t0) - 1e-12 && hi <= 3 * std::exp(-t0) + 1e-9 + 3 * dt * dt,
                  what + ": upper bound");
            check(lo <= std::exp(-t0 - dt) + 1e-12 &&
                      lo >= std::exp(-t0 - dt) - 3 * dt * std::exp(-t0),
                  what + ": lower bound");
        }
    }
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        encloses_the_trajectories(b);
        encloses_both_time_points(b);
        is_tight_in_one_dimension(b);
    });
    return test::finish("linearSys reach (enclosure)");
}
