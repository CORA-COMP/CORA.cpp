// test_linear_sys_reach_enclosure - linear_sys reach encloses: the true trajectories, both time
// points of a step, and a known 1-D flow tightly
//
// CORA: contDynamics/@linearSys/reach.m

#include "cont_dynamics/linear_sys/linear_sys_testing.h"
#include "cora/global/rng.h"
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
        const Zonotope X0 = Zonotope::generate_random(n, m, rng);
        const Eigen::MatrixXd c0 = host_matrix(X0.center().data(), n, 1),
                              g0 = host_matrix(X0.generators().data(), n, m);
        const Tensor A = tensor_of(tc.A);
        for (const std::string &algorithm : kAlgorithms) {
            const ReachSet r = LinearSys(A).reach({.initial_set = X0, .t_final = 2.0},
                                                  test::lin::algorithm(algorithm, tc.dt, 10));
            const std::string what = b + ": " + tc.what + ": " + name(algorithm);

            const int samples = 300;
            Eigen::MatrixXd beta(m, samples), dirs(n, samples), times(1, samples);
            rng.uniform(beta.data(), beta.size(), -1.0, 1.0);
            rng.normal(dirs.data(), dirs.size(), 1.0);
            rng.uniform(times.data(), times.size(), 0.0, 1.0);
            double worst = 1e9;
            for (std::size_t k = 0; k < r.time_interval.size(); ++k)
                for (int i = 0; i < samples; ++i) {
                    const Eigen::MatrixXd flow = (tc.A * (double(k) + times(i)) * tc.dt).exp();
                    const Eigen::VectorXd x = flow * (c0 + g0 * beta.col(i));
                    const Eigen::VectorXd d = dirs.col(i);
                    worst = std::min(worst, (support(r.time_interval[k], d) - d.dot(x)) / d.norm());
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
    const Zonotope X0 = Zonotope::generate_random(3, 4, rng);
    Eigen::MatrixXd dirs(3, 40);
    rng.normal(dirs.data(), dirs.size(), 1.0);
    for (const std::string &algorithm : kAlgorithms) {
        const ReachSet r = LinearSys(system_matrix(s))
                               .reach({.initial_set = X0, .t_final = 1.0},
                                      test::lin::algorithm(algorithm, 0.2, 6));
        double worst = 1e9;
        for (std::size_t k = 0; k < r.time_interval.size(); ++k)
            for (int i = 0; i < 40; ++i) {
                const Eigen::VectorXd d = dirs.col(i);
                const double hull = support(r.time_interval[k], d);
                worst = std::min({worst, hull - support(r.time_point[k], d),
                                  hull - support(r.time_point[k + 1], d)});
            }
        check(worst >= -1e-9, b + ": " + name(algorithm) + ": a time point left its enclosure");
    }
}

/// `x' = -x` on `[1, 3]`: the flow moves the set down, so the upper bound of a step is the
/// start's `3 e^{-t}`, tight to `dt²`. The lower bound is `e^{-t-dt}` at best; `linear_combination`
/// builds a symmetric zonotope and gives away up to `dt` on that far side.
void is_tight_in_one_dimension(const std::string &b) {
    const System &s = matlab_reference::scalar();
    const double dt = s.time_step;
    for (const std::string &algorithm : kAlgorithms) {
        const ReachSet r = LinearSys(system_matrix(s))
                               .reach({.initial_set = set_of(s), .t_final = 1.0},
                                      test::lin::algorithm(algorithm, dt, 8));
        for (std::size_t k = 0; k < r.time_interval.size(); ++k) {
            const double t0 = double(k) * dt;
            const double hi = support(r.time_interval[k], Eigen::VectorXd::Ones(1));
            const double lo = -support(r.time_interval[k], -Eigen::VectorXd::Ones(1));
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
    return test::finish("linear_sys reach (enclosure)");
}
