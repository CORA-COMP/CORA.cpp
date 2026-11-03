// test_linear_sys_reach_matlab - linear_sys reach against MATLAB CORA: the enclosures and time
// points of three systems
//
// CORA: contDynamics/@linearSys/reach.m

#include "cont_dynamics/linear_sys/linear_sys_testing.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <algorithm>
#include <cmath>

using namespace cora;
using test::check;
using namespace test::lin;
using matlab_reference::System;

namespace {

/// Against MATLAB CORA. Without inputs its two algorithms give the wrapping-free sets; our
/// standard algorithm agrees on the first step, where `F X0` is the same.
void matches_matlab_cora(const std::string &b) {
    for (const System *s : {&matlab_reference::oscillator(), &matlab_reference::three_dimensional(),
                            &matlab_reference::scalar()}) {
        const std::size_t steps = s->time_interval.size();
        const int n = s->n;
        for (const std::string &algorithm : kAlgorithms) {
            const ReachSet r =
                LinearSys(system_matrix(*s))
                    .reach({.initial_set = set_of(*s), .t_final = s->time_step * steps},
                           test::lin::algorithm(algorithm, s->time_step, s->taylor_terms));
            const std::string what = b + ": " + name(algorithm) + " n=" + std::to_string(n);
            check(r.time_interval.size() == steps && r.time_point.size() == steps + 1,
                  what + ": the number of steps");
            double ti_err = 0.0, tp_err = 0.0;
            for (std::size_t k = 0; k < steps; ++k) {
                // Only the first step of the standard algorithm is CORA's.
                if (algorithm == "standard" && k > 0) break;
                for (std::size_t i = 0; i < s->dirs.size() / n; ++i) {
                    const Eigen::VectorXd d = host_matrix(s->dirs, s->dirs.size() / n, n).row(i);
                    ti_err = std::max(
                        ti_err, std::abs(support(r.time_interval[k], d) - s->time_interval[k][i]));
                }
            }
            for (std::size_t k = 0; k < s->time_point.size(); ++k)
                for (std::size_t i = 0; i < s->dirs.size() / n; ++i) {
                    const Eigen::VectorXd d = host_matrix(s->dirs, s->dirs.size() / n, n).row(i);
                    tp_err = std::max(tp_err,
                                      std::abs(support(r.time_point[k], d) - s->time_point[k][i]));
                }
            check(ti_err < 1e-12,
                  what + ": enclosures differ from MATLAB CORA by " + std::to_string(ti_err));
            check(tp_err < 1e-12,
                  what + ": time points differ from MATLAB CORA by " + std::to_string(tp_err));
        }
    }
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) { matches_matlab_cora(b); });
    return test::finish("linear_sys reach (MATLAB)");
}
