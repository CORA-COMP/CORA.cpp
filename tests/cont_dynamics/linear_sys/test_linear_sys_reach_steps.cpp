// test_linear_sys_reach_steps - linear_sys reach: the time points are exact, the number of steps,
// and a system that does not move

#include "cont_dynamics/linear_sys/linear_sys_testing.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
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
    const ReachSet r = LinearSys(system_matrix(s))
                           .reach({.initial_set = set_of(s), .t_final = 1.0},
                                  alg::Standard{.time_step = 0.2, .taylor_terms = 6});
    for (std::size_t k = 0; k < r.time_point.size(); ++k) {
        const Eigen::MatrixXd flow = (A * 0.2 * double(k)).exp();
        const Eigen::VectorXd back = flow.transpose() * d;
        const double want = back.dot(c) + (back.transpose() * G).cwiseAbs().sum();
        check(close(support(r.time_point[k], d), want, 1e-10),
              b + ": time point " + std::to_string(k));
    }
}

/// Without dynamics nothing moves, so every enclosure is the initial set itself.
void a_zero_system_stays_put(const std::string &b) {
    cora::Rng rng(5);
    const Zonotope X0 = Zonotope::generate_random(2, 3, rng);
    const Eigen::VectorXd d = Eigen::VectorXd::Ones(2);
    for (const std::string &algorithm : kAlgorithms) {
        const ReachSet r = LinearSys(Tensor::zeros({2, 2}))
                               .reach({.initial_set = X0, .t_final = 2.0},
                                      test::lin::algorithm(algorithm, 0.5, 5));
        for (const Zonotope &Z : r.time_interval)
            check(close(support(Z, d), support(X0, d), 1e-12),
                  b + ": a zero system's enclosure grew");
    }
}

/// The last step is a full one even when `t_final` is not a multiple of the step.
void counts_steps_up(const std::string &b) {
    const System &s = matlab_reference::scalar();
    const LinearSys sys(system_matrix(s));
    check(sys.reach({.initial_set = set_of(s), .t_final = 0.25},
                    alg::Standard{.time_step = 0.1, .taylor_terms = 6})
                  .time_interval.size() == 3,
          b + ": ceil(0.25 / 0.1)");
    check(sys.reach({.initial_set = set_of(s), .t_final = 0.3},
                    alg::Standard{.time_step = 0.1, .taylor_terms = 6})
                  .time_interval.size() == 3,
          b + ": 0.3 / 0.1 stays 3");
    check(sys.reach({.initial_set = set_of(s), .t_final = 0.05},
                    alg::Standard{.time_step = 0.1, .taylor_terms = 6})
                  .time_point.size() == 2,
          b + ": one step");
}

/// t_final 0 and time_step 0 are refused; with the checks off the sets are the same.
void checks_and_their_switch() {
    const LinearSys sys = LinearSys(Tensor({{-1.0, 0.5}, {-0.5, -1.0}}));
    const Zonotope X0 = Zonotope(Tensor({1.0, 1.0}), Tensor({{0.1, 0.0}, {0.0, 0.1}}));
    check(
        test::raises(
            [&] { sys.reach({.initial_set = X0, .t_final = 0}, alg::Standard{.time_step = 0.1}); },
            Error::wrong_value),
        "reach: t_final 0");
    check(test::raises(
              [&] { sys.reach({.initial_set = X0, .t_final = 1}, alg::Standard{.time_step = 0}); },
              Error::wrong_value),
          "reach: time_step 0");
    const ReachProblem problem{.initial_set = X0, .t_final = 1};
    const Tensor on =
        sys.reach(problem, alg::Standard{.time_step = 0.1}).time_point.back().center();
    ChecksScope off(false);
    check(
        test::close(sys.reach(problem, alg::Standard{.time_step = 0.1}).time_point.back().center(),
                    on, 0),
        "reach: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        time_points_are_exact(b);
        a_zero_system_stays_put(b);
        counts_steps_up(b);
    });
    checks_and_their_switch();
    return test::finish("linear_sys reach (steps)");
}
