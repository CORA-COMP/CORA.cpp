// test_linear_sys_trajectory_error - the error of a linearized trajectory
//
// At the expansion point only the remainder of the series is left.
//
// CORA: contDynamics/@linearSys/trajectoryError.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/linear_sys_steps_testing.h"

using namespace cora;
using namespace test::steps;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

Tensor matrix() { return Tensor({{-1.0, 2.0}, {-2.0, -1.0}}); }
constexpr double kTime = 0.3;
constexpr int kOrder = 12;

Zonotope start_set() { return {test::column({1.0, 0.5}), Tensor({{0.1, 0.0}, {0.0, 0.1}})}; }

/// For t_min = t_max = t_c = t and a point start set every term of the series cancels but the
/// remainder: the error set is a few 1e-11, c_tilde is (e^{At} - I - At)' n and d_tilde is n'
/// (e^{At} - I - At) x0.
void at_the_expansion_point_only_the_series_remains(const std::string &b) {
    const Tensor n = test::column({1.0, -0.5}), zero = test::column({0.0, 0.0});
    const TrajectoryError e = LinearSys(matrix()).trajectory_error({zero, zero}, zero, start_set(),
                                                                   kTime, kTime, kTime, kOrder, n);
    const Zonotope point(start_set().center(), Tensor::zeros({2, 0}));
    const TrajectoryError p = LinearSys(matrix()).trajectory_error({zero, zero}, zero, point, kTime,
                                                                   kTime, kTime, kOrder, n);
    check(support(p.error, {1, 1}) < 1e-8 && support(p.error, {-1, 1}) < 1e-8,
          b + ": the error set");
    const Tensor beyond = (matrix() * kTime).expm() - Tensor::eye(2) - matrix() * kTime;
    const Tensor c_center = (e.c_tilde.inf() + e.c_tilde.sup()) * 0.5;
    check(test::close(c_center, beyond.matrix_transpose().matmul(n), 1e-9), b + ": c_tilde");
    const double d_center = ((e.d_tilde.inf() + e.d_tilde.sup()) * 0.5).item();
    check(test::close(d_center,
                      n.matrix_transpose().matmul(beyond).matmul(start_set().center()).item(),
                      1e-9),
          b + ": d_tilde");
    // one value of the flow stands for both
    const TrajectoryError same = LinearSys(matrix()).trajectory_error(
        {zero}, zero, start_set(), kTime, kTime, kTime, kOrder, n);
    check(test::close(same.error.generators(), e.error.generators(), 0), b + ": one flow value");
}

/// A wider time interval and a difference of the flow values widen the error set.
void the_time_and_the_flow_widen_the_error(const std::string &b) {
    const LinearSys sys(matrix());
    const Tensor n = test::column({1.0, 0.0}), zero = test::column({0.0, 0.0});
    const auto width = [&](double t_min, double t_max, const Tensor &f2) {
        const Zonotope e =
            sys.trajectory_error({zero, f2}, zero, start_set(), t_min, t_max, kTime, kOrder, n)
                .error;
        return support(e, {1, 0}) + support(e, {-1, 0});
    };
    const double point = width(kTime, kTime, zero), interval = width(0.2, 0.4, zero);
    check(interval > point + 1e-4, b + ": a wider time interval");
    // the flow gap f1 - f2 = (-0.1, 0) is scaled by [0.2, 0.4]: a generator of 0.1 * 0.1
    check(width(0.2, 0.4, test::column({0.1, 0.0})) > interval + 0.015,
          b + ": the flow values differ");
}

/// An order of zero, a third flow value and a vector of the wrong size are refused.
void checks_and_their_switch(const std::string &b) {
    const LinearSys sys(matrix());
    const Tensor n = test::column({1.0, 0.0}), zero = test::column({0.0, 0.0});
    const auto call = [&](const std::vector<Tensor> &f, int order, const Tensor &u) {
        return sys.trajectory_error(f, u, start_set(), kTime, kTime, kTime, order, n);
    };
    check(test::raises([&] { call({zero}, 0, zero); }, Error::wrong_value), b + ": order 0");
    check(test::raises([&] { call({zero, zero, zero}, 3, zero); }, Error::wrong_value),
          b + ": three flow values");
    check(test::raises([&] { call({zero}, 3, test::column({0.0})); }, Error::dimension_mismatch),
          b + ": u of one row");
    const Tensor on = call({zero}, 3, zero).error.center();
    ChecksScope off(false);
    check(test::close(call({zero}, 3, zero).error.center(), on, 0),
          b + ": the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        at_the_expansion_point_only_the_series_remains(b);
        the_time_and_the_flow_widen_the_error(b);
        checks_and_their_switch(b);
    });
    return test::finish("linear_sys trajectory_error");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
