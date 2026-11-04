// test_nonlinear_sys_simulate_random - nonlinearSys simulate_random: points of the set, repeatable
//
// CORA: contDynamics/@contDynamics/simulateRandom.m

#include "cora/cont_dynamics/nonlinear_sys/nonlinear_sys.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

NonlinearSys decay() {
    return NonlinearSys(
        [](const std::vector<Expr> &x) { return std::vector<Expr>{-x[0], -2 * x[1]}; }, 2);
}

void the_start_points_lie_in_the_set_and_repeat(const std::string &b) {
    const Zonotope X0(Tensor({1.0, 2.0}), Tensor({{0.1, 0.0}, {0.0, 0.2}}));
    cora::Rng first(3), second(3);
    const std::vector<Tensor> a = decay().simulate_random(
        {.initial_set = X0, .t_final = 0.5}, alg::RungeKutta4{.time_step = 0.1}, 6, first);
    const std::vector<Tensor> c = decay().simulate_random(
        {.initial_set = X0, .t_final = 0.5}, alg::RungeKutta4{.time_step = 0.1}, 6, second);
    check(a.size() == 6 && a[0].shape() == std::vector<int64_t>({2, 6}), b + ": six columns");
    check(test::close(a.back(), c.back(), 0.0), b + ": the same seed gives the same points");
    const std::vector<double> start = a[0].data();
    bool inside = true;
    for (int j = 0; j < 6; ++j)
        inside = inside && std::abs(start[j] - 1.0) <= 0.1 && std::abs(start[6 + j] - 2.0) <= 0.2;
    check(inside, b + ": the start points are points of the set");
}

/// No points are refused; with the checks off the draw is the same.
void checks_and_their_switch() {
    const NonlinearSys sys = NonlinearSys(
        [](const std::vector<Expr> &x) {
            {
                return std::vector<Expr>{{x[1], -x[0]}};
            }
        },
        2);
    const RandomSimulationProblem problem{
        .initial_set = Set(Zonotope(Tensor({1.0, 1.0}), Tensor({{0.1, 0.0}, {0.0, 0.1}}))),
        .t_final = 1};
    cora::Rng rng(6), again(6);
    check(test::raises(
              [&] { sys.simulate_random(problem, alg::RungeKutta4{.time_step = 0.1}, 0, rng); },
              Error::wrong_value),
          "simulate_random: no points");
    const Tensor on =
        sys.simulate_random(problem, alg::RungeKutta4{.time_step = 0.1}, 2, rng).back();
    ChecksScope off(false);
    check(test::close(
              sys.simulate_random(problem, alg::RungeKutta4{.time_step = 0.1}, 2, again).back(), on,
              0),
          "simulate_random: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend(
        [](const std::string &b) { the_start_points_lie_in_the_set_and_repeat(b); });
    checks_and_their_switch();
    return test::finish("nonlinearSys simulate_random");
}
