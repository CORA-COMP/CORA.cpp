// test_linear_sys_simulate_random - linear_sys simulate_random: seeded trajectories from any
// initial set

#include "cont_dynamics/linear_sys/linear_sys_testing.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using namespace test::lin;
using matlab_reference::System;

namespace {

/// `simulate_random` draws from the initial set: same seed, same trajectories.
void simulate_random_is_seeded(const std::string &b) {
    const Zonotope X0(Tensor({1.0, 0.0, 0.0}), Tensor({{0.5, 0.0}, {0.0, 0.5}, {0.1, 0.1}}));
    const LinearSys sys(tensor_of(oscillator3()));
    cora::Rng first(5), second(5), other(6);
    const std::vector<Tensor> a = sys.simulate_random({.initial_set = X0, .t_final = 1.0},
                                                      alg::Exact{.time_step = 0.1}, 8, first);
    const std::vector<Tensor> c = sys.simulate_random({.initial_set = X0, .t_final = 1.0},
                                                      alg::Exact{.time_step = 0.1}, 8, second);
    const std::vector<Tensor> d = sys.simulate_random({.initial_set = X0, .t_final = 1.0},
                                                      alg::Exact{.time_step = 0.1}, 8, other);
    check(a.size() == 11 && a[0].shape() == std::vector<int64_t>({3, 8}), b + ": shapes");
    check(close(a.back(), c.back()), b + ": the same seed gives the same trajectories");
    check(!close(a.back(), d.back()), b + ": another seed gives others");
    // The initial set is an interval too: any ContSet works.
    const Interval box(Tensor({0.9, -0.1, -0.1}), Tensor({1.1, 0.1, 0.1}));
    cora::Rng rng(1);
    check(sys.simulate_random({.initial_set = box, .t_final = 0.5}, alg::Exact{.time_step = 0.1}, 5,
                              rng)
                  .size() == 6,
          b + ": simulate_random from an interval");
}

/// No points are refused; with the checks off the draw is the same.
void checks_and_their_switch() {
    const LinearSys sys = LinearSys(Tensor({{-1.0, 0.5}, {-0.5, -1.0}}));
    const RandomSimulationProblem problem{
        .initial_set = Set(Zonotope(Tensor({1.0, 1.0}), Tensor({{0.1, 0.0}, {0.0, 0.1}}))),
        .t_final = 1};
    cora::Rng rng(5), again(5);
    check(test::raises([&] { sys.simulate_random(problem, alg::Exact{.time_step = 0.1}, 0, rng); },
                       Error::wrong_value),
          "simulate_random: no points");
    const Tensor on = sys.simulate_random(problem, alg::Exact{.time_step = 0.1}, 3, rng).back();
    ChecksScope off(false);
    check(test::close(sys.simulate_random(problem, alg::Exact{.time_step = 0.1}, 3, again).back(),
                      on, 0),
          "simulate_random: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) { simulate_random_is_seeded(b); });
    checks_and_their_switch();
    return test::finish("linear_sys simulate_random");
}
