// test_neur_net_contr_sys_simulate - the runs of a neural network controlled system
//
// The network output is held over a control period: runs sampled at two rates agree at the common
// times, a horizon that is no multiple of the sampling ends with a shorter period, and a missing
// external input is refused.
//
// CORA: contDynamics/@neurNetContrSys/simulate.m
// CORA authors: Niklas Kochdumper, Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "neur_net_contr_sys_fixture.h"
#include "testing.h"

using namespace cora;
using namespace fixture;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Two sampling rates give the same state, and an odd horizon ends with a shorter period.
void the_output_is_held_over_a_period(const std::string &b) {
    const NeurNetContrSys sys(plant(), network(), 0.05);
    const Tensor x0({{1.4, 1.5}, {2.3, 2.1}}), u({{0.0, 0.01}});
    const ClosedLoopTrajectories fine = sys.simulate(x0, u, 0.1, 0.025);
    const ClosedLoopTrajectories coarse = sys.simulate(x0, u, 0.1, 0.05);
    check(test::close(fine.x[4], coarse.x[2], 1e-6), b + ": the same state after 0.1");
    // a horizon that is no multiple of the sampling
    const ClosedLoopTrajectories odd = sys.simulate(x0, u, 0.12, 0.05);
    check(odd.time.back() == 0.12 && odd.x.size() == 4, b + ": the last period is shorter");
}

/// The output of a control period comes from the state at its start: a state on the flow of the
/// open loop with that output held is reproduced.
void a_period_holds_the_output_of_its_start(const std::string &b) {
    const NeurNetContrSys sys(plant(), network(), 0.05);
    const Tensor x0({{1.4}, {2.3}});
    const Tensor held = Tensor::concat({network().evaluate(x0), Tensor({{0.0}})}, 0);
    const ClosedLoopTrajectories run = sys.simulate(x0, Tensor({{0.0}}), 0.05, 0.05);
    const Tensor open = plant()
                            .simulate({.initial_state = x0, .input = held, .t_final = 0.05},
                                      alg::RungeKutta4{.time_step = 0.05})
                            .back();
    check(test::close(run.x.back(), open, 1e-9), b + ": the open loop with the output held");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        the_output_is_held_over_a_period(b);
        a_period_holds_the_output_of_its_start(b);
    });
    const NeurNetContrSys sys(plant(), network(), 0.05);
    check(test::raises([&] { sys.simulate(Tensor({{1.0}, {2.0}}), std::nullopt, 0.1, 0.05); },
                       Error::dimension_mismatch),
          "simulate: the external input is missing");
    check(test::raises([&] { sys.simulate(Tensor({{1.0}, {2.0}}), Tensor({{0.0}}), -1.0, 0.05); },
                       Error::wrong_value),
          "simulate: a negative horizon");
    ChecksScope off(false);
    check(sys.simulate(Tensor({{1.0}, {2.0}}), Tensor({{0.0}}), 0.1, 0.05).x.size() == 3,
          "simulate: with the checks off");
    return test::finish("neurNetContrSys simulate");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
