// test_neur_net_contr_sys_simulate_random - runs from random points of the initial set
//
// The runs start in the initial set, the corners come first, the external inputs are drawn from the
// input set, the same seed gives the same runs, and the wrong arguments are refused.
//
// CORA: contDynamics/@contDynamics/simulateRandom.m
// CORA authors: Matthias Althoff, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/rng.h"
#include "neur_net_contr_sys_fixture.h"
#include "testing.h"

using namespace cora;
using namespace fixture;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The starts lie in the initial set, and the first ones are corners of it.
void the_runs_start_in_the_initial_set(const std::string &b) {
    const NeurNetContrSys sys(plant(), network(), 0.05);
    cora::Rng rng(1);
    const ClosedLoopTrajectories runs = sys.simulate_random(X0(), U(), 0.1, 0.05, 10, 0.5, rng);
    check(runs.x[0].shape() == std::vector<int64_t>({2, 10}), b + ": ten runs in two dimensions");
    // every start in the box of X0, the corners on its boundary
    const std::vector<double> start = runs.x[0].data();
    int inside = 0, corners = 0;
    for (int j = 0; j < 10; ++j) {
        const double x1 = start[j], x2 = start[10 + j];
        inside += std::abs(x1 - 1.4) <= 0.05 + 1e-12 && std::abs(x2 - 2.3) <= 0.04 + 1e-12;
        corners += std::abs(std::abs(x1 - 1.4) - 0.05) < 1e-12;
    }
    check(inside == 10, b + ": the starts lie in the initial set");
    check(corners >= 5, b + ": half of the starts are corners");
}

/// The same seed gives the same runs.
void a_seed_repeats_the_runs(const std::string &b) {
    const NeurNetContrSys sys(plant(), network(), 0.05);
    cora::Rng first(7), second(7);
    const ClosedLoopTrajectories a = sys.simulate_random(X0(), U(), 0.1, 0.05, 5, 0.5, first);
    const ClosedLoopTrajectories c = sys.simulate_random(X0(), U(), 0.1, 0.05, 5, 0.5, second);
    check(test::close(a.x.back(), c.x.back(), 0), b + ": the same seed, the same runs");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        the_runs_start_in_the_initial_set(b);
        a_seed_repeats_the_runs(b);
    });
    const NeurNetContrSys sys(plant(), network(), 0.05);
    cora::Rng rng(1);
    check(test::raises([&] { sys.simulate_random(X0(), U(), 0.1, 0.05, 0, 0.5, rng); },
                       Error::wrong_value),
          "simulate_random: no point");
    check(test::raises([&] { sys.simulate_random(X0(), std::nullopt, 0.1, 0.05, 3, 0.5, rng); },
                       Error::wrong_value),
          "simulate_random: the external input set is missing");
    ChecksScope off(false);
    check(sys.simulate_random(X0(), U(), 0.1, 0.05, 3, 0.5, rng).x.size() == 3,
          "simulate_random: with the checks off");
    return test::finish("neurNetContrSys simulate_random");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
