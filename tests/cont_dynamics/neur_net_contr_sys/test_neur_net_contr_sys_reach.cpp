// test_neur_net_contr_sys_reach - the reach of a neural network controlled system
//
// A van der Pol oscillator with a controlled and an external input and a tanh network: the sets
// of reach, on zonotopes and on polynomial zonotopes, hold every simulated closed-loop run at the
// times of the steps; a specification stops the run at the first step that violates it; wrong
// inputs are refused, and the checks can be switched off.
//
// CORA: contDynamics/@neurNetContrSys/reach.m
// CORA: contDynamics/@neurNetContrSys/neurNetContrSys.m
// PROPERTY: contDynamics/@neurNetContrSys/reach.m
// CORA authors: Niklas Kochdumper, Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/polytope/polytope.h"
#include "cora/global/rng.h"
#include "neur_net_contr_sys_fixture.h"
#include "testing.h"

using namespace cora;
using namespace fixture;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// How many simulated points lie outside the box of the set at their time and over their step.
template <class S>
int outside(const ClosedLoopReach<S> &R, const ClosedLoopTrajectories &traj) {
    int count = 0;
    for (std::size_t k = 1; k < traj.x.size(); ++k) {
        // the k-th sample is at the end of the step k - 1
        const Interval at = R.time_point[k].interval(), over = R.time_interval[k - 1].interval();
        for (const Interval *box : {&at, &over}) {
            const std::vector<double> lo = box->inf().data(), hi = box->sup().data(),
                                      p = traj.x[k].data();
            const std::size_t N = p.size() / lo.size();
            for (std::size_t j = 0; j < N; ++j)
                for (std::size_t i = 0; i < lo.size(); ++i)
                    count += !(p[i * N + j] >= lo[i] - 1e-9 && p[i * N + j] <= hi[i] + 1e-9);
        }
    }
    return count;
}

/// The sets hold the runs from random points and corners with random external inputs.
void the_sets_hold_every_run(const std::string &b) {
    const NeurNetContrSys sys(plant(), network(), 0.05);
    cora::Rng rng(1);
    const ClosedLoopTrajectories traj = sys.simulate_random(X0(), U(), 0.5, 0.025, 30, 0.5, rng);
    // the zonotope reach
    const ClosedLoopReach<Zonotope> Z = sys.reach(X0(), U(), 0.5, options());
    check(Z.completed && Z.time_interval.size() == 20, b + ": the zonotope reach completes");
    check(outside(Z, traj) == 0, b + ": " + std::to_string(outside(Z, traj)) + " points left Z");
    // the polynomial zonotope reach
    const ClosedLoopReach<PolyZonotope> P = sys.reach(PolyZonotope(X0()), U(), 0.5, options());
    check(P.completed && P.time_interval.size() == 20, b + ": the polynomial reach completes");
    check(outside(P, traj) == 0, b + ": " + std::to_string(outside(P, traj)) + " points left P");
}

/// A specification that the sets violate stops the run at that step; a loose one does not.
void the_specification_stops_the_run(const std::string &b) {
    const NeurNetContrSys sys(plant(), network(), 0.05);
    const Polytope tight(Tensor({{1.0, 0.0}}), Tensor({1.0}));
    const Polytope loose(Tensor({{1.0, 0.0}}), Tensor({10.0}));
    const ClosedLoopReach<Zonotope> stopped =
        sys.reach(X0(), U(), 0.5, options(), TimedSpecification{Specification::safe_set(tight)});
    check(stopped.violated && !stopped.completed && stopped.time.back() < 0.1,
          b + ": the run stops at the first step");
    const ClosedLoopReach<Zonotope> fine =
        sys.reach(X0(), U(), 0.5, options(), TimedSpecification{Specification::safe_set(loose)});
    check(fine.completed && !fine.violated, b + ": a loose specification does not stop it");
    // a window after the horizon never applies
    const ClosedLoopReach<Zonotope> later =
        sys.reach(X0(), U(), 0.5, options(),
                  TimedSpecification{Specification::safe_set(tight), std::make_pair(2.0, 3.0)});
    check(later.completed, b + ": a window after the horizon does not apply");
}

/// Wrong inputs are refused with their identifiers.
void wrong_arguments_are_described() {
    const NeurNetContrSys sys(plant(), network(), 0.05);
    check(test::raises([&] { sys.reach(X0(), std::nullopt, 0.5, options()); }, Error::wrong_value),
          "reach: an external input set is needed");
    check(test::raises([&] { sys.reach(X0(), U(), -1.0, options()); }, Error::wrong_value),
          "reach: a negative horizon");
    const Zonotope small(Tensor({1.0}), Tensor({{0.1}}));
    check(test::raises([&] { sys.reach(small, U(), 0.5, options()); }, Error::dimension_mismatch),
          "reach: an initial set of another dimension");
    check(test::raises([&] { NeurNetContrSys(plant(), network(), 0.0); },
                       Error::wrong_input_in_constructor),
          "constructor: a sampling time of 0");
    // a network with more outputs than the plant has inputs
    const NeuralNetwork wide(
        {{Tensor({{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}}), Tensor({0.0, 0.0, 0.0})}});
    check(test::raises([&] { NeurNetContrSys(plant(), wide, 0.05); },
                       Error::wrong_input_in_constructor),
          "constructor: more outputs than inputs");
}

/// The same sets with the checks off.
void checks_and_their_switch() {
    const NeurNetContrSys sys(plant(), network(), 0.05);
    const ClosedLoopReach<Zonotope> on = sys.reach(X0(), U(), 0.2, options());
    ChecksScope off(false);
    const ClosedLoopReach<Zonotope> without = sys.reach(X0(), U(), 0.2, options());
    check(test::close(on.time_point.back().center(), without.time_point.back().center(), 0),
          "reach: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        the_sets_hold_every_run(b);
        the_specification_stops_the_run(b);
    });
    wrong_arguments_are_described();
    checks_and_their_switch();
    return test::finish("neurNetContrSys reach");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
