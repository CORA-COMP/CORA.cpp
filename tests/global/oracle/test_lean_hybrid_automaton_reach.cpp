// test_lean_hybrid_automaton_reach - hybrid reach against the CORALean oracle
//
// With time_point_swap and check_flow off and linear_step coralean, HybridAutomaton::reach is the
// oracle's `hybridAutomaton.reach`: the same branches in the same order, the same time-interval
// sets bit for bit, the same start times. The automata are small random ones in two states: a
// drift towards hyperplane guards, halfspace invariants that end each stay, linear resets, a
// guard that two stays cross at different times and a clock-like location. The invariant exit
// follows CORALean (a row of the invariant lies strictly beyond the set). With
// CORACPP_ORACLE_SLOW the rendezvous instances SRA01, SRNA01 and SRU01 run as well (minutes).
//
// ORACLE: hybridDynamics/@hybridAutomaton/reach.m
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#include "../../../examples/cpp/benchmarks/benchmark_rendezvous_model.h"
#include "cora/global/oracle/hybrid_automaton.h"
#include "cora/hybrid_dynamics/hybrid_automaton/hybrid_automaton.h"
#include "oracle_testing.h"

#include <cstdlib>

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The hyperplane normal'x = d.
Polytope aux_hyperplane(const std::vector<double> &normal, double d) {
    const auto n = int64_t(normal.size());
    return {Tensor::zeros({0, n}), Tensor::zeros({0, 1}), Tensor::from_data(normal, {1, n}),
            Tensor::from_data({d}, {1, 1})};
}

/// The halfspace normal'x <= d.
Polytope aux_halfspace(const std::vector<double> &normal, double d) {
    return {Tensor::from_data(normal, {1, int64_t(normal.size())}), Tensor::from_data({d}, {1, 1})};
}

/// One transition: guard, reset x -> W x + w and target.
struct Edge {
    Polytope guard;
    Tensor W, w;
    int64_t target;
};

/// A location: the flow x' = A x + c, the invariant, the edges and the time step.
struct Spec {
    Tensor A, c;
    Polytope invariant;
    std::vector<Edge> edges;
    double dt;
};

/// The automaton of the locations.
HybridAutomaton aux_build(const std::vector<Spec> &specs) {
    std::vector<Location> locations;
    int k = 0;
    for (const Spec &s : specs) {
        std::vector<Transition> transitions;
        for (const Edge &e : s.edges)
            transitions.emplace_back(e.guard, LinearReset(e.W, std::nullopt, e.w), e.target);
        const int64_t n = s.A.shape()[0];
        locations.emplace_back("l" + std::to_string(k++), s.invariant, transitions,
                               LinearSys(s.A, Tensor::zeros({n, 1})), s.c);
    }
    return HybridAutomaton("random", locations);
}

/// A small matrix of entries in [-scale, scale].
Tensor aux_small(test::oracle::Random &rnd, int64_t n, double scale) {
    return rnd.matrix(n, n) * scale;
}

/// The automaton of the trial: location 0 drifts along x1 and is left through x1 = 2 (location 1)
/// and through x2 = 1.2 (location 2); location 1 is left through x2 = 3.5 into location 2, which
/// runs to the horizon. Every stay ends at its invariant.
std::vector<Spec> aux_specs(test::oracle::Random &rnd, double dt) {
    const Tensor I = Tensor::eye(2);
    const auto drift = [&](double a, double b) { return Tensor::from_data({a, b}, {2, 1}); };
    std::vector<Spec> specs;
    specs.push_back({aux_small(rnd, 2, 0.1),
                     drift(1.0, 0.25),
                     aux_halfspace({1, 0}, 2.6),
                     {{aux_hyperplane({1, 0}, 2.0), I * 0.9, drift(0.1, 0.0), 1},
                      {aux_hyperplane({0, 1}, 1.2), I, drift(0.0, -0.5), 2}},
                     dt});
    specs.push_back(
        {aux_small(rnd, 2, 0.1),
         drift(0.1, 1.0),
         aux_halfspace({0, 1}, 4.0),
         {{aux_hyperplane({0, 1}, 3.5), aux_small(rnd, 2, 0.5) + I, drift(0.0, 0.0), 2}},
         dt});
    specs.push_back({aux_small(rnd, 2, 0.1), drift(0.2, 0.2), Polytope::inf(2), {}, dt});
    return specs;
}

/// CORA.cpp's reach of the automaton without CORA's two unsound steps against the oracle's, branch
/// by branch: the location, the producing branch, the start times and every set bit for bit.
std::pair<std::size_t, std::size_t>
aux_compare(const HybridAutomaton &HA, const HybridReachProblem &problem, alg::HybridStandard alg) {
    alg.linear_step = alg::LinearStep::coralean;
    alg.check_flow = false;
    alg.time_point_swap = false;
    const HybridReachSet mine = HA.reach(problem, alg);
    const std::vector<lean::HybridBranch> theirs = lean::HybridAutomaton::reach(HA, problem, alg);
    test::oracle::check("branches", int64_t(mine.branches.size()), int64_t(theirs.size()));
    std::size_t sets = 0;
    for (std::size_t b = 0; b < theirs.size() && b < mine.branches.size(); ++b) {
        const LocationReachSet &m = mine.branches[b];
        const lean::HybridBranch &t = theirs[b];
        const std::string name = "branch " + std::to_string(b);
        test::oracle::check(name + ".location", m.location, t.location);
        test::oracle::check(name + ".parent", m.parent, t.parent.value_or(-1));
        test::oracle::check(name + ".sets", int64_t(m.time_interval.size()),
                            int64_t(t.time_interval.size()));
        // the times of a step size that is not a power of two are rounded in the oracle's answer
        for (const int k : {0, 1})
            test::check(std::abs(m.time_point_times.front()[std::size_t(k)] -
                                 t.t_start[std::size_t(k)]) <= 1e-9,
                        name + ": the start time");
        for (std::size_t k = 0; k < t.time_interval.size() && k < m.time_interval.size(); ++k) {
            const std::string set = name + ".R[" + std::to_string(k) + "]";
            const auto &Z = m.time_interval[k].as<Zonotope>();
            test::oracle::check(set + ".c", Z.center(), t.time_interval[k].center());
            test::oracle::check(set + ".G", Z.generators(), t.time_interval[k].generators());
        }
        sets += t.time_interval.size();
    }
    return {theirs.size(), sets};
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("hybridAutomaton.reach", [](test::oracle::Random &rnd) {
        std::size_t stays = 0, sets = 0;
        // small random automata: three locations, two guards of the first, a step size 1/8 and 1/16
        for (const double dt : {0.125, 0.0625}) {
            const HybridAutomaton HA = aux_build(aux_specs(rnd, dt));
            const Zonotope R0(Tensor::from_data({0.0, 0.0}, {2, 1}), rnd.matrix(2, 2) * 0.05);
            alg::HybridStandard alg{{dt}, 6, 8};
            const auto [n, m] = aux_compare(HA, {.initial_set = R0, .t_final = 6.0}, alg);
            stays += n;
            sets += m;
        }
        test::check(stays >= 6 && sets > 100, "hybridAutomaton.reach: the stays and their sets");
        // the rendezvous instances that the op serves, with coarse steps (the oracle is slow)
        if (std::getenv("CORACPP_ORACLE_SLOW") != nullptr)
            for (const char *name : {"SRA01", "SRNA01", "SRU01"}) {
                auto [problem, alg] = cora::bench::rendezvous::options(name);
                // a start near the docking guard x = -100 and a short horizon
                problem.initial_set =
                    Zonotope(Tensor::from_data({-120, -30, 0, 0, 0}, {5, 1}),
                             Tensor::from_data({1, 0, 0, 0, 0, 0, 1, 0, 0, 0}, {5, 2}));
                problem.t_final = 10;
                alg.time_step = std::vector<double>(alg.time_step.size() == 3 ? 3 : 2, 0.25);
                alg.time_step[0] = 0.5;
                const auto [n, m] = aux_compare(cora::bench::rendezvous::automaton(
                                                    name, cora::bench::rendezvous::variant(name)),
                                                problem, alg);
                test::check(n >= 2 && m > 50, std::string("hybridAutomaton.reach: ") + name);
            }
    });
    return test::finish("lean hybrid automaton reach");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
