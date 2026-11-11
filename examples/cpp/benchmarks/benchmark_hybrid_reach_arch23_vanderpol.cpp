// benchmark_hybrid_reach_arch23_vanderpol - coupled van der Pol oscillators (CVDP23)
//
// MATLAB CORA's benchmark_nonlinear_reach_ARCH23_vanDerPol.m with the model of
// models/Cora/coupledVanDerPol/coupledVanDerPol_ARCH23.m, numbers copied: two van der Pol
// oscillators (x1, x2) and (x3, x4) coupled by the parameter x5, a state with x5' = 0. The
// hybrid automaton has three locations on both sides of the plane c'x = d (the dynamics are the
// same in all of them, the plane only splits the analysis); the initial set is split into 13 boxes,
// each a polynomial zonotope, and the specification is x2 < 2.75 and x4 < 2.75 for all of them.
// Algorithm "poly" with tensor order 3, guards by zonoGirard and PCA. The 13 runs are timed
// together.
//
// Syntax:   build/core/benchmarks/benchmark_hybrid_reach_arch23_vanderpol [CVDP23]
// Outputs:  one line 'benchmark,instance,result,time', as the MATLAB script
// See also: benchmark_hybrid.h, benchmark_hybrid_reach_arch23_spacecraft.cpp
//
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "benchmark_hybrid.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using namespace cora;
using cora::bench::column;
using cora::bench::hyperplane;
using cora::bench::matrix;

/// The dynamics of the model: x1' = x2, x2' = (1 - x1^2) x2 + x5 (x3 - x1) - x1, and the same
/// for (x3, x4) with the coupling reversed, x5' = 0; the input u is not used.
std::vector<Expr> aux_dynamics(const std::vector<Expr> &x, const std::vector<Expr> &) {
    const Expr coupling = x[4] * (x[2] - x[0]);
    return {x[1], (1 - pow(x[0], 2)) * x[1] + coupling - x[0], x[3],
            (1 - pow(x[2], 2)) * x[3] - coupling - x[2], Expr(0.0)};
}

/// The automaton: the plane c'x = d splits the state space, and the jumps map by the identity.
HybridAutomaton aux_automaton() {
    const std::vector<double> c = {-0.9808, -0.7286, -0.9808, -0.7286, 0};
    const double d = -0.9811;
    std::vector<double> minus_c;
    minus_c.reserve(c.size());
    for (const double v : c) minus_c.push_back(-v);
    const NonlinearSys sys(aux_dynamics, 5, 1);
    const LinearReset id(Tensor::eye(5), std::nullopt, column({0, 0, 0, 0, 0}));
    Location loc1("loc1", Polytope(matrix(1, 5, c), column({d})),
                  {Transition(hyperplane(c, d), id, 1)}, sys);
    Location loc2("loc2", Polytope(matrix(1, 5, minus_c), column({-d})),
                  {Transition(hyperplane(minus_c, -d), id, 2)}, sys);
    Location loc3("loc3", Polytope(matrix(1, 5, {1, 0, 0, 0, 0}), column({1000})), {}, sys);
    return HybridAutomaton("vanDerPol_hybrid", {loc1, loc2, loc3});
}

/// The 13 boxes of the initial set: (x1, x2) and (x3, x4) in I1 or I2, x5 in a part of [1, 3].
std::vector<Interval> aux_initial_sets() {
    const auto box = [](double lo1, double hi1, double lo2, double hi2, double x5_lo,
                        double x5_hi) {
        return Interval(column({lo1, lo2, lo1, lo2, x5_lo}), column({hi1, hi2, hi1, hi2, x5_hi}));
    };
    const std::vector<std::array<double, 2>> x5 = {
        {1, 1.4}, {1.4, 1.8}, {1.8, 2.1}, {2.1, 2.3}, {2.3, 2.45}, {2.45, 2.66}, {2.66, 3}};
    const std::vector<std::array<double, 2>> x5_2 = {{1, 1.4},    {1.4, 1.8},   {1.8, 2.2},
                                                     {2.2, 2.43}, {2.43, 2.66}, {2.66, 3}};
    std::vector<Interval> sets;
    sets.reserve(x5.size() + x5_2.size());
    // (x1, x2) in [1.25, 1.4] x [2.35, 2.45] for the first seven, [1.4, 1.55] x ... for the rest
    for (const auto &b : x5) sets.push_back(box(1.25, 1.4, 2.35, 2.45, b[0], b[1]));
    for (const auto &b : x5_2) sets.push_back(box(1.4, 1.55, 2.35, 2.45, b[0], b[1]));
    return sets;
}

/// The specifications: x2 < 2.75 and x4 < 2.75 (the unsafe sets -x2 <= -2.75 and -x4 <= -2.75).
std::vector<HybridSpecification> aux_specs() {
    return {
        {Specification::unsafe_set(Polytope(matrix(1, 5, {0, -1, 0, 0, 0}), column({-2.75}))), {}},
        {Specification::unsafe_set(Polytope(matrix(1, 5, {0, 0, 0, -1, 0}), column({-2.75}))), {}}};
}

/// The options of the script.
alg::HybridStandard aux_alg() {
    alg::HybridStandard a;
    a.time_step = {0.005};
    a.taylor_terms = 3;
    a.zonotope_order = 100;
    a.tensor_order = 3;
    a.error_order = 10;
    a.intermediate_order = 50;
    a.polynomialization = true;
    a.guard_intersect = alg::ZonoGirard{{alg::Enclose::pca}};
    return a;
}

/// Runs the 13 boxes in turn and prints the line of the script: the verdict is all of them.
bool aux_run(const std::string &name) {
    if (name != "CVDP23") CORA_FAIL(Error::wrong_value, "vanderpol: unknown instance " + name);
    const HybridAutomaton HA = aux_automaton();
    const alg::HybridStandard a = aux_alg();
    const std::vector<HybridSpecification> specs = aux_specs();
    const auto start = std::chrono::steady_clock::now();
    bool all_verified = true;
    std::size_t index = 0;
    for (const Interval &box : aux_initial_sets()) {
        HybridReachProblem problem;
        problem.initial_poly_set = PolyZonotope(box);
        problem.start_location = 0;
        problem.t_final = 7;
        const HybridReachSet R = HA.reach(problem, a, specs);
        std::cerr << "  split " << ++index << "/13: " << (R.verified ? "verified" : "not verified")
                  << '\n';
        all_verified = all_verified && R.verified;
        // with CORACPP_BENCHMARK_VERBOSE, each branch: its location, sets and times
        if (std::getenv("CORACPP_BENCHMARK_VERBOSE"))
            for (const LocationReachSet &b : R.branches)
                std::cerr << "    location " << b.location << ": " << b.time_interval.size()
                          << " sets from " << b.time_point_times.front()[0] << " to "
                          << b.time_point_times.back()[1] << '\n';
    }
    const std::chrono::duration<double> wall = std::chrono::steady_clock::now() - start;
    cora::bench::print_line("CVDP23", "", all_verified, wall.count());
    return all_verified;
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main(int argc, char **argv) { return cora::bench::run_family(argc, argv, {"CVDP23"}, aux_run); }

// ---------------------------------------  END OF CODE  ---------------------------------------- //
