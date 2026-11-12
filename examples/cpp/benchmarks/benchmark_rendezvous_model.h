// benchmark_rendezvous_model - the spacecraft rendezvous automata of ARCH 2023 and their options
//
// The instances of MATLAB CORA's benchmark_hybrid_reach_ARCH23_rendezvous_<instance>.m with the
// automata of models/Cora/spacecraftRendezvous/rendezvous_<instance>.m, numbers copied. The state
// is (x, y, vx, vy, t); the chaser approaches (location 0), docks (1) or aborts into the passive
// mode (2). The instances differ by the abort times, the guard intersection and the specs. Shared
// by the benchmark program and the test against the CORALean oracle.
//
// Syntax:   auto HA = cora::bench::rendezvous::automaton("SRA01", variant("SRA01"));
//           auto [problem, alg] = cora::bench::rendezvous::options("SRA01");
// See also: benchmark_hybrid_reach_arch23_rendezvous.cpp, benchmark_hybrid.h
//
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   12-November-2026
// Reviewed:      ---

#pragma once

#include "benchmark_hybrid.h"

#include <array>
#include <cmath>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::bench::rendezvous {

using namespace cora;

/// The time windows [lower, upper] of the abort guards.
using Windows = std::vector<std::array<double, 2>>;

/// What the models of the instances differ in.
struct Variant {
    bool timed = true;              ///< false: SRNA01, no clock in the invariants
    bool x_first = false;           ///< the approach invariant lists x <= -100 before t
    double t_u = 0;                 ///< the invariants of approach and docking: t <= t_u
    std::optional<double> abort_at; ///< aborts on the hyperplane t = abort_at, else windows
    Windows approach_aborts, docking_aborts;
    double passive_b = 0; ///< the passive invariant -t <= passive_b
};

/// The variant of each instance, from its model file.
inline Variant variant(const std::string &name) {
    // the windows of SRA07, SRA08 and SRU02 in the docking mode
    auto docking = [](double t_l, double t_u) {
        return Windows{{t_l, 110}, {110, 112}, {112, 115}, {115, 120}, {120, 135}, {135, t_u}};
    };
    const Windows early{{0, 0.1}, {0.1, 0.15}, {0.15, 0.2}, {0.2, 3}};
    // timed, x first, t_u, the abort hyperplane or windows, the passive bound
    if (name == "SRA01") return {true, false, 125, 120.0, {}, {}, -120};
    if (name == "SRA02") return {true, false, 125, std::nullopt, {{120, 125}}, {{120, 125}}, -120};
    if (name == "SRA03") return {true, false, 145, std::nullopt, {{120, 145}}, {{120, 145}}, -120};
    if (name == "SRA04") return {true, true, 240, 240.0, {}, {}, -240};
    if (name == "SRA05") return {true, false, 240, std::nullopt, {{235, 240}}, {{235, 240}}, -120};
    if (name == "SRA06") return {true, false, 240, std::nullopt, {{230, 240}}, {{230, 240}}, -120};
    if (name == "SRA07")
        return {true, false, 150, std::nullopt, {{50, 120}, {120, 150}}, docking(50, 150), 40};
    Windows approach = early;
    if (name == "SRA08" || name == "SRU02") {
        const double t_u = name == "SRA08" ? 240 : 260;
        approach.push_back({3, t_u});
        return {true, false, t_u, std::nullopt, approach, docking(0, t_u), 40};
    }
    if (name == "SRU01") return {true, true, 260, 260.0, {}, {}, -260};
    if (name == "SRNA01") return {false, false, 0, std::nullopt, {}, {}, 0};
    CORA_FAIL(Error::wrong_value, "rendezvous: unknown instance " + name);
}

/// The guard t in [lower, upper] (CORA's interval with the other states within dist = 1e5).
inline Interval window(const std::array<double, 2> &w) {
    const double dist = 1e5;
    return {column({-dist, -dist, -dist, -dist, w[0]}), column({dist, dist, dist, dist, w[1]})};
}

/// The aborts of a mode into the passive one (location 2), all with the identity reset.
inline std::vector<Transition> aborts(const Variant &v, const Windows &windows,
                                      const LinearReset &id) {
    std::vector<Transition> out;
    if (v.abort_at) out.emplace_back(hyperplane({0, 0, 0, 0, -1}, -*v.abort_at), id, 2);
    for (const auto &w : windows) out.emplace_back(window(w), id, 2);
    return out;
}

/// A mode's flow: x' = vx, y' = vy, the velocities by the rows a3 and a4, and t' = 0 (from c).
inline Tensor flow(const std::vector<double> &a3, const std::vector<double> &a4) {
    std::vector<double> e = {0, 0, 1, 0, 0, 0, 0, 0, 1, 0};
    for (const auto *row : {&a3, &a4}) {
        e.insert(e.end(), row->begin(), row->end());
        e.push_back(0);
    }
    e.resize(25, 0.0);
    return matrix(5, 5, e);
}

inline HybridAutomaton automaton(const std::string &name, const Variant &v) {
    // the dynamics of the three modes, with t' = 1 as the constant input c
    const Tensor A1 = flow({-0.05759976588, 0.0002009598965, -2.89995084, 0.008772008945},
                           {-0.0001740313574, -0.06651239849, -0.008753511055, -2.903002693});
    const Tensor A2 = flow({-0.5759999431, 0.0002624860794, -19.22997959, 0.008762759318},
                           {-0.0002624860807, -0.5759999402, -0.008762760682, -19.2299766});
    const Tensor A3 = flow({5.758947211E-05, 0, 0, 0.00876276}, {0, 0, -0.00876276, 0});
    const Tensor B = column({0, 0, 0, 0, 0}), c = column({0, 0, 0, 0, 1});
    const LinearReset id(Tensor::eye(5), std::nullopt, column({0, 0, 0, 0, 0}));
    // the line of sight x >= -100 ends the approach
    const Polytope docking_guard = hyperplane({-1, 0, 0, 0, 0}, 100);
    // the docking region: |x|, |y| <= 100 and |x +- y| <= 141.1
    const std::vector<double> region = {0, -1, 0, 0, 0, -1, -1, 0, 0,  0, -1, 0, 0, 0,
                                        0, -1, 1, 0, 0, 0,  0,  1, 0,  0, 0,  1, 1, 0,
                                        0, 0,  1, 0, 0, 0,  0,  1, -1, 0, 0,  0};
    const std::vector<double> region_b = {100, 141.1, 100, 141.1, 100, 141.1, 100, 141.1};
    // SRNA01: no clock, no aborts
    if (!v.timed) {
        Location approach("S1", Polytope(matrix(1, 5, {1, 0, 0, 0, 0}), column({-100})),
                          {Transition(docking_guard, id, 1)}, LinearSys(A1, B), c);
        Location docking("S2", Polytope(matrix(8, 5, region), column(region_b)), {},
                         LinearSys(A2, B), c);
        return HybridAutomaton("rendezvous_" + name, {approach, docking});
    }
    // approach: t <= t_u and x <= -100, in the order of the model file
    const Polytope inv1 =
        v.x_first ? Polytope(matrix(2, 5, {1, 0, 0, 0, 0, 0, 0, 0, 0, 1}), column({-100, v.t_u}))
                  : Polytope(matrix(2, 5, {0, 0, 0, 0, 1, 1, 0, 0, 0, 0}), column({v.t_u, -100}));
    std::vector<Transition> from_approach{Transition(docking_guard, id, 1)};
    for (Transition &t : aborts(v, v.approach_aborts, id)) from_approach.push_back(t);
    std::vector<double> rows2 = {0, 0, 0, 0, 1};
    rows2.insert(rows2.end(), region.begin(), region.end());
    std::vector<double> b2 = {v.t_u};
    b2.insert(b2.end(), region_b.begin(), region_b.end());
    Location approach("S1", inv1, from_approach, LinearSys(A1, B), c);
    Location docking("S2", Polytope(matrix(9, 5, rows2), column(b2)),
                     aborts(v, v.docking_aborts, id), LinearSys(A2, B), c);
    Location passive("S3", Polytope(matrix(1, 5, {0, 0, 0, 0, -1}), column({v.passive_b})), {},
                     LinearSys(A3, B), c);
    return HybridAutomaton("rendezvous_" + name, {approach, docking, passive});
}

/// The problem and the options of the scripts for the instance `name`.
inline std::pair<HybridReachProblem, alg::HybridStandard> options(const std::string &name) {
    const Variant v = variant(name);
    HybridReachProblem problem;
    problem.initial_set = Zonotope(column({-900, -400, 0, 0, 0}),
                                   matrix(5, 5, {25, 0, 0, 0, 0, 0, 25, 0, 0, 0, 0, 0, 0,
                                                 0,  0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0}));
    problem.t_final = 300;
    alg::HybridStandard a;
    a.time_step = v.timed ? std::vector<double>{2e-1, 2e-2, 2e-1} : std::vector<double>{2e-1, 2e-2};
    a.zonotope_order = 10;
    a.taylor_terms = 3;
    // the guard intersection of each script
    using alg::Enclose;
    if (name == "SRA01" || name == "SRNA01" || name == "SRU01")
        a.guard_intersect = alg::ZonoGirard{{Enclose::box}};
    else if (name == "SRA04")
        a.guard_intersect = alg::ZonoGirard{{Enclose::pca}};
    else if (name == "SRA07" || name == "SRA08" || name == "SRU02")
        a.guard_intersect = alg::NondetGuard{{Enclose::pca, Enclose::box}};
    else
        a.guard_intersect = alg::NondetGuard{{Enclose::pca}};
    return {problem, a};
}

} // namespace cora::bench::rendezvous

// ---------------------------------------  END OF CODE  ---------------------------------------- //
