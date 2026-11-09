// benchmark_hybrid_reach_arch23_rendezvous - spacecraft rendezvous benchmark of ARCH 2023
//
// The instances of MATLAB CORA's benchmark_hybrid_reach_ARCH23_rendezvous_<instance>.m with the
// automata of models/Cora/spacecraftRendezvous/rendezvous_<instance>.m, numbers copied. The state
// is (x, y, vx, vy, t); the chaser approaches (location 0), docks (1) or aborts into the passive
// mode (2). The instances differ by the abort times, the guard intersection and the specs.
//
// Syntax:   build/core/benchmarks/benchmark_hybrid_reach_arch23_rendezvous [instance ...]
// Outputs:  one line 'benchmark,instance,result,time' per instance, as the MATLAB scripts
// See also: benchmark_hybrid.h
//
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "benchmark_hybrid.h"

#include <cmath>
#include <optional>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using namespace cora;
using cora::bench::column;
using cora::bench::hyperplane;
using cora::bench::matrix;

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
Variant aux_variant(const std::string &name) {
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
Interval aux_window(const std::array<double, 2> &w) {
    const double dist = 1e5;
    return {column({-dist, -dist, -dist, -dist, w[0]}), column({dist, dist, dist, dist, w[1]})};
}

/// The aborts of a mode into the passive one (location 2), all with the identity reset.
std::vector<Transition> aux_aborts(const Variant &v, const Windows &windows,
                                   const LinearReset &id) {
    std::vector<Transition> out;
    if (v.abort_at) out.emplace_back(hyperplane({0, 0, 0, 0, -1}, -*v.abort_at), id, 2);
    for (const auto &w : windows) out.emplace_back(aux_window(w), id, 2);
    return out;
}

/// A mode's flow: x' = vx, y' = vy, the velocities by the rows a3 and a4, and t' = 0 (from c).
Tensor aux_flow(const std::vector<double> &a3, const std::vector<double> &a4) {
    std::vector<double> e = {0, 0, 1, 0, 0, 0, 0, 0, 1, 0};
    for (const auto *row : {&a3, &a4}) {
        e.insert(e.end(), row->begin(), row->end());
        e.push_back(0);
    }
    e.resize(25, 0.0);
    return matrix(5, 5, e);
}

HybridAutomaton aux_automaton(const std::string &name, const Variant &v) {
    // the dynamics of the three modes, with t' = 1 as the constant input c
    const Tensor A1 = aux_flow({-0.05759976588, 0.0002009598965, -2.89995084, 0.008772008945},
                               {-0.0001740313574, -0.06651239849, -0.008753511055, -2.903002693});
    const Tensor A2 = aux_flow({-0.5759999431, 0.0002624860794, -19.22997959, 0.008762759318},
                               {-0.0002624860807, -0.5759999402, -0.008762760682, -19.2299766});
    const Tensor A3 = aux_flow({5.758947211E-05, 0, 0, 0.00876276}, {0, 0, -0.00876276, 0});
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
    for (Transition &t : aux_aborts(v, v.approach_aborts, id)) from_approach.push_back(t);
    std::vector<double> rows2 = {0, 0, 0, 0, 1};
    rows2.insert(rows2.end(), region.begin(), region.end());
    std::vector<double> b2 = {v.t_u};
    b2.insert(b2.end(), region_b.begin(), region_b.end());
    Location approach("S1", inv1, from_approach, LinearSys(A1, B), c);
    Location docking("S2", Polytope(matrix(9, 5, rows2), column(b2)),
                     aux_aborts(v, v.docking_aborts, id), LinearSys(A2, B), c);
    Location passive("S3", Polytope(matrix(1, 5, {0, 0, 0, 0, -1}), column({v.passive_b})), {},
                     LinearSys(A3, B), c);
    return HybridAutomaton("rendezvous_" + name, {approach, docking, passive});
}

/// ~any(supremum(interval(C x) - d) > 0): CORA's custom check by the box of C x.
Specification aux_box_spec(const Tensor &C, const Tensor &d) {
    return Specification::custom([C, d](const ContSet &S) {
        const Interval I = S.linear_map(C)->interval();
        const std::vector<double> sup = I.sup().data(), b = d.data();
        for (std::size_t i = 0; i < b.size(); ++i)
            if (sup[i] - b[i] > 0) return false;
        return true;
    });
}

/// A safe set {x | C x <= d} as halfspaces.
Specification aux_safe(const Tensor &C, const std::vector<double> &d) {
    std::vector<Halfspace> h;
    h.reserve(d.size());
    for (std::size_t i = 0; i < d.size(); ++i)
        h.push_back({C.slice(-2, int64_t(i), int64_t(i) + 1).matrix_transpose(), d[i]});
    return Specification::safe_set(h);
}

/// CORA's aux_checkSpaceStationHit: the projection onto (x, y) must miss the box [-0.1, 0.1]^2,
/// first by its box, then reduced to order 3.
Specification aux_station() {
    return Specification::custom([](const ContSet &S) {
        const Interval station(column({-0.1, -0.1}), column({0.1, 0.1}));
        // the box of the projection decides most sets, the order-3 zonotope the rest
        const Set proj = S.project({0, 1});
        if (!is_intersecting(station, proj->interval())) return true;
        return !is_intersecting(station, *proj->reduce(3));
    });
}

/// The specifications of the instance, each with the locations it applies to.
std::vector<HybridSpecification> aux_specs(const std::string &name) {
    // the velocity in the docking mode, and the line of sight
    const Tensor Cv =
        matrix(8, 5, {0, 0, 1, 0, 0, 0, 0, -1, 0,  0, 0, 0, 0,  1, 0, 0, 0, 0,  -1, 0,
                      0, 0, 1, 1, 0, 0, 0, 1,  -1, 0, 0, 0, -1, 1, 0, 0, 0, -1, -1, 0});
    const std::vector<double> dv = {3, 3, 3, 3, 4.2, 4.2, 4.2, 4.2};
    const double pi = 4 * std::atan(1.0);
    const double s = std::tan(pi / 6);
    const Tensor Cl = matrix(2, 5, {s, -1, 0, 0, 0, s, 1, 0, 0, 0});
    const std::vector<double> dl = {0, 0};
    // the scripts of SRA07/08 and SRU01/02 check the box of C x, the others the halfspaces
    const bool custom = name == "SRA07" || name == "SRA08" || name == "SRU01" || name == "SRU02";
    std::vector<HybridSpecification> specs;
    if (custom) {
        specs.push_back({aux_box_spec(Cv, column(dv)), {1}});
        specs.push_back({aux_box_spec(Cl, column(dl)), {1}});
    } else {
        specs.push_back({aux_safe(Cv, dv), {1}});
        specs.push_back({aux_safe(Cl, dl), {1}});
    }
    if (name != "SRNA01") specs.push_back({aux_station(), {2}});
    return specs;
}

cora::bench::HybridInstance aux_instance(const std::string &name) {
    // the box of the initial positions, and the options of the scripts
    const Variant v = aux_variant(name);
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
    // the unsafe instances SRU01 and SRU02 are not verified
    const bool expected = name.rfind("SRU", 0) != 0;
    return {"Rendezvous", name,    aux_automaton(name, v), problem, a, aux_specs(name),
            nullptr,      expected};
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main(int argc, char **argv) {
    const std::vector<std::string> family = {"SRA01", "SRA02", "SRA03",  "SRA04", "SRA05", "SRA06",
                                             "SRA07", "SRA08", "SRNA01", "SRU01", "SRU02"};
    return cora::bench::run_family(argc, argv, family, aux_instance);
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
