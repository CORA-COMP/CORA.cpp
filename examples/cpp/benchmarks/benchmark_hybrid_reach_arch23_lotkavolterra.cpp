// benchmark_hybrid_reach_arch23_lotkavolterra - the hybrid Lotka-Volterra benchmark (LOVO25)
//
// MATLAB CORA's benchmark_hybrid_reach_ARCH23_lotkaVolterra.m with the model of
// models/Cora/lotkaVolterra.m, numbers copied: the predator-prey system x1' = 3 x1 - 3 x1 x2,
// x2' = x1 x2 - x2 with a clock x3, a ring of radius 0.161 around (1, 1) that is the invariant of
// 'inside' and the level set guard to it from 'outside', and the initial box x1 in [1.288, 1.312],
// x2 = 1, x3 = 0 up to t = 3.64. The orbit crosses the ring tangentially in the upper arc and fully
// in the lower arc; the script follows each branch after it left the ring with the continuous
// dynamics (aux_reach_rem), as the hybrid analysis stops there, and intersects the final sets with
// the plane x3 = 3.64. The specification of the ARCH-COMP report (Sec. LOVO): the box of x1 x x2
// of all final sets has an area below 1e-2, one final set crossed the ring twice, one four times,
// and none an odd number of times. The crossings of a final set are those of the inside branches
// it passed, an entry and an exit each; the script prints the area and records the verdict true.
//
// Syntax:   build/core/benchmarks/benchmark_hybrid_reach_arch23_lotkavolterra [LOVO25]
// Outputs:  one line 'benchmark,instance,result,time', the area and the crossings on stderr
// See also: benchmark_hybrid_reach_arch23_vanderpol.cpp, benchmark_hybrid.h
//
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "benchmark_hybrid.h"
#include "cora/cont_set/taylm/taylm.h"

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using namespace cora;
using cora::bench::column;

constexpr double kRadius = 0.161;
constexpr double kFinal = 3.64;

/// (x1 - 1)^2 + (x2 - 1)^2 - r^2, the ring of the script's `fun` and of the model.
Expr aux_ring() {
    const Expr x = Expr::var(0), y = Expr::var(1);
    return pow(x - 1, 2) + pow(y - 1, 2) - kRadius * kRadius;
}

NonlinearSys aux_flow() {
    return NonlinearSys(
        [](const std::vector<Expr> &x, const std::vector<Expr> &) {
            return std::vector<Expr>{3 * x[0] - 3 * x[0] * x[1], x[0] * x[1] - x[1], Expr(1.0)};
        },
        3, 1);
}

/// lotkaVolterra.m: 'outside' (ring >= 0) jumps by the level set guard ring == 0 into 'inside'.
HybridAutomaton aux_automaton() {
    const NonlinearSys sys = aux_flow();
    const Expr ring = aux_ring();
    const LinearReset id(Tensor::eye(3), std::nullopt, column({0, 0, 0}));
    Location outside("outside", LevelSet(-ring, 3, CompOp::le),
                     {Transition(LevelSet(-ring, 3, CompOp::eq), id, 1)}, sys);
    Location inside("inside", LevelSet(ring, 3, CompOp::le), {}, sys);
    return HybridAutomaton("lotkaVolterra", {outside, inside});
}

/// The options of the script: lin, 165 steps of tFinal, taylorTerms 3, zonotopeOrder 20.
alg::HybridStandard aux_hybrid_alg() {
    alg::HybridStandard a;
    a.time_step = {kFinal / 165};
    a.taylor_terms = 3;
    a.zonotope_order = 20;
    a.tensor_order = 3;
    a.error_order = 5;
    a.intermediate_order = 20;
    return a;
}

PolyZonotope aux_poly(const Set &S) {
    return S.is<PolyZonotope>() ? S.as<PolyZonotope>() : PolyZonotope(S.as<Zonotope>());
}

// The continuation of a branch --------------------------------------------------------------------

/// aux_isIntersectingPCA: whether the ring may meet the set, by Taylor models over its PCA box.
bool aux_meets_ring(const Expr &ring, const PolyZonotope &pZ) {
    const std::vector<Taylm> tay = Taylm::from_zonotope(pZ.zonotope().reduce_pca());
    const std::array<double, 2> range = Taylm::evaluate(ring, tay).bounds();
    return range[0] <= 0 && 0 <= range[1];
}

/// What aux_reachRem returns: the sets after the ring and the end of its time.
struct Remainder {
    PolyReachSet R;
    double t_stop = 0;
    bool ok = true;
};

/// aux_reachRem: the sets of a branch in the ring (split along their longest generator where
/// they meet it), their box beyond x1 = 1 (flow 'sup': before), contracted to the ring, the
/// intersection with the ring as the start set of the continuous flow up to the final time.
Remainder aux_reach_rem(const std::vector<Set> &branch, double t_final, double t_shift, bool before,
                        double time_step) {
    const Expr ring = aux_ring();
    const LevelSet guard(-ring, 3, CompOp::eq);
    const Tensor w = column({1, 1, 1.0 / 1000}), w_inv = column({1, 1, 1000});
    std::optional<Interval> box;
    // the union of the boxes of the pieces that meet the ring
    for (const Set &S : branch) {
        const PolyZonotope pZ = aux_poly(S);
        if (!aux_meets_ring(ring, pZ)) continue;
        const auto [a, b] = pZ.linear_map(w.diag()).split_longest_gen();
        for (const PolyZonotope &half : {a, b}) {
            if (!aux_meets_ring(ring, half)) continue;
            const Interval I = half.linear_map(w_inv.diag()).interval();
            box = box ? box->unite(I) : I;
        }
    }
    if (!box) return {{}, 0, false};
    // x1 beyond 1 (flow 'inf': x1 >= 1) or before (flow 'sup': x1 <= 1)
    std::vector<double> lo = box->inf().data(), hi = box->sup().data();
    (before ? hi : lo)[0] = 1.0;
    const Interval cut(column(lo), column(hi));
    // the box contracted to the ring, then the intersection with it
    const std::optional<Interval> tight = guard.tighten_domain(cut);
    if (!tight) return {{}, 0, false};
    const std::optional<PolyZonotope> jump = guard.intersect(PolyZonotope(*tight));
    if (!jump) return {{}, 0, false};
    // the continuous flow from there, with the step adapted to the shorter horizon
    const double t_start = tight->inf().data()[2];
    const double horizon = t_final - t_shift - t_start;
    alg::Linearization lin;
    lin.time_step = horizon / std::ceil(horizon / time_step);
    lin.taylor_terms = 3;
    lin.zonotope_order = 20;
    lin.tensor_order = 3;
    lin.error_order = 5;
    lin.intermediate_order = 20;
    Remainder out;
    out.R =
        aux_flow().reach({.initial_set = *jump, .t_final = horizon}, alg::PolyNonlinearReach{lin});
    out.t_stop = horizon + t_start;
    return out;
}

/// The box of the final set at the clock x3 = t_final: hp & conZonotope(zonotope(set)).
Interval aux_final_box(const PolyZonotope &last) {
    return ConZonotope(last.zonotope())
        .intersect(cora::bench::hyperplane({0, 0, 1}, kFinal))
        .interval();
}


/// The final set of a chain of branches and the times the ring was crossed on the way to it.
struct Final {
    Interval box;
    int crossings;
};

/// Entry and exit of every branch of 'inside' that the chain passed (a branch is a stay inside).
int aux_crossings(const std::vector<const LocationReachSet *> &chain) {
    int crossings = 0;
    for (const LocationReachSet *b : chain) {
        if (b->location != 1)
            CORA_FAIL(Error::wrong_value, "lotkaVolterra: a chain passes 'inside' only");
        crossings += 2;
    }
    return crossings;
}

/// The four properties of the report for the final sets: the box of x1 x x2 of all of them has an
/// area below 1e-2, one crossed the ring twice, one four times, none an odd number of times.
bool aux_verdict(const std::vector<Final> &finals, double &area) {
    Interval hull = finals.front().box;
    for (const Final &f : finals) hull = hull.unite(f.box);
    const std::vector<double> lo = hull.inf().data(), hi = hull.sup().data();
    area = (hi[0] - lo[0]) * (hi[1] - lo[1]);
    const auto has = [&](int k) {
        return std::any_of(finals.begin(), finals.end(),
                           [&](const Final &f) { return f.crossings == k; });
    };
    const bool odd = std::any_of(finals.begin(), finals.end(),
                                 [](const Final &f) { return f.crossings % 2 != 0; });
    return area < 1e-2 && has(2) && has(4) && !odd;
}

/// R1 from the box, the continuations R2 (after the tangential crossing R1(2)), R3 (its re-entry
/// into the ring) and R4 (after R3(2)), R5 (after the full crossing R1(3)): the final boxes.
bool aux_run(const std::string &name) {
    if (name != "LOVO25") CORA_FAIL(Error::wrong_value, "lotkaVolterra: unknown instance " + name);
    const HybridAutomaton HA = aux_automaton();
    const alg::HybridStandard a = aux_hybrid_alg();
    const auto start = std::chrono::steady_clock::now();
    // R1: the initial box through 'outside' and the crossings of the ring
    const double e = 0.012;
    HybridReachProblem problem;
    problem.initial_set = Interval(column({1.3 - e, 1, 0}), column({1.3 + e, 1, 0})).zonotope();
    problem.t_final = kFinal;
    const HybridReachSet R1 = HA.reach(problem, a);
    if (R1.branches.size() < 3)
        CORA_FAIL(Error::wrong_value, "lotkaVolterra: R1 has " +
                                          std::to_string(R1.branches.size()) + " branches, not 3");
    // with CORACPP_BENCHMARK_VERBOSE, the branches of R1: location, parent and sets
    if (std::getenv("CORACPP_BENCHMARK_VERBOSE"))
        for (const LocationReachSet &b : R1.branches)
            std::cerr << "    R1: location " << b.location << " (parent " << b.parent << "), "
                      << b.time_interval.size() << " sets\n";
    // R2: after the tangential crossing R1(2), until the time kFinal - 2
    const double dt = a.time_step[0];
    const Remainder R2 = aux_reach_rem(R1.branches[1].time_interval, kFinal, 2, true, dt);
    std::vector<Final> finals;
    double area = 0;
    bool verified = R2.ok;
    if (verified) {
        // R3: back into the hybrid automaton from the end of R2, the full crossing R3(2)
        problem.initial_poly_set = R2.R.time_point.back();
        problem.initial_set.reset();
        problem.t_final = kFinal - R2.t_stop;
        const HybridReachSet R3 = HA.reach(problem, a);
        verified = R3.branches.size() >= 2;
        if (verified) {
            // R4 after R3(2) (the chain R1(2), R3(2)), R5 after R1(3)
            const Remainder R4 = aux_reach_rem(R3.branches[1].time_interval, kFinal, 0, false, dt);
            const Remainder R5 = aux_reach_rem(R1.branches[2].time_interval, kFinal, 0, false, dt);
            verified = R4.ok && R5.ok;
            if (verified) {
                finals.push_back({aux_final_box(R4.R.time_point.back()),
                                  aux_crossings({&R1.branches[1], &R3.branches[1]})});
                finals.push_back(
                    {aux_final_box(R5.R.time_point.back()), aux_crossings({&R1.branches[2]})});
            }
        }
    }
    verified = verified && aux_verdict(finals, area);
    const std::chrono::duration<double> wall = std::chrono::steady_clock::now() - start;
    cora::bench::print_line("LotkaVolterra", name, verified, wall.count());
    std::cerr << "  area of the final box: " << area;
    for (const Final &f : finals) std::cerr << ", a final set with " << f.crossings << " crossings";
    std::cerr << " (the limit is 1e-2)\n";
    return verified;
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main(int argc, char **argv) { return cora::bench::run_family(argc, argv, {"LOVO25"}, aux_run); }

// ---------------------------------------  END OF CODE  ---------------------------------------- //
