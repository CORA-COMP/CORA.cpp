// benchmark_hybrid_reach_arch23_lotkavolterra - the hybrid Lotka-Volterra benchmark (LOVO25)
//
// The predator-prey system of lotkaVolterra.m with a clock x3 and a ring of radius 0.161 around
// (1, 1) (invariant of 'inside', level set guard from 'outside'), x1 in [1.288, 1.312], x2 = 1 up
// to t = 3.64. The ARCH-COMP report asks: the box of all final sets has an area below 1e-2, one
// final set crossed the ring twice, one four times, none an odd number of times. LOVO25-script
// follows CORA's script, which steers by hand and records the verdict 1 untested: it reproduces
// CORA's area and decides nothing. LOVO25 decides: the automaton, with a jump back out of
// 'inside', reaches pieces of the initial box (a jump is a crossing), the area is that of the sets
// at x3 = 3.64, 'no odd count' that none is in 'inside'; where the tangential arc makes the
// linearization error explode, the flow alone encloses the piece. Exactly 2 and 4 crossings are
// shown by simulated trajectories. The verdict is 1, 0 (a violating trajectory) or unknown.
//
// Syntax:   benchmark_hybrid_reach_arch23_lotkavolterra [LOVO25] [LOVO25-script]
// Outputs:  one line 'benchmark,instance,result,time'; the areas and witnesses on stderr
// See also: benchmark_hybrid_reach_arch23_vanderpol.cpp, benchmark_hybrid.h
//
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   11-November-2026
// Reviewed:      ---

#include "benchmark_hybrid.h"
#include "cora/cont_set/taylm/taylm.h"

#include <algorithm>
#include <cmath>
#include <map>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using namespace cora;
using cora::bench::column;

constexpr double kRadius = 0.161;
constexpr double kFinal = 3.64;
constexpr double kHalfWidth = 0.012; // of the initial box around x1 = 1.3
constexpr int kPieces = 8;           // equal pieces of the initial box
constexpr int kWitnesses = 241;      // start points of the simulation, 1e-4 apart

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

/// lotkaVolterra.m: 'outside' (ring >= 0) jumps by the level set guard ring == 0 into 'inside';
/// with `back`, 'inside' (ring <= 0) jumps by ring == 0 into 'outside', so that every crossing of
/// the ring is a jump and no continuation by hand is needed.
HybridAutomaton aux_automaton(bool back = false) {
    const NonlinearSys sys = aux_flow();
    const Expr ring = aux_ring();
    const LinearReset id(Tensor::eye(3), std::nullopt, column({0, 0, 0}));
    Location outside("outside", LevelSet(-ring, 3, CompOp::le),
                     {Transition(LevelSet(-ring, 3, CompOp::eq), id, 1)}, sys);
    std::vector<Transition> out_of_ring;
    if (back) out_of_ring.emplace_back(LevelSet(ring, 3, CompOp::eq), id, 0);
    Location inside("inside", LevelSet(ring, 3, CompOp::le), out_of_ring, sys);
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
        aux_flow().reach({.initial_set = jump, .t_final = horizon}, alg::PolyNonlinearReach{lin});
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
bool aux_run_script(const std::string &name) {
    const HybridAutomaton HA = aux_automaton();
    const alg::HybridStandard a = aux_hybrid_alg();
    const auto start = std::chrono::steady_clock::now();
    // R1: the initial box through 'outside' and the crossings of the ring
    HybridReachProblem problem;
    problem.initial_set =
        Interval(column({1.3 - kHalfWidth, 1, 0}), column({1.3 + kHalfWidth, 1, 0})).zonotope();
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


// The reachable set of the automaton --------------------------------------------------------------

/// A final set: a branch of the automaton that is alive at the clock x3 = T, the number of jumps
/// it took (each one a crossing of the ring) and the box of its sets at x3 = T. A piece that
/// the flow alone enclosed has no branch: jumps is -1.
struct Alive {
    int64_t location;
    int jumps;
    Interval box;
};

/// The box of x1 x x2 x x3 of the set at the clock x3 = T, certified (the dual certificates of
/// ConZonotope::Cut, ten Fourier-Motzkin passes); none if the set does not reach T.
std::optional<Interval> aux_slice(const Set &S) {
    const Interval box = S->interval();
    const double lo = box.inf().data()[2], hi = box.sup().data()[2];
    if (hi < kFinal || kFinal < lo) return std::nullopt;
    // the polynomial zonotope as the zonotope that holds it, cut by the plane x3 = T
    const ConZonotope::Cut cut =
        ConZonotope(aux_poly(S).zonotope()).cut(cora::bench::hyperplane({0, 0, 1}, kFinal));
    const Interval slice = cut.interval_cert_tight(10);
    if (slice.is_empty()) return std::nullopt;
    return slice;
}

/// Whether the box of x1 x x2 is certainly outside the ring: its distance to (1, 1) exceeds r.
bool aux_outside_ring(const Interval &box) {
    const std::vector<double> lo = box.inf().data(), hi = box.sup().data();
    const double dx = std::max({0.0, lo[0] - 1, 1 - hi[0]});
    const double dy = std::max({0.0, lo[1] - 1, 1 - hi[1]});
    return dx * dx + dy * dy > kRadius * kRadius;
}

/// The jumps of every branch: those of its parent and one (a parent is always an earlier branch).
std::vector<int> aux_jump_counts(const HybridReachSet &R) {
    std::vector<int> jumps(R.branches.size(), 0);
    for (std::size_t i = 0; i < R.branches.size(); ++i)
        if (R.branches[i].parent >= 0) jumps[i] = jumps[std::size_t(R.branches[i].parent)] + 1;
    return jumps;
}

/// Every branch that has a set at x3 = T, with the hull of the boxes of its sets there.
std::vector<Alive> aux_alive(const HybridReachSet &R) {
    const std::vector<int> jumps = aux_jump_counts(R);
    std::vector<Alive> out;
    for (std::size_t i = 0; i < R.branches.size(); ++i) {
        std::optional<Interval> hull;
        for (const Set &S : R.branches[i].time_interval)
            if (const std::optional<Interval> slice = aux_slice(S))
                hull = hull ? hull->unite(*slice) : *slice;
        if (hull) out.push_back({R.branches[i].location, jumps[i], *hull});
    }
    return out;
}

/// The initial box x1 in [lo, hi], x2 = 1, x3 = 0.
Zonotope aux_start(double lo, double hi) {
    return Interval(column({lo, 1, 0}), column({hi, 1, 0})).zonotope();
}

/// The box at x3 = T of the flow alone (no jumps) from the piece, with the options of the
/// automaton; none if it does not reach T. The ring is then a property of the position.
std::optional<Interval> aux_flow_box(double lo, double hi, const alg::HybridStandard &a) {
    alg::Linearization lin;
    lin.time_step = a.time_step[0];
    lin.taylor_terms = a.taylor_terms;
    lin.zonotope_order = a.zonotope_order;
    lin.tensor_order = a.tensor_order;
    lin.error_order = a.error_order;
    lin.intermediate_order = a.intermediate_order;
    const ReachSet flow = aux_flow().reach(
        {.initial_set = aux_start(lo, hi), .t_final = kFinal + lin.time_step}, lin);
    std::optional<Interval> hull;
    for (const Zonotope &Z : flow.time_interval)
        if (const std::optional<Interval> slice = aux_slice(Z))
            hull = hull ? hull->unite(*slice) : *slice;
    return hull;
}

// The pieces of the initial box -------------------------------------------------------------------

/// What the pieces of the initial box gave: the sets alive at T, and how each piece was reached.
struct Cover {
    std::vector<Alive> alive;
    std::size_t by_automaton = 0, by_flow = 0, branches = 0;
    std::string error;
};

/// The automaton from the piece up to T and a step more (so that every branch has a set at T).
/// Where CORA's linearization error explodes (the sets in contact with the ring keep jumping in
/// and out along the tangential arc, their times growing), the flow alone encloses the piece.
void aux_cover(const HybridAutomaton &HA, const alg::HybridStandard &a, double lo, double hi,
               Cover &out) {
    HybridReachProblem problem;
    problem.initial_set = aux_start(lo, hi);
    problem.t_final = kFinal + a.time_step[0];
    try {
        const HybridReachSet R = HA.reach(problem, a);
        for (const Alive &f : aux_alive(R)) out.alive.push_back(f);
        out.branches += R.branches.size();
        ++out.by_automaton;
    } catch (const Error &e) {
        out.error = e.what();
        ++out.by_flow;
        if (const std::optional<Interval> box = aux_flow_box(lo, hi, a))
            out.alive.push_back({-1, -1, *box});
    }
}

// The witnesses -----------------------------------------------------------------------------------

/// What a simulated trajectory did: the crossings of the ring up to T, how shallow its dip into
/// the ring was at the least (a tangential approach), and its state at T.
struct Trajectory {
    double x1;
    int crossings = 0;
    double shallowest = -1e300;
    std::vector<double> end;
};

/// The trajectories from `count` points (x1, 1, 0) of the initial box, the ends and an even grid,
/// in steps of 1e-4; the crossings are the sign changes of the ring between the steps.
std::vector<Trajectory> aux_witnesses(int count) {
    std::vector<double> starts;
    std::vector<Trajectory> out;
    out.resize(std::size_t(count));
    for (int i = 0; i < count; ++i) {
        out[std::size_t(i)].x1 = 1.3 - 0.012 + 0.024 * i / (count - 1);
        starts.insert(starts.end(), {out[std::size_t(i)].x1, 1.0, 0.0});
    }
    // the start points as columns (3, count); the system has one input, which is 0
    const Tensor x0 = Tensor::from_data(starts, {count, 3}).matrix_transpose();
    const std::vector<Tensor> path = aux_flow().simulate(
        {.initial_state = x0, .input = Tensor::zeros({1, 1}), .t_final = kFinal},
        alg::RungeKutta4{.time_step = 1e-4, .max_step = 1e-4});
    const auto ring = [](double a, double b) {
        return (a - 1) * (a - 1) + (b - 1) * (b - 1) - kRadius * kRadius;
    };
    std::vector<double> prev(std::size_t(count), 0.0), dip(std::size_t(count), 0.0);
    for (std::size_t k = 0; k < path.size(); ++k) {
        const std::vector<double> p = path[k].data(); // row-major (3, count)
        for (std::size_t i = 0; i < std::size_t(count); ++i) {
            const double g = ring(p[i], p[std::size_t(count) + i]);
            Trajectory &t = out[i];
            const bool crossed = k > 0 && (g > 0) != (prev[i] > 0);
            // the deepest point of the stretch inside the ring ends when it is left
            if (crossed && g > 0) t.shallowest = std::max(t.shallowest, dip[i]);
            if (crossed) ++t.crossings;
            dip[i] = crossed && g <= 0 ? g : std::min(dip[i], g);
            prev[i] = g;
            if (k + 1 == path.size()) t.end = {p[i], p[std::size_t(count) + i]};
        }
    }
    return out;
}

// The verdict -------------------------------------------------------------------------------------

/// LOVO25: the automaton from the pieces of the initial box, the sound checks on the sets alive at
/// T, the witnesses by simulation, the verdict from both.
bool aux_run_verdict(const std::string &name) {
    const HybridAutomaton HA = aux_automaton(true);
    const alg::HybridStandard a = aux_hybrid_alg();
    const auto start = std::chrono::steady_clock::now();
    // the initial box in equal pieces, those off the tangential arc reached with jumps
    Cover cover;
    for (int i = 0; i < kPieces; ++i)
        aux_cover(HA, a, 1.3 - kHalfWidth + 2 * kHalfWidth * i / kPieces,
                  1.3 - kHalfWidth + 2 * kHalfWidth * (i + 1) / kPieces, cover);
    // the sound checks: the hull's area, and no state at T in the ring (odd jumps: in 'inside')
    Interval hull = cover.alive.empty() ? aux_start(0, 0).interval() : cover.alive.front().box;
    bool odd = false;
    for (const Alive &f : cover.alive) {
        hull = hull.unite(f.box);
        odd = odd || (f.jumps < 0 ? !aux_outside_ring(f.box) : f.jumps % 2 != 0);
    }
    const std::vector<double> lo = hull.inf().data(), hi = hull.sup().data();
    const double area = (hi[0] - lo[0]) * (hi[1] - lo[1]);
    // the witnesses: trajectories with exactly 2 and exactly 4 crossings, none with an odd number
    const std::vector<Trajectory> paths = aux_witnesses(kWitnesses);
    const auto witness = [&](int k) {
        return std::find_if(paths.begin(), paths.end(),
                            [&](const Trajectory &t) { return t.crossings == k; });
    };
    const auto violating = std::find_if(paths.begin(), paths.end(),
                                        [](const Trajectory &t) { return t.crossings % 2 != 0; });
    // every simulated end state must lie in the hull (a check of the enclosure)
    const bool enclosed = std::all_of(paths.begin(), paths.end(), [&](const Trajectory &t) {
        return lo[0] <= t.end[0] && t.end[0] <= hi[0] && lo[1] <= t.end[1] && t.end[1] <= hi[1];
    });
    const std::chrono::duration<double> wall = std::chrono::steady_clock::now() - start;
    // the verdict: 1 if proved and witnessed, 0 if a trajectory violates, else unknown
    std::string verdict = "unknown", reason;
    if (violating != paths.end()) {
        verdict = "0";
        reason = "the trajectory from x1 = " + std::to_string(violating->x1) +
                 " crosses the ring " + std::to_string(violating->crossings) + " times";
    } else if (cover.alive.empty() || !(area < 1e-2) || odd || !enclosed) {
        reason = cover.alive.empty() ? "no set reaches T"
                 : odd               ? "a set at T may lie in the ring"
                 : !enclosed         ? "a simulated end state is outside the enclosure"
                                     : "the area of the final sets is not below 1e-2";
    } else if (witness(2) == paths.end() || witness(4) == paths.end()) {
        reason = "no trajectory with exactly 2 and exactly 4 crossings was found";
    } else {
        verdict = "1";
    }
    // the line of the scripts, then the areas, the branches and the witnesses
    std::cout << "LotkaVolterra," << name << "," << verdict << "," << wall.count() << '\n'
              << std::flush;
    std::cerr << "  area of the final box " << area << " (limit 1e-2): " << cover.by_automaton
              << " pieces by the automaton (" << cover.branches << " branches), " << cover.by_flow
              << " by the flow alone\n";
    if (cover.by_flow > 0) std::cerr << "  the automaton failed there: " << cover.error << '\n';
    // for comparison: the flow alone from the whole box (a sound bound of the same states)
    if (const std::optional<Interval> whole = aux_flow_box(1.3 - kHalfWidth, 1.3 + kHalfWidth, a)) {
        const std::vector<double> l = whole->inf().data(), h = whole->sup().data();
        std::cerr << "  the flow alone from the whole box: area " << (h[0] - l[0]) * (h[1] - l[1])
                  << ", outside the ring: " << aux_outside_ring(*whole) << '\n';
    }
    std::map<int, int> by_jumps;
    for (const Alive &f : cover.alive) ++by_jumps[f.jumps];
    for (const auto &[jumps, count] : by_jumps)
        std::cerr << "  " << count
                  << (jumps < 0 ? " final pieces of the flow alone"
                                : " final branches with " + std::to_string(jumps) + " jumps")
                  << '\n';
    for (int k : {2, 4})
        if (witness(k) != paths.end())
            std::cerr << "  witness with " << k << " crossings: x1 = " << witness(k)->x1
                      << " (shallowest dip into the ring " << witness(k)->shallowest << ")\n";
    if (!reason.empty()) std::cerr << "  " << verdict << ": " << reason << '\n';
    return verdict == "1";
}

/// The instances: LOVO25 (the verdict) and LOVO25-script (CORA's script, as in the report's run).
bool aux_run(const std::string &name) {
    if (name == "LOVO25") return aux_run_verdict(name);
    if (name == "LOVO25-script") return aux_run_script(name);
    CORA_FAIL(Error::wrong_value, "lotkaVolterra: unknown instance " + name);
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main(int argc, char **argv) {
    return cora::bench::run_family(argc, argv, {"LOVO25", "LOVO25-script"}, aux_run);
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
