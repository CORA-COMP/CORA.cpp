// test_location_level_set - a location with level sets: the invariant, potInt and checkFlow
//
// A particle moves right at speed 1 (x' = (1 + u, 0)); the invariant is the unit disc, a guard is
// the unit circle. The reach stops when the sets have left the disc, the circle is hit by the
// sets that cross it, a set moving inward is dropped by the flow check, and the enclosure of a
// level set guard is refused as not ported (it is the next step of the port).
//
// CORA: hybridDynamics/@location/potInt.m
// CORA: hybridDynamics/@location/checkFlow.m
// CORA: hybridDynamics/@location/reach.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/hybrid_dynamics/hybrid_automaton/hybrid_automaton.h"
#include "hybrid_dynamics/location/private/priv.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// x^2 + y^2 - 1.
Expr circle() {
    const Expr x = Expr::var(0), y = Expr::var(1);
    return x * x + y * y - 1.0;
}

/// The particle: the unit disc as the invariant and the circle as a guard (or none), with an
/// input u in the speed.
Location particle(bool with_guard) {
    const NonlinearSys sys(
        [](const std::vector<Expr> &, const std::vector<Expr> &u) {
            return std::vector<Expr>{1.0 + u[0], Expr(0.0)};
        },
        2, 1);
    std::vector<Transition> trans;
    if (with_guard)
        trans.emplace_back(LevelSet(circle(), 2, CompOp::eq), LinearReset(Tensor::eye(2)), 0);
    return Location("fly", LevelSet(circle(), 2, CompOp::le), trans, sys);
}

Zonotope box_at(double x, double y) {
    return Zonotope(column({x, y}), Tensor({{0.05, 0.0}, {0.0, 0.05}}));
}

/// The invariant stops the reach when the sets are out of the disc.
void the_invariant_ends_the_reach(const std::string &b) {
    HybridAutomaton HA("disc", {particle(false)});
    HybridReachProblem p;
    p.initial_set = box_at(-0.5, 0.0);
    p.t_final = 3.0;
    alg::HybridStandard a;
    a.time_step = {0.1};
    const HybridReachSet R = HA.reach(p, a);
    // the box leaves the disc at x = 1 + 0.05 after 1.55 s, so about 16 steps of 30
    const std::size_t steps = R.branches.at(0).time_interval.size();
    check(R.verified && steps >= 15 && steps <= 19,
          b + ": the reach ends at the disc: " + std::to_string(steps));
}

/// The region helpers and the pot_int of a level set guard.
void the_circle_is_hit_by_the_crossing_sets(const std::string &b) {
    const Location loc = particle(true);
    LocationReachSet R;
    // the sets at x = -0.5, 0.97 (crossing: it spans 0.92..1.02) and 1.5
    for (const double x : {-0.5, 0.97, 1.5}) R.time_interval.emplace_back(box_at(x, 0.0));
    const GuardHits hits = priv_pot_int(loc, R, {});
    check(hits.transitions.size() == 1 && hits.sets.at(0) == 1, b + ": only the crossing set");
    check(priv_region_contains(loc.invariant(), column({0.0, 0.0})) &&
              !priv_region_contains(loc.invariant(), column({2.0, 0.0})),
          b + ": the invariant contains the center and not a far point");
    check(!priv_region_empty(loc.invariant()), b + ": the disc is not empty");
    check(priv_region_empty(LevelSet::empty(2)), b + ": the empty level set is");
}

/// The flow check keeps a set that moves out through the circle and drops one that moves in.
void the_flow_decides(const std::string &b) {
    const Location loc = particle(true);
    const LevelSet guard(circle(), 2, CompOp::eq);
    const Zonotope U(column({0.0}), Tensor({{0.1}}));
    // the first set, inside the disc, orients the guard; (1, 0) flows out, (-1, 0) flows in
    const std::vector<Set> sets{box_at(0.5, 0.0), box_at(1.0, 0.0), box_at(-1.0, 0.0)};
    const std::vector<Set> kept = priv_check_flow(loc, guard, sets, U);
    check(kept.size() == 2, b + ": the set moving in is dropped");
    // the property: every kept set has a point where the flow is outward
    for (const Set &S : kept) check(S->center().data()[0] > 0, b + ": a kept set is on the right");
    // at the top the center flow is 0 but the box has points that move out: the set stays
    const std::vector<Set> top{box_at(0.5, 0.0), box_at(0.0, 1.0)};
    check(priv_check_flow(loc, guard, top, U).size() == 2, b + ": a set that may move out stays");
}

/// A set that reaches a level set guard cannot be enclosed yet.
void the_enclosure_is_not_ported(const std::string &b) {
    HybridAutomaton HA("disc", {particle(true)});
    HybridReachProblem p;
    p.initial_set = box_at(-0.5, 0.0);
    p.t_final = 3.0;
    alg::HybridStandard a;
    a.time_step = {0.1};
    check(test::raises([&] { HA.reach(p, a); }, Error::not_supported),
          b + ": the guard enclosure is refused");
}

/// The constructors take level sets and check their dimension.
void construction() {
    const LinearReset id(Tensor::eye(2));
    check(test::raises([&] { Transition(LevelSet(Expr::var(0), 1, CompOp::eq), id, 0); },
                       Error::wrong_input_in_constructor),
          "transition: a guard of another dimension");
    const NonlinearSys sys([](const std::vector<Expr> &x) { return std::vector<Expr>{x[0]}; }, 1);
    check(test::raises([&] { Location("l", LevelSet(circle(), 2, CompOp::le), {}, sys); },
                       Error::wrong_input_in_constructor),
          "location: an invariant of another dimension");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        the_invariant_ends_the_reach(b);
        the_circle_is_hit_by_the_crossing_sets(b);
        the_flow_decides(b);
        the_enclosure_is_not_ported(b);
    });
    construction();
    ChecksScope off(false);
    check(particle(true).transitions().size() == 1, "location: valid with the checks off");
    return test::finish("location level set");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
