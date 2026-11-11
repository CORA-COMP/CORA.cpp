// test_hybrid_automaton_reach - HybridAutomaton: construction, and reach on hand-checkable models
//
// A bouncing ball (x' = v, v' = -g, the guard x = 0 with v <= 0, the reset v := -0.75 v) is
// reached with every guard intersection; its jumps happen when the exact solution says, and
// every exact trajectory from random initial points lies, at every sampled time, in a set whose
// time range holds that time (the stated property). A clock that reaches a hyperplane exactly at
// a step jumps with the time-point set; specifications stop the analysis where they fail. The
// ball with a LinParamSys flow (x' = (1 +- 0.001) v) bounces the same way, and so does the ball
// written as a NonlinearSys.
//
// CORA: hybridDynamics/@hybridAutomaton/hybridAutomaton.m
// CORA: hybridDynamics/@hybridAutomaton/reach.m
// CORA: hybridDynamics/@location/reach.m
// CORA: hybridDynamics/@location/potInt.m
// CORA: hybridDynamics/@location/guardIntersect.m
// CORA: hybridDynamics/@location/guardIntersect_zonoGirard.m
// CORA: hybridDynamics/@location/guardIntersect_nondetGuard.m
// CORA: hybridDynamics/@location/guardIntersect_conZonotope.m
// CORA: hybridDynamics/@location/calcBasis.m
// CORA: hybridDynamics/@location/checkFlow.m
// PROPERTY: hybridDynamics/@hybridAutomaton/reach.m
// PROPERTY: hybridDynamics/@location/reach.m
// PROPERTY: hybridDynamics/@location/potInt.m
// PROPERTY: hybridDynamics/@location/guardIntersect.m
// PROPERTY: hybridDynamics/@location/guardIntersect_zonoGirard.m
// PROPERTY: hybridDynamics/@location/guardIntersect_nondetGuard.m
// PROPERTY: hybridDynamics/@location/guardIntersect_conZonotope.m
// PROPERTY: hybridDynamics/@location/calcBasis.m
// PROPERTY: hybridDynamics/@location/checkFlow.m
// CORA authors: Matthias Althoff, Mark Wetzlinger, Niklas Kochdumper, Stefan Liu
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   09-November-2026
// Reviewed:      ---

#include "cora/cont_set/registry.h"
#include "cora/hybrid_dynamics/hybrid_automaton/hybrid_automaton.h"
#include "spec_testing.h"
#include "testing.h"

#include <cmath>

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kG = 9.81, kBounce = 0.75;

/// The flow of the ball: a LinearSys, a LinParamSys, or the same equations as a NonlinearSys.
enum class BallFlow { linear, parametric, nonlinear };

/// The ball on the floor x >= 0: one location, which the bounce enters again; its flow a
/// LinearSys, a LinParamSys whose x' = (1 +- 0.001) v holds the nominal ball, or a NonlinearSys.
HybridAutomaton aux_ball(BallFlow kind = BallFlow::linear) {
    const Polytope floor(Tensor({{-1.0, 0.0}}), column({0.0}));
    const Polytope guard(Tensor({{0.0, 1.0}}), column({0.0}), Tensor({{1.0, 0.0}}), column({0.0}));
    const LinearReset bounce(Tensor({{1.0, 0.0}, {0.0, -kBounce}}));
    const std::vector<Transition> trans{Transition(guard, bounce, 0)};
    const Tensor A({{0.0, 1.0}, {0.0, 0.0}});
    if (kind == BallFlow::nonlinear) {
        const NonlinearSys sys(
            [](const std::vector<Expr> &x) { return std::vector<Expr>{x[1], Expr(-kG)}; }, 2);
        return HybridAutomaton("ball", {Location("fall", floor, trans, sys)});
    }
    if (kind == BallFlow::parametric) {
        const MatZonotope set(A, Tensor({{0.0, 0.001}, {0.0, 0.0}}).expand_dims(0));
        const LinParamSys sys(set, Tensor::zeros({2, 1}), column({0.0, -kG}));
        return HybridAutomaton("ball", {Location("fall", floor, trans, sys)});
    }
    const Location fall("fall", floor, trans, LinearSys(A, Tensor::zeros({2, 1})),
                        column({0.0, -kG}));
    return HybridAutomaton("ball", {fall});
}

/// The ball dropped from heights in [1, 1.2] at rest, up to 1.5 s.
HybridReachProblem aux_drop() {
    HybridReachProblem p;
    p.initial_set = Interval(column({1.0, 0.0}), column({1.2, 0.0})).zonotope();
    p.t_final = 1.5;
    return p;
}

alg::HybridStandard aux_alg(alg::GuardIntersect method) {
    alg::HybridStandard a;
    a.time_step = {0.01};
    a.taylor_terms = 6;
    a.zonotope_order = 20;
    a.guard_intersect = std::move(method);
    return a;
}

/// The exact state at time t from (x, 0): falls, bounces with v := -0.75 v.
std::array<double, 2> aux_exact(double x, double t) {
    double v = 0, now = 0;
    for (;;) {
        // the time to the floor from (x, v): x + v s - g s^2 / 2 = 0
        const double s = (v + std::sqrt(v * v + 2 * kG * x)) / kG;
        if (now + s > t) {
            const double dt = t - now;
            return {x + v * dt - kG * dt * dt / 2, v - kG * dt};
        }
        now += s;
        v = -kBounce * (v - kG * s);
        x = 0;
    }
}

/// Whether some set of R whose time range holds t contains the point (up to tol).
bool aux_covered(const HybridReachSet &R, double t, const Tensor &p) {
    for (const LocationReachSet &b : R.branches)
        for (std::size_t k = 0; k < b.time_interval.size(); ++k) {
            const TimeRange &range = b.time_interval_times[k];
            if (t < range[0] || t > range[1]) continue;
            const Interval box = b.time_interval[k]->interval();
            // the box first, then the set
            if ((box.inf() - p).max().item() <= 1e-9 && (p - box.sup()).max().item() <= 1e-9 &&
                contains(*b.time_interval[k], Zonotope(p, p.zeros_like({2, 0})), 1e-9))
                return true;
        }
    return false;
}

/// The first jump at the hit times sqrt(2 x0 / g) of [1, 1.2], with the speed 0.75 sqrt(2 g x0)
/// up; the exact trajectories of random heights lie in the sets of their times.
void ball(const std::string &name, alg::GuardIntersect method, BallFlow kind = BallFlow::linear) {
    const HybridReachSet R = aux_ball(kind).reach(aux_drop(), aux_alg(std::move(method)));
    check(R.verified && R.branches.size() >= 3, name + ": the ball bounces twice by 1.5 s");
    const LocationReachSet &second = R.branches[1];
    const double t_lo = std::sqrt(2 * 1.0 / kG), t_hi = std::sqrt(2 * 1.2 / kG);
    check(second.parent == 0 && second.time_point_times[0][0] <= t_lo &&
              second.time_point_times[0][1] >= t_hi && second.time_point_times[0][0] > t_lo - 0.05,
          name + ": the first bounce between the hit times");
    const Interval start = second.time_point[0]->interval();
    const double v_lo = kBounce * std::sqrt(2 * kG * 1.0), v_hi = kBounce * std::sqrt(2 * kG * 1.2);
    check(start.inf().data()[1] <= v_lo + 1e-6 && start.sup().data()[1] >= v_hi - 1e-6 &&
              start.sup().data()[1] < v_hi + 0.5 && std::abs(start.center().data()[0]) < 0.05,
          name + ": the ball leaves the floor upward at the bounced speed");
    // the property: exact trajectories from random heights at random times are covered
    Rng rng(41);
    for (int k = 0; k < 40; ++k) {
        double u[2];
        rng.uniform(u, 2, 0.0, 1.0);
        const double x0 = 1.0 + 0.2 * u[0], t = 1.45 * u[1];
        const std::array<double, 2> s = aux_exact(x0, t);
        check(aux_covered(R, t, column({s[0], s[1]})), name + ": an exact state is reached");
    }
}

/// The nonlinear ball on polynomial zonotopes (alg "poly"): the sets are polynomial zonotopes, the
/// bounce lies between the hit times, and exact trajectories lie in the boxes of their times.
void ball_poly() {
    alg::HybridStandard a = aux_alg(alg::ZonoGirard{{alg::Enclose::box}});
    a.polynomialization = true;
    a.tensor_order = 3;
    a.error_order = 5;
    a.intermediate_order = 10;
    HybridReachProblem p = aux_drop();
    p.initial_poly_set = PolyZonotope(*p.initial_set);
    p.initial_set.reset();
    const HybridReachSet R = aux_ball(BallFlow::nonlinear).reach(p, a);
    check(R.verified && R.branches.size() >= 3 && R.branches[0].time_interval[0].is<PolyZonotope>(),
          "poly ball: polynomial zonotopes, two bounces by 1.5 s");
    const double t_lo = std::sqrt(2 * 1.0 / kG), t_hi = std::sqrt(2 * 1.2 / kG);
    const TimeRange first = R.branches[1].time_point_times[0];
    check(first[0] <= t_lo && first[1] >= t_hi && first[0] > t_lo - 0.05,
          "poly ball: the first bounce between the hit times");
    // exact trajectories from random heights lie in the boxes of the sets of their times
    Rng rng(43);
    int missed = 0;
    for (int k = 0; k < 40; ++k) {
        double u[2];
        rng.uniform(u, 2, 0.0, 1.0);
        const double t = 1.45 * u[1];
        const std::array<double, 2> exact = aux_exact(1.0 + 0.2 * u[0], t);
        bool covered = false;
        for (const LocationReachSet &b : R.branches)
            for (std::size_t j = 0; j < b.time_interval.size() && !covered; ++j) {
                const Interval box = b.time_interval[j]->interval();
                const std::vector<double> lo = box.inf().data(), hi = box.sup().data();
                covered = t >= b.time_interval_times[j][0] && t <= b.time_interval_times[j][1] &&
                          exact[0] >= lo[0] - 1e-9 && exact[0] <= hi[0] + 1e-9 &&
                          exact[1] >= lo[1] - 1e-9 && exact[1] <= hi[1] + 1e-9;
            }
        missed += covered ? 0 : 1;
    }
    check(missed == 0, "poly ball: exact states lie in the boxes");
    // exactly one initial set, and polynomialization goes with the polynomial one
    HybridReachProblem both = p;
    both.initial_set = aux_drop().initial_set;
    check(test::raises([&] { aux_ball(BallFlow::nonlinear).reach(both, a); }, Error::wrong_value),
          "poly ball: two initial sets");
    // without polynomialization a polynomial start set runs "lin" on it (CORA's reach with alg lin)
    alg::HybridStandard lin = aux_alg(alg::ZonoGirard{});
    lin.polynomialization = false;
    check(aux_ball(BallFlow::nonlinear).reach(p, lin).branches.size() >= 2,
          "poly ball: a polynomial set runs lin without polynomialization");
    // and polynomialization needs the polynomial start set
    alg::HybridStandard poly = aux_alg(alg::ZonoGirard{});
    poly.polynomialization = true;
    check(test::raises([&] { aux_ball(BallFlow::nonlinear).reach(aux_drop(), poly); },
                       Error::wrong_value),
          "poly ball: polynomialization needs a polynomial set");
}

/// A clock reaches t = 1 exactly at a step: the jump uses the set at t = 1, at that time only.
void time_triggered() {
    const Polytope before(Tensor({{0.0, 1.0}}), column({1.0}));
    const Polytope at(Tensor::zeros({0, 2}), Tensor::zeros({0, 1}), Tensor({{0.0, 1.0}}),
                      column({1.0}));
    const LinearSys decay(Tensor({{-1.0, 0.0}, {0.0, 0.0}}), Tensor::zeros({2, 1}));
    const Location first("first", before, {Transition(at, LinearReset(Tensor::eye(2)), 1)}, decay,
                         column({0.0, 1.0}));
    const Location second("second", Polytope(Tensor({{0.0, -1.0}}), column({0.0})), {}, decay,
                          column({0.0, 1.0}));
    const HybridAutomaton HA("clock", {first, second});
    // a flat initial clock: the time-point set at t = 1 lies on the guard
    HybridReachProblem p;
    p.initial_set = Zonotope(column({1.0, 0.0}), Tensor({{0.1}, {0.0}}));
    p.t_final = 1.5;
    const HybridReachSet R = HA.reach(p, aux_alg(alg::ZonoGirard{}));
    check(R.branches.size() == 2 && R.branches[1].location == 1, "clock: one jump");
    const TimeRange jump = R.branches[1].time_point_times[0];
    check(close(jump[0], 1.0, 1e-9) && close(jump[1], 1.0, 1e-9), "clock: at t = 1 exactly");
    const Interval x = R.branches[1].time_point[0]->interval();
    check(close(x.center().data()[0], std::exp(-1.0), 1e-6) && close(x.sup().data()[1], 1.0, 1e-9),
          "clock: the state e^-1 x0 at t = 1");
}

/// A safe set the ball leaves (x <= 1.1) fails at once; a custom predicate that holds is asked
/// on every time-interval set and keeps every branch.
void specifications() {
    const HybridAutomaton HA = aux_ball();
    const alg::HybridStandard a = aux_alg(alg::ZonoGirard{});
    const HybridReachSet low = HA.reach(
        aux_drop(), a, {{Specification::safe_set(test::halfspace(column({1.0, 0.0}), 1.1)), {}}});
    check(!low.verified && low.branches.size() == 1 && low.branches[0].time_interval.size() == 1,
          "specs: a violated safe set stops at the first set");
    // the predicate counts its calls
    int asked = 0;
    const Specification count = Specification::custom([&](const ContSet &) { return ++asked > 0; });
    const HybridReachSet all = HA.reach(aux_drop(), a, {{count, {}}});
    std::size_t sets = 0;
    for (const LocationReachSet &b : all.branches) sets += b.time_interval.size();
    check(all.verified && std::size_t(asked) == sets, "specs: a custom check on every set");
}

/// The constructor and reach refuse wrong input; with the checks off reach is the same.
void checks_and_their_switch() {
    const HybridAutomaton HA = aux_ball();
    const LinearSys flow(Tensor::zeros({2, 2}));
    const Polytope inv(Tensor({{1.0, 0.0}}), column({0.0}));
    // the constructor: at least one location, and targets that exist
    check(test::raises([] { HybridAutomaton("none", {}); }, Error::wrong_input_in_constructor),
          "constructor: no location");
    check(test::raises(
              [&] {
                  HybridAutomaton(
                      "bad", {Location("a", inv, {Transition(inv, LinearReset(Tensor::eye(2)), 3)},
                                       flow)});
              },
              Error::wrong_input_in_constructor),
          "constructor: a target that is no location");
    // reach: the start location and the time horizon
    const alg::HybridStandard a = aux_alg(alg::ZonoGirard{});
    HybridReachProblem p = aux_drop();
    p.start_location = 1;
    check(test::raises([&] { HA.reach(p, a); }, Error::wrong_value), "reach: start location");
    p = aux_drop();
    p.t_final = 0;
    check(test::raises([&] { HA.reach(p, a); }, Error::wrong_value), "reach: t_final");
    p = aux_drop();
    p.input_set = {Zonotope(column({0.0, 0.0}), Tensor::eye(2))};
    check(test::raises([&] { HA.reach(p, a); }, Error::dimension_mismatch), "reach: input set");
    // the algorithm and the specifications against the automaton
    alg::HybridStandard two = a;
    two.time_step = {0.1, 0.1};
    check(test::raises([&] { HA.reach(aux_drop(), two); }, Error::wrong_value),
          "reach: a step size per location");
    // the options of the guard intersections are checked
    check(test::raises([&] { HA.reach(aux_drop(), aux_alg(alg::ConZonotopeGuard{})); },
                       Error::wrong_value),
          "reach: ConZonotopeGuard needs guard_order");
    check(test::raises(
              [&] { HA.reach(aux_drop(), aux_alg(alg::NondetGuard{{}, alg::CutBox::fm, -1})); },
              Error::wrong_value),
          "reach: NondetGuard fm_passes is not negative");
    check(test::raises(
              [&] {
                  HA.reach(
                      aux_drop(), a,
                      {{Specification::safe_set(test::halfspace(column({1.0, 0.0}), 5)), {2}}});
              },
              Error::wrong_value),
          "reach: a specification of a location that is not there");
    const std::size_t on = HA.reach(aux_drop(), a).branches.size();
    ChecksScope off(false);
    check(HA.reach(aux_drop(), a).branches.size() == on, "reach: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    using alg::Enclose;
    ball("zonoGirard box", alg::ZonoGirard{{Enclose::box}});
    ball("zonoGirard box pca", alg::ZonoGirard{{Enclose::box, Enclose::pca}});
    ball("zonoGirard flow", alg::ZonoGirard{{Enclose::flow}});
    ball("nondetGuard pca", alg::NondetGuard{{Enclose::pca}});
    ball("conZonotope box pca", alg::ConZonotopeGuard{{Enclose::box, Enclose::pca}, 5});
    // the cuts boxed by certificates of linear programs (CORA's ranges), by both, and by one pass
    using alg::CutBox;
    ball("conZonotope cert cut", alg::ConZonotopeGuard{{Enclose::box}, 5, CutBox::cert});
    ball("conZonotope both cut", alg::ConZonotopeGuard{{Enclose::box}, 5, CutBox::both, 3});
    ball("nondetGuard cert cut", alg::NondetGuard{{Enclose::box}, CutBox::cert});
    ball("nondetGuard one pass", alg::NondetGuard{{Enclose::box}, CutBox::fm, 1});
    // a LinParamSys flow: CORA's initReach and post in the location
    ball("LinParamSys zonoGirard box", alg::ZonoGirard{{Enclose::box}}, BallFlow::parametric);
    ball("LinParamSys conZonotope box pca", alg::ConZonotopeGuard{{Enclose::box, Enclose::pca}, 5},
         BallFlow::parametric);
    // a NonlinearSys flow: alg "lin" in the location, the flow evaluated by checkFlow
    ball("NonlinearSys nondetGuard pca", alg::NondetGuard{{Enclose::pca}}, BallFlow::nonlinear);
    ball("NonlinearSys zonoGirard flow", alg::ZonoGirard{{Enclose::flow}}, BallFlow::nonlinear);
    ball_poly();
    time_triggered();
    specifications();
    checks_and_their_switch();
    return test::finish("hybrid_automaton reach");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
