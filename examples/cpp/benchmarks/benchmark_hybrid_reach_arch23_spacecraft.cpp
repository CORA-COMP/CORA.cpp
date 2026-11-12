// benchmark_hybrid_reach_arch23_spacecraft - nonlinear spacecraft rendezvous (SPRE22) of ARCH 2023
//
// MATLAB CORA's benchmark_hybrid_reach_ARCH23_spacecraft.m with the automaton of
// models/Cora/spacecraftRendezvous/rendezvous_nonlinear_passive_nondet.m, numbers copied: the
// state is (x, y, vx, vy, t), the flows are the nonlinear orbital equations of the three modes
// (approach, docking, passive), and the passive mode is entered for t in [120, 150]. In the
// docking mode the velocity and the line of sight are bounded, in the passive mode the space
// station must not be hit. Algorithm "lin" with tensor order 2, guards by nondetGuard and PCA.
//
// Syntax:   build/core/benchmarks/benchmark_hybrid_reach_arch23_spacecraft [SPRE22]
// Outputs:  one line 'benchmark,instance,result,time', as the MATLAB script
// See also: benchmark_hybrid.h, benchmark_hybrid_reach_arch23_rendezvous.cpp
//
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "benchmark_hybrid.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using namespace cora;
using cora::bench::column;
using cora::bench::hyperplane;
using cora::bench::matrix;

/// The gravity of the target's orbit in the equations of vx and vy: mu (x + r) / D^(3/2) and
/// mu y / D^(3/2) with D = (x + r)^2 - y^2, as the model files write them.
std::array<Expr, 2> aux_gravity(const std::vector<Expr> &x) {
    const Expr r = x[0] + 42164000;
    const Expr D = pow(pow(r, 2) - pow(x[1], 2), 1.5);
    return {1434960000000000000.0 * r / D, 1434960000000000000.0 * x[1] / D};
}

/// The flow of a mode: x' = vx, y' = vy, vx' and vy' by the coefficients a (of x, y, vx, vy in
/// vx') and b (in vy') plus the gravity and the constant of vx', t' = 1.
NonlinearSys aux_flow(const std::array<double, 4> &a, const std::array<double, 4> &b) {
    return NonlinearSys(
        [a, b](const std::vector<Expr> &x) {
            const auto [gx, gy] = aux_gravity(x);
            const Expr vx = a[0] * x[0] + a[1] * x[1] + a[2] * x[2] + a[3] * x[3] - gx +
                            807.15359572684597539321366928407;
            const Expr vy = b[0] * x[0] + b[1] * x[1] + b[2] * x[2] + b[3] * x[3] - gy;
            return std::vector<Expr>{x[2], x[3], vx, vy, Expr(1.0)};
        },
        5);
}

HybridAutomaton aux_automaton() {
    const LinearReset id(Tensor::eye(5), std::nullopt, column({0, 0, 0, 0, 0}));
    // the passive mode is entered in the window t in [120, 150]
    const Polytope window(matrix(2, 5, {0, 0, 0, 0, -1, 0, 0, 0, 0, 1}), column({-120, 150}));
    const NonlinearSys approach =
        aux_flow({-0.057638256806855922248709263104, 0.000201, -2.8999508, 0.008759786984672},
                 {-0.000174, -0.066493256806855922248709263104, -0.008741346984672, -2.9030026});
    const NonlinearSys docking =
        aux_flow({-0.576038456806855922248709263104, 0.0002624, -19.2299796, 0.008750586984672},
                 {-0.0002624, -0.575980856806855922248709263104, -0.008750586984672, -19.2299766});
    const NonlinearSys passive =
        aux_flow({0.000019143193144077751290736896, 0, 0, 0.008750586984672},
                 {0, 0.000019143193144077751290736896, -0.008750586984672, 0});
    // approach: t <= 150 and x <= -100, left at x = -100 or by the window
    Location S1("S1", Polytope(matrix(2, 5, {0, 0, 0, 0, 1, 1, 0, 0, 0, 0}), column({150, -100})),
                {Transition(hyperplane({-1, 0, 0, 0, 0}, 100), id, 1), Transition(window, id, 2)},
                approach);
    // docking: t <= 150 and the octagon |x|, |y| <= 100, |x +- y| <= 141.1
    const std::vector<double> rows = {0,  0, 0, 0, 1, 0,  -1, 0, 0, 0, -1, -1, 0, 0, 0,
                                      -1, 0, 0, 0, 0, -1, 1,  0, 0, 0, 0,  1,  0, 0, 0,
                                      1,  1, 0, 0, 0, 1,  0,  0, 0, 0, 1,  -1, 0, 0, 0};
    Location S2(
        "S2",
        Polytope(matrix(9, 5, rows), column({150, 100, 141.1, 100, 141.1, 100, 141.1, 100, 141.1})),
        {Transition(window, id, 2)}, docking);
    Location S3("S3",
                Polytope(matrix(2, 5, {1, 0, 0, 0, 0, -1, 0, 0, 0, 0}), column({10000, 10000})), {},
                passive);
    return HybridAutomaton("spacecraft_passive", {S1, S2, S3});
}

/// A safe set {x | C x <= d}; the one place that builds the specifications of this program.
Specification aux_safe(const Tensor &C, const std::vector<double> &d) {
    return Specification::safe_set(Polytope(C, column(d)));
}

/// CORA's aux_checkSpaceStationHit: the projection onto (x, y) must miss the box [-1, 1]^2, first
/// by its box, then reduced to order 3.
Specification aux_station() {
    return Specification::custom([](const ContSet &S) {
        const Interval station(column({-1, -1}), column({1, 1}));
        const Set proj = S.project({0, 1});
        if (!is_intersecting(station, proj->interval())) return true;
        return !is_intersecting(station, *proj->reduce(3));
    });
}

/// Docking: the velocity in the octagon |v| <= 3, |vx +- vy| <= 4.2, and the line of sight
/// within 20 degrees; passive: the station is not hit.
std::vector<HybridSpecification> aux_specs() {
    const Tensor Cv =
        matrix(8, 5, {0, 0, 1, 0, 0, 0, 0, -1, 0,  0, 0, 0, 0,  1, 0, 0, 0, 0,  -1, 0,
                      0, 0, 1, 1, 0, 0, 0, 1,  -1, 0, 0, 0, -1, 1, 0, 0, 0, -1, -1, 0});
    const double s = std::tan(4 * std::atan(1.0) / 9);
    const Tensor Cl = matrix(2, 5, {s, -1, 0, 0, 0, s, 1, 0, 0, 0});
    return {{aux_safe(Cv, {3, 3, 3, 3, 4.2, 4.2, 4.2, 4.2}), {1}},
            {aux_safe(Cl, {0, 0}), {1}},
            {aux_station(), {2}}};
}

cora::bench::HybridInstance aux_instance(const std::string &name) {
    if (name != "SPRE22") CORA_FAIL(Error::wrong_value, "spacecraft: unknown instance " + name);
    HybridReachProblem problem;
    problem.initial_set = Zonotope(column({-900, -400, 2.5, 2.5, 0}),
                                   matrix(5, 5, {25, 0, 0, 0, 0, 0,   25, 0, 0, 0, 0, 0, 2.5,
                                                 0,  0, 0, 0, 0, 2.5, 0,  0, 0, 0, 0, 0}));
    problem.t_final = 200;
    // the options of the script
    alg::HybridStandard a;
    // the script's steps are for CORA's loop; the region check of CORALean needs smaller ones
    a.time_step = cora::bench::error_bound() == alg::ErrorBound::loop
                      ? std::vector<double>{0.2, 0.05, 0.2}
                      : std::vector<double>{0.05, 0.025, 0.05};
    // CORACPP_SPACECRAFT_STEPS = a,b,c overrides the step sizes of the three locations (tuning)
    if (const char *tune = std::getenv("CORACPP_SPACECRAFT_STEPS"))
        std::sscanf(tune, "%lf,%lf,%lf", &a.time_step[0], &a.time_step[1], &a.time_step[2]);
    a.taylor_terms = 10;
    a.zonotope_order = 10;
    a.tensor_order = 2;
    a.linear_step = cora::bench::linear_step();
    a.error_bound = cora::bench::error_bound();
    a.guard_intersect = alg::NondetGuard{{alg::Enclose::pca}};
    return {"Spacecraft", name, aux_automaton(), problem, a, aux_specs(), nullptr, true};
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main(int argc, char **argv) {
    return cora::bench::run_hybrid_family(argc, argv, {"SPRE22"}, aux_instance);
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
