// benchmark_hybrid_reach_arch23_gearbox - gearbox benchmark of ARCH 2023
//
// The instances of MATLAB CORA's benchmark_hybrid_reach_ARCH23_gearbox_<instance>.m with the
// automaton of models/Cora/gearbox.m (its formulas copied): the sleeve (velocities v_x, v_y,
// positions p_x, p_y, impulse I) bounces off the teeth of the second gear until it meshes. The
// meshed location is final, so reach stops at it; the script then checks I < 20 on every set.
//
// Syntax:   build/core/benchmarks/benchmark_hybrid_reach_arch23_gearbox [instance ...]
// Outputs:  one line 'benchmark,instance,result,time' per instance, as the MATLAB scripts
// See also: benchmark_hybrid.h
//
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "benchmark_hybrid.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using namespace cora;
using cora::bench::column;
using cora::bench::matrix;

/// The automaton of models/Cora/gearbox.m.
HybridAutomaton aux_gearbox() {
    // parameters
    const double theta = 0.9, m_s = 3.2, m_g2 = 18.1, J_g2 = 0.7, R_s = 0.08;
    const double Theta = 36.0 / 180 * (4 * std::atan(1.0)), b = 0.01, delta_p = -0.003;
    const double n = 0, F_s = 70, T_f = 1;
    // the flow: x' = A x + B F_s + (0, -R_s T_f / J_g2, 0, 0, 0)
    const Tensor A =
        matrix(5, 5, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0});
    const Tensor B = column({1 / m_s, 0, 0, 0, 0});
    const Tensor c = column({F_s / m_s, -R_s * T_f / J_g2, 0, 0, 0});
    const double tanT = std::tan(Theta), sinT = std::sin(Theta), cosT = std::cos(Theta);
    const double denominator = m_s * cosT * cosT + m_g2 * sinT * sinT;
    // transition 1: the upper tooth flank
    const Polytope guard1(matrix(3, 5, {0, 0, 1, 0, 0, 0, 0, -1, 0, 0, -sinT, -cosT, 0, 0, 0}),
                          column({delta_p, b / tanT, 0}), matrix(1, 5, {0, 0, -tanT, -1, 0}),
                          column({-2 * n * b}));
    // the impact maps of the velocities; the flanks differ in the signs of r12, r21 and r52
    const double r11 = (m_s * cosT * cosT - m_g2 * theta * sinT * sinT) / denominator;
    const double r12 = (-1 * (theta + 1) * m_g2 * sinT * cosT) / denominator;
    const double r21 = (-1 * (theta + 1) * m_s * sinT * cosT) / denominator;
    const double r22 = (m_g2 * sinT * sinT - m_s * theta * cosT * cosT) / denominator;
    const double r51 = ((theta + 1) * m_s * m_g2 * sinT) / denominator;
    const double r52 = ((theta + 1) * m_s * m_g2 * cosT) / denominator;
    const LinearReset reset1(matrix(5, 5, {r11, r12, 0, 0, 0, r21, r22, 0,   0,   0, 0, 0, 1,
                                           0,   0,   0, 0, 0, 1,   0,   r51, r52, 0, 0, 1}));
    // transition 2: the lower tooth flank
    const Polytope guard2(matrix(3, 5, {0, 0, 1, 0, 0, 0, 0, -1, 0, 0, -sinT, cosT, 0, 0, 0}),
                          column({delta_p, b / tanT, 0}), matrix(1, 5, {0, 0, -tanT, 1, 0}),
                          column({2 * n * b}));
    const LinearReset reset2(matrix(5, 5, {r11, -r12, 0, 0, 0, -r21, r22, 0,   0,    0, 0, 0, 1,
                                           0,   0,    0, 0, 0, 1,    0,   r51, -r52, 0, 0, 1}));
    // transition 3: meshed
    const Polytope guard3(Tensor::zeros({0, 5}), Tensor::zeros({0, 1}),
                          matrix(1, 5, {0, 0, -1, 0, 0}), column({-delta_p}));
    const LinearReset reset3(matrix(
        5, 5, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, m_s, m_s, 0, 0, 1}));
    // the invariant: between the flanks, before the mesh, with a margin of 1e-3
    const Tensor Cinv = matrix(3, 5, {0, 0, tanT, 1, 0, 0, 0, tanT, -1, 0, 0, 0, 1, 0, 0});
    const Tensor dinv = column({-2 * n * b + 1e-3, 2 * n * b + 1e-3, b / tanT + 1e-3});
    const Polytope inv(Cinv, dinv);
    // one location with the three transitions; the meshed one leads to the final location
    Location loc1("loc1", inv,
                  {Transition(guard1, reset1, 0), Transition(guard2, reset2, 0),
                   Transition(guard3, reset3, 1)},
                  LinearSys(A, B), c);
    Location loc2("loc2", inv, {}, LinearSys(Tensor::zeros({5, 5}), Tensor::zeros({5, 1})));
    return HybridAutomaton("gearbox", {loc1, loc2});
}

cora::bench::HybridInstance aux_instance(const std::string &name) {
    HybridReachProblem problem;
    if (name == "GRBX01")
        problem.initial_set =
            Interval(column({0, 0, -0.0168, 0.0029, 0}), column({0, 0, -0.0166, 0.0031, 0}))
                .zonotope();
    else if (name == "GRBX02")
        problem.initial_set =
            Interval(column({0, 0, -0.01675, 0.00285, 0}), column({0, 0, -0.01665, 0.00315, 0}))
                .zonotope();
    else
        CORA_FAIL(Error::wrong_value, "gearbox: unknown instance " + name);
    // the meshed location is final; the options of the script
    problem.final_locations = {1};
    problem.t_final = 0.2;
    alg::HybridStandard a;
    a.time_step = {0.2 / 182};
    a.zonotope_order = 20;
    a.taylor_terms = 3;
    a.guard_intersect = alg::ZonoGirard{{alg::Enclose::box, alg::Enclose::pca}};
    // the script's check: the impulse x5 stays below 20 on every set
    auto check = [](const HybridReachSet &R) {
        for (const Set &S : cora::bench::all_sets(R))
            if (S->project({4})->interval().sup().item() >= 20) return false;
        return true;
    };
    return {"Gearbox", name, aux_gearbox(), problem, a, {}, check, true};
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main(int argc, char **argv) {
    return cora::bench::run_family(argc, argv, {"GRBX01", "GRBX02"}, aux_instance);
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
