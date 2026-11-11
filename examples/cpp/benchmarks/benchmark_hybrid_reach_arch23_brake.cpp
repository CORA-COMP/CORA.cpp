// benchmark_hybrid_reach_arch23_brake - electro-mechanical brake benchmark of ARCH 2023
//
// The instances of MATLAB CORA's benchmark_hybrid_reach_ARCH23_brake_<instance>.m, the model
// copied from the scripts: BRKDC01 samples the controller every T_sample by 1000 reach calls of
// the linear system and the reset between them; BRKNC01 is the hybrid automaton whose clock
// x5 jumps within [T_sample + t_min, T_sample + t_max]. Both check x2 <= x0 on every set.
// BRKNP01 is BRKNC01 with a LinParamSys flow; MATLAB CORA's verdict is 0, as here: the curvature
// of its constant input is unbounded at this step size and reach raises reach_set_explosion.
//
// Syntax:   build/core/benchmarks/benchmark_hybrid_reach_arch23_brake [instance ...]
// Outputs:  one line 'benchmark,instance,result,time' per instance, as the MATLAB scripts
// See also: benchmark_hybrid.h
//
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "benchmark_hybrid.h"

#include <chrono>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using namespace cora;
using cora::bench::column;
using cora::bench::matrix;

/// The parameters of the scripts.
constexpr double L = 0.001, K_P = 10000, K_I = 1000, R = 0.5, K = 0.02, d_rot = 0.1;
constexpr double i_gear = 113.1167, T_sample = 0.0001, x0 = 0.05;

/// The rows of A of the four states of the plant, padded with `pad` zero columns and rows.
Tensor aux_A(int64_t pad) {
    const int64_t n = 4 + pad;
    std::vector<double> A(std::size_t(n * n), 0.0);
    A[0] = -(R + K * K / d_rot) / L;
    A[2] = K_P / L;
    A[3] = K_I / L;
    A[std::size_t(n)] = K / (i_gear * d_rot);
    return matrix(n, n, A);
}

/// Whether x2 <= x0 on every set (the scripts' check).
bool aux_below(const std::vector<Set> &sets) {
    for (const Set &S : sets)
        if (S->project({1})->interval().sup().item() > x0) return false;
    return true;
}

/// BRKDC01: 1000 periods, each a reach of T_sample and the reset of the sampled controller.
bool aux_brkdc01() {
    const LinearSys sys(aux_A(0), column({0, 0, 0, 0}));
    const Tensor resetA = matrix(4, 4, {1, 0, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, -T_sample, 0, 1});
    const Tensor resetc = column({0, 0, x0, T_sample * x0});
    ReachProblem problem{Zonotope(column({0, 0, 0, 0}), Tensor::zeros({4, 0})), std::nullopt,
                         T_sample};
    const alg::Standard a{T_sample / 5, 5, 20};
    std::vector<Set> sets;
    for (int k = 0; k < 1000; ++k) {
        const ReachSet Rk = sys.reach(problem, a);
        sets.insert(sets.end(), Rk.time_interval.begin(), Rk.time_interval.end());
        problem.initial_set = Rk.time_point.back().linear_map(resetA) + resetc;
    }
    return aux_below(sets);
}

/// The sampled controller of BRKNC01 and BRKNP01: the clock x5 jumps nondeterministically within
/// [T + t_min, T + t_max], and the transition resets x3, x4 and x5.
struct Cycle {
    Polytope invariant, guard;
    LinearReset reset;
};

Cycle aux_cycle() {
    const double t_min = -1e-8, t_max = 1e-7;
    return {Polytope(matrix(1, 5, {0, 0, 0, 0, 1}), column({T_sample + t_max})),
            Polytope(matrix(2, 5, {0, 0, 0, 0, 1, 0, 0, 0, 0, -1}),
                     column({T_sample + t_max, -(T_sample + t_min)})),
            LinearReset(matrix(5, 5, {1, 0, 0, 0,         0, 0, 1, 0, 0, 0, 0, -1, 0,
                                      0, 0, 0, -T_sample, 0, 1, 0, 0, 0, 0, 0, 1}),
                        column({0, 0, 0, 0, 0}), column({0, 0, x0, T_sample * x0, -T_sample}))};
}

/// The options of both: conZonotope guards of order 5, enclosed by box and pca; the cuts are boxed
/// by the dual certificates of linear programs (`CutBox::cert`, as tight as CORA's ranges): the
/// Fourier-Motzkin boxes lose the verdict of BRKNC01.
alg::HybridStandard aux_options(int taylor_terms) {
    alg::HybridStandard a;
    a.time_step = {T_sample / 5};
    a.taylor_terms = taylor_terms;
    a.zonotope_order = 20;
    a.guard_intersect =
        alg::ConZonotopeGuard{{alg::Enclose::box, alg::Enclose::pca}, 5, alg::CutBox::cert};
    return a;
}

/// BRKNC01: the clock x5 jumps nondeterministically within [T + t_min, T + t_max].
bool aux_brknc01() {
    const Cycle c = aux_cycle();
    const Location loc("loc1", c.invariant, {Transition(c.guard, c.reset, 0)},
                       LinearSys(aux_A(1), column({0, 0, 0, 0, 0})), column({0, 0, 0, 0, 1}));
    const HybridAutomaton HA("brake_BRKNC01", {loc});
    HybridReachProblem problem;
    problem.initial_set = Zonotope(column({0, 0, 0, 0, 0}), Tensor::eye(5) * 1e-8);
    problem.t_final = 0.1;
    const HybridReachSet Rs = HA.reach(problem, aux_options(5));
    std::cerr << "  " << Rs.branches.size() << " branches\n";
    return Rs.verified && aux_below(cora::bench::all_sets(Rs));
}

/// BRKNP01: BRKNC01 with the entry A_11 uncertain by +-3 (a LinParamSys, constParam) and the input
/// U = {1} into x5. The script finds the last period whose sets stay below x0 (the break leaves
/// the inner loop only) and holds iff its end is before 0.1.
bool aux_brknp01() {
    // the matrix zonotope of the script: A with the generator 3 at (1, 1)
    const Cycle c = aux_cycle();
    std::vector<double> spread(25, 0.0);
    spread[0] = 3;
    const LinParamSys flow(MatZonotope(aux_A(1), matrix(5, 5, spread).expand_dims(0)),
                           column({0, 0, 0, 0, 1}));
    const Location loc("loc1", c.invariant, {Transition(c.guard, c.reset, 0)}, flow);
    const HybridAutomaton HA("brake_BRKNP01", {loc});
    HybridReachProblem problem;
    problem.initial_set = Zonotope(column({0, 0, 0, 0, 0}), Tensor::eye(5) * 1e-8);
    problem.input_set = {Zonotope(column({1}), Tensor::zeros({1, 0}))};
    problem.t_final = 0.1;
    alg::HybridStandard a = aux_options(10);
    a.intermediate_terms = 3;
    // the curvature of U = {1} is unbounded here (cora-differences.md): no verdict "verified"
    std::optional<HybridReachSet> reached;
    try {
        reached = HA.reach(problem, a);
    } catch (const Error &e) {
        if (e.id() != Error::reach_set_explosion) throw;
        std::cerr << "  " << e.what() << '\n';
        return false;
    }
    const HybridReachSet &Rs = *reached;
    // the period t of each branch; the last set below x0 sets t_max
    double t_max = 0, t = 0;
    for (const LocationReachSet &branch : Rs.branches) {
        for (const Set &S : branch.time_interval) {
            const double sup_x = S->project({1})->interval().sup().item();
            if (sup_x < x0) {
                t_max = S->project({4})->interval().sup().item() + t;
                break;
            }
        }
        t += T_sample;
    }
    std::cerr << "  " << Rs.branches.size() << " branches, max. time horizon " << t_max << '\n';
    return t_max < 0.1;
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main(int argc, char **argv) {
    // MATLAB CORA verifies all but BRKNP01 (verdict 0 expected)
    const std::vector<std::string> family = {"BRKDC01", "BRKNC01", "BRKNP01"};
    return cora::bench::run_family(argc, argv, family, [](const std::string &name) {
        const auto start = std::chrono::steady_clock::now();
        bool verified = false;
        if (name == "BRKDC01")
            verified = aux_brkdc01();
        else if (name == "BRKNC01")
            verified = aux_brknc01();
        else if (name == "BRKNP01")
            verified = aux_brknp01();
        else
            CORA_FAIL(Error::wrong_value, "brake: unknown instance " + name);
        const std::chrono::duration<double> wall = std::chrono::steady_clock::now() - start;
        cora::bench::print_line("Brake", name, verified, wall.count());
        return verified == (name != "BRKNP01");
    });
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
