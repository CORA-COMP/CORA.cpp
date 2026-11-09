// benchmark_hybrid_reach_arch23_brake - electro-mechanical brake benchmark of ARCH 2023
//
// The instances of MATLAB CORA's benchmark_hybrid_reach_ARCH23_brake_<instance>.m, the model
// copied from the scripts: BRKDC01 samples the controller every T_sample by 1000 reach calls of
// the linear system and the reset between them; BRKNC01 is the hybrid automaton whose clock
// x5 jumps within [T_sample + t_min, T_sample + t_max]. Both check x2 <= x0 on every set.
// BRKNP01 needs LinParamSys (the uncertain parameters) and comes with it.
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

/// BRKNC01: the clock x5 jumps nondeterministically within [T + t_min, T + t_max].
bool aux_brknc01() {
    const double t_min = -1e-8, t_max = 1e-7;
    // the sampled controller resets x3, x4 and the clock x5
    const LinearReset reset(matrix(5, 5, {1, 0, 0, 0,         0, 0, 1, 0, 0, 0, 0, -1, 0,
                                          0, 0, 0, -T_sample, 0, 1, 0, 0, 0, 0, 0, 1}),
                            column({0, 0, 0, 0, 0}), column({0, 0, x0, T_sample * x0, -T_sample}));
    const Polytope inv(matrix(1, 5, {0, 0, 0, 0, 1}), column({T_sample + t_max}));
    const Polytope guard(matrix(2, 5, {0, 0, 0, 0, 1, 0, 0, 0, 0, -1}),
                         column({T_sample + t_max, -(T_sample + t_min)}));
    const Location loc("loc1", inv, {Transition(guard, reset, 0)},
                       LinearSys(aux_A(1), column({0, 0, 0, 0, 0})), column({0, 0, 0, 0, 1}));
    const HybridAutomaton HA("brake_BRKNC01", {loc});
    // reach as the script: conZonotope guards of order 5, enclosed by box and pca
    HybridReachProblem problem;
    problem.initial_set = Zonotope(column({0, 0, 0, 0, 0}), Tensor::eye(5) * 1e-8);
    problem.t_final = 0.1;
    alg::HybridStandard a;
    a.time_step = {T_sample / 5};
    a.taylor_terms = 5;
    a.zonotope_order = 20;
    a.guard_intersect = alg::ConZonotopeGuard{{alg::Enclose::box, alg::Enclose::pca}, 5};
    const HybridReachSet Rs = HA.reach(problem, a);
    std::cerr << "  " << Rs.branches.size() << " branches\n";
    return Rs.verified && aux_below(cora::bench::all_sets(Rs));
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main(int argc, char **argv) {
    const std::vector<std::string> family = {"BRKDC01", "BRKNC01"};
    const std::vector<std::string> names =
        argc < 2 ? family : std::vector<std::string>(argv + 1, argv + argc);
    int differ = 0;
    for (const std::string &name : names) {
        const auto start = std::chrono::steady_clock::now();
        bool verified = false;
        if (name == "BRKDC01")
            verified = aux_brkdc01();
        else if (name == "BRKNC01")
            verified = aux_brknc01();
        else
            CORA_FAIL(Error::wrong_value, "brake: unknown instance " + name);
        const std::chrono::duration<double> wall = std::chrono::steady_clock::now() - start;
        std::cout << "Brake," << name << "," << verified << "," << wall.count() << '\n';
        if (!verified) ++differ; // MATLAB CORA: 1 for both
    }
    return differ;
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
