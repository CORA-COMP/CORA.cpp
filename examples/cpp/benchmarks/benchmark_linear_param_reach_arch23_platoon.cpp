// benchmark_linear_param_reach_arch23_platoon - platoon benchmark of ARCH 2023
//
// The instances of MATLAB CORA's benchmark_linearParam_reach_ARCH23_platoon_<instance>.m: a
// platoon of three cars whose communication may fail, a controlled system A_c and an
// uncontrolled one A_n. PLAA01 lets the system switch arbitrarily, as the matrix zonotope
// (A_c + A_n) / 2 +- (A_c - A_n) / 2 with constant parameters; PLAN01 the same with varying
// parameters, then reaches on from an enlargement of the last set until a set holds it (an
// invariant). PLAD01 switches every 5 s, c, n, c, n, with two linear systems. Every time-interval
// set must keep x1, x4 and x7 above -bound. As the script, PLAD01-BND30 checks only the first
// set of each of its four reach calls (length(R(i)) is 1).
//
// Syntax:   build/core/benchmarks/benchmark_linear_param_reach_arch23_platoon [instance ...]
// Outputs:  one line 'benchmark,instance,result,time' per instance, as the MATLAB scripts
// See also: benchmark_run.h, LinParamSys
//
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "benchmark_hybrid.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using namespace cora;
using cora::bench::column;
using cora::bench::matrix;

// The model ---------------------------------------------------------------------------------------

/// The controlled system A_c of the scripts.
Tensor aux_controlled() {
    return matrix(9, 9,
                  {0,      1.0000, 0,       0,       0,      0,       0,       0,      0,
                   0,      0,      -1.0000, 0,       0,      0,       0,       0,      0,
                   1.6050, 4.8680, -3.5754, -0.8198, 0.4270, -0.0450, -0.1942, 0.3626, -0.0946,
                   0,      0,      0,       0,       1.0000, 0,       0,       0,      0,
                   0,      0,      1.0000,  0,       0,      -1.0000, 0,       0,      0,
                   0.8718, 3.8140, -0.0754, 1.1936,  3.6258, -3.2396, -0.5950, 0.1294, -0.0796,
                   0,      0,      0,       0,       0,      0,       0,       1.0000, 0,
                   0,      0,      0,       0,       0,      1.0000,  0,       0,      -1.0000,
                   0.7132, 3.5730, -0.0964, 0.8472,  3.2568, -0.0876, 1.2726,  3.0720, -3.1356});
}

/// The uncontrolled system A_n: the controlled one without the feedback of the other cars.
Tensor aux_uncontrolled() {
    return matrix(
        9, 9,
        {0,      1.0000, 0,       0,      0,      0,       0,      0,      0,       0, 0, -1.0000,
         0,      0,      0,       0,      0,      0,       1.6050, 4.8680, -3.5754, 0, 0, 0,
         0,      0,      0,       0,      0,      0,       0,      1.0000, 0,       0, 0, 0,
         0,      0,      1.0000,  0,      0,      -1.0000, 0,      0,      0,       0, 0, 0,
         1.1936, 3.6258, -3.2396, 0,      0,      0,       0,      0,      0,       0, 0, 0,
         0,      1.0000, 0,       0,      0,      0,       0,      0,      1.0000,  0, 0, -1.0000,
         0.7132, 3.5730, -0.0964, 0.8472, 3.2568, -0.0876, 1.2726, 3.0720, -3.1356});
}

/// The input matrix B, the initial set (the origin) and the input set [-9, 1].
Tensor aux_B() { return column({0, 1, 0, 0, 0, 0, 0, 0, 0}); }
Zonotope aux_R0() { return Zonotope(Tensor::zeros({9, 1}), Tensor::zeros({9, 0})); }
Zonotope aux_U() { return Interval(column({-9}), column({1})).zonotope(); }

/// The arbitrarily switching system: the matrix zonotope A_mid +- A_gen.
LinParamSys aux_switching(ParamType type) {
    const Tensor A_c = aux_controlled(), A_n = aux_uncontrolled();
    const Tensor A_mid = (A_c + A_n) * 0.5, A_gen = (A_c - A_n) * 0.5;
    return LinParamSys(MatZonotope(A_mid, A_gen.expand_dims(0)), aux_B(), std::nullopt, type);
}

// The checks --------------------------------------------------------------------------------------

/// The lowest bound of x1, x4 and x7 over the sets.
double aux_lowest(const std::vector<Zonotope> &sets) {
    double low = std::numeric_limits<double>::infinity();
    for (const Zonotope &S : sets)
        for (double v : S.project({0, 3, 6})->interval().inf().data()) low = std::min(low, v);
    return low;
}

/// Whether x1, x4 and x7 stay above -bound on every set.
bool aux_above(const std::vector<Zonotope> &sets, double bound) {
    return aux_lowest(sets) >= -bound;
}

// The instances -----------------------------------------------------------------------------------

/// PLAA01: the switching system with constant parameters over 20 s.
bool aux_plaa(bool bnd42) {
    alg::LinParamStandard a;
    a.intermediate_terms = bnd42 ? 3 : 2;
    a.time_step = bnd42 ? 20.0 / 2222 : 0.02;
    a.zonotope_order = bnd42 ? 800 : 400;
    a.taylor_terms = bnd42 ? 3 : 4;
    a.comp_time_point = false;
    const ReachSet R = aux_switching(ParamType::constant).reach({aux_R0(), aux_U(), 20}, a);
    return aux_above(R.time_interval, bnd42 ? 42 : 50);
}

/// PLAD01: c, n, c, n for 5 s each, every call from the last set of the one before.
bool aux_plad(bool bnd42) {
    const LinearSys sys_c(aux_controlled(), aux_B()), sys_n(aux_uncontrolled(), aux_B());
    const alg::Standard a{0.02, 4, bnd42 ? 20 : 200};
    ReachProblem problem{aux_R0(), aux_U(), 5};
    std::vector<Zonotope> checked;
    for (int i = 0; i < 4; ++i) {
        const ReachSet R = (i % 2 == 0 ? sys_c : sys_n).reach(problem, a);
        // BND42 checks every set, BND30 the first of each call (the script's length(R(i)))
        if (bnd42)
            checked.insert(checked.end(), R.time_interval.begin(), R.time_interval.end());
        else
            checked.push_back(R.time_interval.front());
        problem.initial_set = R.time_point.back();
    }
    return aux_above(checked, bnd42 ? 42 : 30);
}

/// Whether every projection of Z1 onto two axes contains that of Z2 (the script's
/// aux_inViaProj: its comment says Z1 in Z2, its call contains(Z1proj, Z2proj) the converse).
bool aux_in_via_proj(const Zonotope &Z1, const Zonotope &Z2) {
    const int64_t n = Z1.dim();
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = i + 1; j < n; ++j)
            if (!Z1.project({i, j}).as<Zonotope>().contains(Z2.project({i, j}).as<Zonotope>()))
                return false;
    return true;
}

/// PLAN01: 50 s with varying parameters, then the sets from 1.01 times the last until a set
/// holds that start set in every projection (the script's aux_reachInv).
bool aux_plan() {
    const LinParamSys sys = aux_switching(ParamType::varying);
    alg::LinParamStandard a;
    a.intermediate_terms = 3;
    a.time_step = 0.01;
    a.zonotope_order = 400;
    a.taylor_terms = 3;
    a.comp_time_point = false;
    const ReachSet R = sys.reach({aux_R0(), aux_U(), 50}, a);
    double low = aux_lowest(R.time_interval);
    std::cerr << "  reach: lowest x1, x4, x7 " << low << '\n';
    // the invariant: the steps from the enlarged last set, the input without its center
    const Zonotope R0 = R.time_interval.back().reduce(400).enlarge(1.01).as<Zonotope>();
    a.zonotope_order = 800;
    const Zonotope U = aux_U();
    LinParamStep step = sys.init_reach(R0, U - U.center(), U.center(), a);
    constexpr int kMaxSteps = 1000000; // the script has no bound
    for (int k = 0; k < kMaxSteps; ++k) {
        low = std::min(low, aux_lowest({step.time_interval}));
        step = sys.post(step, a);
        if (aux_in_via_proj(step.time_interval, R0)) {
            std::cerr << "  invariant after " << k + 1 << " steps; lowest " << low << '\n';
            return low >= -50;
        }
    }
    CORA_FAIL(Error::wrong_value, "PLAN01: the sets do not hold the start set after 10^6 steps");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main(int argc, char **argv) {
    const std::vector<std::string> family = {"PLAA01_BND42", "PLAA01_BND50", "PLAD01_BND30",
                                             "PLAD01_BND42", "PLAN01"};
    // each instance by its function; MATLAB CORA verifies all but PLAN01 (R2024b, CORA v2026.1.0)
    return cora::bench::run_family(argc, argv, family, [](const std::string &name) {
        const auto start = std::chrono::steady_clock::now();
        bool verified = false;
        std::string label = name;
        if (name == "PLAA01_BND42" || name == "PLAA01_BND50")
            verified = aux_plaa(name == "PLAA01_BND42");
        else if (name == "PLAD01_BND30" || name == "PLAD01_BND42")
            verified = aux_plad(name == "PLAD01_BND42");
        else if (name == "PLAN01") {
            verified = aux_plan();
            label = "PLAN01_UNB50";
        } else
            CORA_FAIL(Error::wrong_value, "platoon: unknown instance " + name);
        const std::chrono::duration<double> wall = std::chrono::steady_clock::now() - start;
        label[6] = '-';
        cora::bench::print_line("Platoon", label, verified, wall.count());
        return verified == (name != "PLAN01");
    });
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
