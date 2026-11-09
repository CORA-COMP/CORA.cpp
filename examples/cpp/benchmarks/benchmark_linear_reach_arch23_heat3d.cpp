// benchmark_linear_reach_arch23_heat3d - HEAT03 of ARCH 2023, by the Krylov subspace algorithm
//
// MATLAB CORA's benchmark_linear_reach_ARCH23_heat3D_HEAT03.m: the 8000 states of heat20 (data
// written by examples/python/benchmarks/heat3d_data.py) and the script's transposed system
// x' = A' x, y = [c G]' x from e_mid (indMid), whose outputs are the temperature at e_mid of the
// center c and the generators G of R0 (the duality of e_mid' e^{At} x0). The maximum temperature
// over the steps (each set's first row summed plus |rest|, as the script) must lie in
// 0.01716509 +- 1e-4. The script's random simulation and plot are left out.
//
// Syntax:   build/core/benchmarks/benchmark_linear_reach_arch23_heat3d [HEAT03]
// Outputs:  the line 'Heat3D,HEAT03,result,time', as the MATLAB script; the maximum on stderr
// See also: benchmark_data.h, benchmark_linear_verify_fast_arch23_heat3d.cpp
//
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "benchmark_data.h"

#include <cmath>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using namespace cora;

/// The script's maximum of one set Z of [c G]' x: the first row of [c, G] summed plus the
/// absolute values of the other rows (its zonotope of [sum(row 1); rows 2..p]).
double aux_temperature(const Zonotope &Z) {
    const Tensor R = Tensor::concat({Z.center(), Z.generators()}, -1);
    const int64_t p = R.shape()[0];
    return R.slice(0, 0, 1).sum().item() + R.slice(0, 1, p).abs().sum().item();
}

/// HEAT03: the reach of the transposed system and the maximum temperature against the goal.
bool aux_heat03(const std::string &name) {
    const auto [meta, bin] = bench::read_instance_files(name);
    const bench::Json &mat = meta["matrices"];
    const SparseMatrix At = bench::read_sparse(mat["A"], bin).transpose();
    const int64_t n = At.rows();
    const Tensor c = bench::read_tensor(mat, bin, "R0c"), G = bench::read_tensor(mat, bin, "R0G");
    const Tensor C = bench::read_tensor(mat, bin, "C");
    // the transposed system and its initial point C' (CORA's sysTrans and params.R0)
    const LinearSys sys(At, Tensor::zeros({n, 1}), Tensor::concat({c, G}, -1).matrix_transpose());
    const ReachProblem problem{.initial_set = Zonotope(C.matrix_transpose(), Tensor::zeros({n, 0})),
                               .t_final = meta["tFinal"].number};
    const alg::Krylov krylov{.time_step = 0.005,
                             .taylor_terms = 10,
                             .zonotope_order = 2,
                             .krylov_error = std::numeric_limits<double>::epsilon(),
                             .krylov_step = 20};

    const auto start = std::chrono::steady_clock::now();
    const ReachSet R = sys.reach(problem, krylov);
    const std::chrono::duration<double> seconds = std::chrono::steady_clock::now() - start;
    // the maximum temperature over the steps, in the goal 0.01716509 + [-1e-4, 1e-4]
    double maximum = -std::numeric_limits<double>::infinity();
    for (const Zonotope &Z : R.time_interval) maximum = std::max(maximum, aux_temperature(Z));
    const bool verified = std::abs(maximum - 0.01716509) <= 1e-4;
    bench::print_line(meta["benchmark"].string, meta["label"].string, verified, seconds.count());
    std::cerr << "  maximum temperature " << maximum << " (goal 0.01716509 +- 1e-4)\n";
    return verified == (meta["expected"]["verified"].integer() == 1);
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main(int argc, char **argv) {
    // the instance of the family; the exit code counts the verdicts that differ from MATLAB's
    return cora::bench::run_family(argc, argv, {"HEAT03"}, aux_heat03);
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
