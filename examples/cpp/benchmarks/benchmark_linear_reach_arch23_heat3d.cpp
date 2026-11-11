// benchmark_linear_reach_arch23_heat3d - HEAT03 of ARCH 2023, by the Krylov subspace algorithm
//
// MATLAB CORA's benchmark_linear_reach_ARCH23_heat3D_HEAT03.m: the 8000 states of heat20 (data
// written by examples/python/benchmarks/heat3d_data.py) and the script's transposed system
// x' = A' x, y = [c G]' x from e_mid (indMid), whose outputs are the temperature at e_mid of the
// center c and the generators G of R0 (the duality of e_mid' e^{At} x0). The maximum temperature
// over the steps (each set's first row summed plus |rest|, as the script) must lie in
// 0.01716509 +- 1e-4. The script's random simulation and plot are left out.
//
// HEAT04 needs the step 0.001 (HEAT03: 0.005): the enlargement of a time interval grows with the
// number of outputs (1387), and the step 0.005 gives 0.0158 against the true Tmax 0.01161.
// HEAT04 (ARCH 2026, heat50, 125 000 states, Tmax 0.01161) is the same script with heat50: its A
// (860 000 entries, 14 MB as data) is built here from the mesh instead of being stored; it equals
// CORA's models/Cora/heat3D/heat50.mat (860 000 entries, the same product A x for a test vector;
// heat3d_data.py is the same construction).
//
// Syntax:   build/core/benchmarks/benchmark_linear_reach_arch23_heat3d [HEAT03|HEAT04]
//           (CORACPP_HEAT04_T_FINAL=1 shortens the horizon of HEAT04; the exit code then ignores
//           the verdict)
// Outputs:  the line 'Heat3D,HEAT03,result,time', as the MATLAB script; the maximum on stderr
// See also: benchmark_data.h, benchmark_linear_verify_fast_arch23_heat3d.cpp
//
// Authors:       Claude Opus 5.5, Claude Sonnet 5.5
// Written:       09-November-2026
// Last update:   11-November-2026 (HEAT04)
// Reviewed:      ---

#include "benchmark_data.h"

#include <cmath>
#include <cstdlib>
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

/// The instance of heat3d_data.py for samples^3 states built from the mesh: the finite differences
/// A (x fastest, then y, then z; a = 0.01 / d^2, -6 a on the diagonal plus a per insulated side,
/// the outflow a / (1 + 0.5 d) at x = 1 added last), the script's R0 (aux_getInit: 1 on
/// [0, 0.4] x [0, 0.2] x [0, 0.1] of the block, the generators diag(0.1 x0)) and C at the middle.
struct Heat {
    SparseMatrix A;
    Tensor c, G, C;
};

Heat aux_mesh(int64_t samples) {
    const int64_t n = samples * samples * samples;
    const double d = 1.0 / double(samples + 1), a = 0.01 / (d * d), leaving = a / (1 + 0.5 * d);
    std::vector<int64_t> rows, cols;
    std::vector<double> vals;
    for (int64_t z = 0; z < samples; ++z)
        for (int64_t y = 0; y < samples; ++y)
            for (int64_t x = 0; x < samples; ++x) {
                const int64_t i = x + y * samples + z * samples * samples;
                double diag = -6 * a;
                bool outflow = false;
                const int64_t coord[3] = {x, y, z}, stride[3] = {1, samples, samples * samples};
                for (int k = 0; k < 3; ++k) {
                    if (coord[k] > 0) {
                        rows.push_back(i), cols.push_back(i - stride[k]), vals.push_back(a);
                    } else {
                        diag += a;
                    }
                    if (coord[k] < samples - 1) {
                        rows.push_back(i), cols.push_back(i + stride[k]), vals.push_back(a);
                    } else if (k == 0) {
                        outflow = true;
                    } else {
                        diag += a;
                    }
                }
                if (outflow) diag += leaving;
                rows.push_back(i), cols.push_back(i), vals.push_back(diag);
            }
    // the initial region: x in [0, 0.4], y in [0, 0.2], z in [0, 0.1] (0.2 for 5 samples)
    const int64_t nx = int64_t(std::floor(0.4 * double(samples) + 1e-9)) + 1;
    const int64_t ny = int64_t(std::floor(0.2 * double(samples) + 1e-9)) + 1;
    const int64_t nz = int64_t(std::floor(0.1 * double(samples) + 1e-9)) + 1;
    std::vector<int64_t> hot;
    for (int64_t z = 0; z < nz; ++z)
        for (int64_t y = 0; y < ny; ++y)
            for (int64_t x = 0; x < nx; ++x) hot.push_back(x + y * samples + z * samples * samples);
    const int64_t m = int64_t(hot.size());
    std::vector<double> c(size_t(n), 0.0), G(size_t(n * m), 0.0), C(size_t(n), 0.0);
    for (int64_t j = 0; j < m; ++j) c[size_t(hot[j])] = 1.0, G[size_t(hot[j] * m + j)] = 0.1;
    C[size_t((samples / 2) * (1 + samples + samples * samples))] = 1.0;
    return {SparseMatrix(n, n, rows, cols, vals), Tensor::from_data(c, {n, 1}),
            Tensor::from_data(G, {n, m}), Tensor::from_data(C, {1, n})};
}

/// The horizon t_final of HEAT04, or $CORACPP_HEAT04_T_FINAL if set: the coverage build, 3 times
/// slower, would take over the 3600 s of a test (it only needs the files the run executes).
double aux_horizon(double t_final) {
    const char *shorter = std::getenv("CORACPP_HEAT04_T_FINAL");
    return shorter ? std::atof(shorter) : t_final;
}

/// HEAT03/HEAT04: the reach of the transposed system and the maximum temperature against the goal.
bool aux_heat03(const std::string &name) {
    const bool mesh = name == "HEAT04";
    bench::Json meta;
    std::string bin;
    if (!mesh) std::tie(meta, bin) = bench::read_instance_files(name);
    const Heat heat = mesh ? aux_mesh(50)
                           : Heat{bench::read_sparse(meta["matrices"]["A"], bin),
                                  bench::read_tensor(meta["matrices"], bin, "R0c"),
                                  bench::read_tensor(meta["matrices"], bin, "R0G"),
                                  bench::read_tensor(meta["matrices"], bin, "C")};
    const SparseMatrix At = heat.A.transpose();
    const int64_t n = At.rows();
    const Tensor &c = heat.c, &G = heat.G, &C = heat.C;
    const double goal = mesh ? 0.01161 : 0.01716509,
                 t_final = mesh ? aux_horizon(40) : meta["tFinal"].number;
    const std::string benchmark = mesh ? "Heat3D" : meta["benchmark"].string;
    const std::string label = mesh ? name : meta["label"].string;
    // the transposed system and its initial point C' (CORA's sysTrans and params.R0)
    const LinearSys sys(At, Tensor::zeros({n, 1}), Tensor::concat({c, G}, -1).matrix_transpose());
    const ReachProblem problem{.initial_set = Zonotope(C.matrix_transpose(), Tensor::zeros({n, 0})),
                               .t_final = t_final};
    const alg::Krylov krylov{.time_step = mesh ? 0.001 : 0.005,
                             .taylor_terms = 10,
                             .zonotope_order = 2,
                             .krylov_error = std::numeric_limits<double>::epsilon(),
                             .krylov_step = 20};

    const auto start = std::chrono::steady_clock::now();
    // the maximum temperature over the steps, in the goal + [-1e-4, 1e-4]; the sets are not kept
    // (HEAT04: 1387 outputs, 2774 generators per set, 8000 steps)
    double maximum = -std::numeric_limits<double>::infinity();
    sys.reach(problem, krylov,
              [&](const Zonotope &Z) { maximum = std::max(maximum, aux_temperature(Z)); });
    const std::chrono::duration<double> seconds = std::chrono::steady_clock::now() - start;
    const bool verified = std::abs(maximum - goal) <= 1e-4;
    bench::print_line(benchmark, label, verified, seconds.count());
    std::cerr << "  maximum temperature " << maximum << " (goal " << goal << " +- 1e-4)\n";
    return mesh ? (verified || std::getenv("CORACPP_HEAT04_T_FINAL") != nullptr)
                : verified == (meta["expected"]["verified"].integer() == 1);
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main(int argc, char **argv) {
    // the instance of the family; the exit code counts the verdicts that differ from MATLAB's
    return cora::bench::run_family(argc, argv, {"HEAT03", "HEAT04"}, aux_heat03);
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
