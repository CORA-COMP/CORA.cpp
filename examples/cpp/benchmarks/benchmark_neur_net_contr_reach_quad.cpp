// benchmark_neur_net_contr_reach_quad - QUAD (ARCH-COMP 2026)
//
// CORA: examples/contDynamics/neurNetContrSys/benchmark_neuralNet_reach_09_QUAD.m
//
// Quadrotor: x3 in [0.94, 1.06] at t = 5 from x1..x6 in [-0.4, 0.4], with CORA's poly_method
// regression of order 1 and the splitting of verify. The network is
// examples/benchmarks/data/networks/<name>.nn.json and .nn.bin that
// matlab/benchmarks/export_network_data.m writes from the script; the plant is written out as the
// model file writes it. The line of the scripts, 'benchmark,instance,verdict,time', goes to stdout
// (the time of verify, simulations included), the expected verdict and time of the ARCH-COMP 2026
// report (CORA, Table 1) go to stderr. The exit code counts the verdicts that differ from the
// report's.
//
// Syntax:   build/core/benchmarks/benchmark_neur_net_contr_reach_quad [QUAD-reach ...]
// Outputs:  one line 'benchmark,instance,verdict,time' per instance on stdout
// See also: benchmark_neur_net.h
//
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#include "benchmark_neur_net.h"
#include "benchmark_run.h"
#include "cora/cora.h"
#include <cstdlib>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using namespace cora;
using cora::bench::AinncsInstance;
using cora::bench::box_poly;
using cora::bench::box_polytope;
using cora::bench::kInf;
using cora::bench::loop_options;
using cora::bench::make_verify;

using Vars = std::vector<Expr>;

// The plant ------------------------------------------------------------------------------------

/// dynamics_quad.m (tau = 0) with the report's x8', x9'; CORACPP_QUAD_SCRIPT=1 takes the
/// script's, which differ from the report (cora-differences.md).
NonlinearSys aux_quad() {
    const bool report = std::getenv("CORACPP_QUAD_SCRIPT") == nullptr;
    return {
        [=](const Vars &x, const Vars &u) {
            constexpr double g = 9.81, m = 1.4, Jx = 0.054, Jy = 0.054, Jz = 0.104;
            // the sines and cosines of the angles, then the positions, velocities and rates
            const Expr s7 = sin(x[6]), c7 = cos(x[6]), s8 = sin(x[7]), c8 = cos(x[7]),
                       s9 = sin(x[8]), c9 = cos(x[8]);
            return Vars{
                c8 * c9 * x[3] + (s7 * s8 * c9 - c7 * s9) * x[4] + (c7 * s8 * c9 + s7 * s9) * x[5],
                c8 * s9 * x[3] + (s7 * s8 * s9 + c7 * c9) * x[4] + (c7 * s8 * s9 - s7 * c9) * x[5],
                s8 * x[3] - s7 * c8 * x[4] - c7 * c8 * x[5],
                x[11] * x[4] - x[10] * x[5] - g * s8,
                x[9] * x[5] - x[11] * x[3] + g * c8 * s7,
                x[10] * x[3] - x[9] * x[4] + g * c8 * c7 - g - u[0] / m,
                x[9] + s7 * tan(x[7]) * x[10] + c7 * tan(x[7]) * x[11],
                report ? c7 * x[10] - s7 * x[11] : c7 * x[10] + s7 * x[11],
                report ? s7 / c8 * x[10] - c7 / c8 * x[11] : s7 / c8 * x[10] - s7 * x[11],
                (Jy - Jz) / Jx * x[10] * x[11] + u[1] / Jx,
                (Jz - Jx) / Jy * x[9] * x[11] + u[2] / Jy,
                (Jx - Jy) / Jz * x[9] * x[10] + 0.0 / Jz};
        },
        12, 3};
}

// The instances ----------------------------------------------------------------------------------

std::vector<AinncsInstance> aux_instances() {
    const std::optional<Zonotope> none = std::nullopt;
    std::vector<AinncsInstance> v;
    // QUAD: x3 in [0.94, 1.06] at t = 5, from x1..x6 in [-0.4, 0.4]
    {
        std::vector<double> lo(12, 0.0), hi(12, 0.0), lo_g(12, -kInf), hi_g(12, kInf);
        for (int i = 0; i < 6; ++i) lo[std::size_t(i)] = -0.4, hi[std::size_t(i)] = 0.4;
        lo_g[2] = 0.94, hi_g[2] = 1.06;
        v.push_back(
            {"QUAD", "reach", Verdict::verified, 33.415,
             make_verify(aux_quad(), "quad", 0.1, box_poly(lo, hi), none, 5.0,
                         loop_options(0.1, 2, 1, 20, 0, 0, PolyMethod::regression),
                         TimedSpecification{Specification::safe_set(box_polytope(lo_g, hi_g)),
                                            std::make_pair(5.0, 5.0)},
                         {.depth = 4}),
             "order-1 enclosure and splitting (CORA: order 2 in one layer, no splitting)"});
    }
    return v;
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main(int argc, char **argv) { return cora::bench::run_instances(argc, argv, aux_instances); }

// ---------------------------------------  END OF CODE  ---------------------------------------- //
