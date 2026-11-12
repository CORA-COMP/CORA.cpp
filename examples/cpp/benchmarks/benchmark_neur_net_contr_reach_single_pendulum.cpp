// benchmark_neur_net_contr_reach_single_pendulum - Single Pendulum (ARCH-COMP 2026)
//
// CORA: examples/contDynamics/neurNetContrSys/example_neuralNet_reach_04_singlePendulum.m
//
// Single pendulum: the angle stays in [0, 1] from 0.5 to 1 s. The network is
// examples/benchmarks/data/networks/<name>.nn.json and .nn.bin that
// matlab/benchmarks/export_network_data.m writes from the script; the plant is written out as the
// model file writes it. The line of the scripts, 'benchmark,instance,verdict,time', goes to stdout
// (the time of verify, simulations included), the expected verdict and time of the ARCH-COMP 2026
// report (CORA, Table 1) go to stderr. The exit code counts the verdicts that differ from the
// report's.
//
// Syntax:   build/core/benchmarks/benchmark_neur_net_contr_reach_single_pendulum
// [Single_Pendulum-reach ...] Outputs:  one line 'benchmark,instance,verdict,time' per instance on
// stdout See also: benchmark_neur_net.h
//
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#include "benchmark_neur_net.h"
#include "benchmark_run.h"
#include "cora/cora.h"

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

/// example_neuralNet_reach_04_singlePendulum.m: m = 0.5, L = 0.5, c = 0, g = 1.
NonlinearSys aux_single_pendulum() {
    constexpr double m = 0.5, L = 0.5, c = 0, g = 1;
    return {[=](const Vars &x, const Vars &u) {
                return Vars{x[1], g / L * sin(x[0]) + 1 / (m * L * L) * (u[0] - c * x[1])};
            },
            2, 1};
}

// The instances ----------------------------------------------------------------------------------

std::vector<AinncsInstance> aux_instances() {
    const std::optional<Zonotope> none = std::nullopt;
    std::vector<AinncsInstance> v;
    // Single Pendulum: the angle in [0, 1] from 0.5 to 1
    v.push_back(
        {"Single_Pendulum", "reach", Verdict::verified, 4.083,
         make_verify(aux_single_pendulum(), "single_pendulum", 0.05,
                     box_poly({1.0, 0.0}, {1.175, 0.2}), none, 1.0, loop_options(0.02, 2, 1, 10),
                     {Specification::safe_set(box_polytope({0.0, -kInf}, {1.0, kInf})),
                      std::make_pair(0.5, 1.0)})});
    return v;
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main(int argc, char **argv) { return cora::bench::run_instances(argc, argv, aux_instances); }

// ---------------------------------------  END OF CODE  ---------------------------------------- //
