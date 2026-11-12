// benchmark_neur_net_contr_reach_unicycle - Unicycle (ARCH-COMP 2026)
//
// CORA: examples/contDynamics/neurNetContrSys/example_neuralNet_reach_01_unicycle.m
//
// Unicycle, once as the script (its default disturbance zonotope(0)) and once with the disturbance
// of the report. The network is examples/benchmarks/data/networks/<name>.nn.json and .nn.bin that
// matlab/benchmarks/export_network_data.m writes from the script; the plant is written out as the
// model file writes it. The line of the scripts, 'benchmark,instance,verdict,time', goes to stdout
// (the time of verify, simulations included), the expected verdict and time of the ARCH-COMP 2026
// report (CORA, Table 1) go to stderr. The exit code counts the verdicts that differ from the
// report's.
//
// Syntax:   build/core/benchmarks/benchmark_neur_net_contr_reach_unicycle [Unicycle-reach ...]
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

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using namespace cora;
using cora::bench::AinncsInstance;
using cora::bench::box_polytope;
using cora::bench::box_zonotope;
using cora::bench::loop_options;
using cora::bench::make_verify;

using Vars = std::vector<Expr>;

// The plant ------------------------------------------------------------------------------------

/// example_neuralNet_reach_01_unicycle.m
NonlinearSys aux_unicycle() {
    return {[](const Vars &x, const Vars &u) {
                return Vars{x[3] * cos(x[2]), x[3] * sin(x[2]), u[1] - 20.0, u[0] - 20.0 + u[2]};
            },
            4, 3};
}

// The instances ----------------------------------------------------------------------------------

std::vector<AinncsInstance> aux_instances() {
    std::vector<AinncsInstance> v;
    // Unicycle: the disturbance input is the script's default zonotope(0)
    const Zonotope no_disturbance(Tensor::from_data({0.0}, {1, 1}), Tensor::zeros({1, 0}));
    v.push_back({"Unicycle", "reach", Verdict::verified, 8.015,
                 make_verify(aux_unicycle(), "unicycle", 0.2,
                             box_zonotope({9.5, -4.5, 2.1, 1.5}, {9.55, -4.45, 2.11, 1.51}),
                             no_disturbance, 10.0, loop_options(0.1, 2, 1, 10),
                             {Specification::safe_set(
                                  box_polytope({-0.6, -0.2, -0.06, -0.3}, {0.6, 0.2, 0.06, 0.3})),
                              std::make_pair(10.0, 10.0)})});
    // the same with the report's disturbance w in 1e-4 [-1, 1] on the speed, which the script drops
    const Zonotope disturbance(Tensor::from_data({0.0}, {1, 1}), Tensor::from_data({1e-4}, {1, 1}));
    v.push_back({"Unicycle", "reach-disturbance", Verdict::verified, 8.015,
                 make_verify(aux_unicycle(), "unicycle", 0.2,
                             box_zonotope({9.5, -4.5, 2.1, 1.5}, {9.55, -4.45, 2.11, 1.51}),
                             disturbance, 10.0, loop_options(0.1, 2, 1, 10),
                             {Specification::safe_set(
                                  box_polytope({-0.6, -0.2, -0.06, -0.3}, {0.6, 0.2, 0.06, 0.3})),
                              std::make_pair(10.0, 10.0)}),
                 "the disturbance w of the report (the script sets w = 0)"});
    return v;
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main(int argc, char **argv) { return cora::bench::run_instances(argc, argv, aux_instances); }

// ---------------------------------------  END OF CODE  ---------------------------------------- //
