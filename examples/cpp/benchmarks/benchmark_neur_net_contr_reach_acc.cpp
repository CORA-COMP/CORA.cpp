// benchmark_neur_net_contr_reach_acc - ACC (ARCH-COMP 2026)
//
// CORA: examples/contDynamics/neurNetContrSys/example_neuralNet_reach_02_ACC.m
//
// Adaptive cruise control, the safe distance D_rel >= 10 + 1.4 v_ego. The network is
// examples/benchmarks/data/networks/<name>.nn.json and .nn.bin that
// matlab/benchmarks/export_network_data.m writes from the script; the plant is written out as the
// model file writes it. The line of the scripts, 'benchmark,instance,verdict,time', goes to stdout
// (the time of verify, simulations included), the expected verdict and time of the ARCH-COMP 2026
// report (CORA, Table 1) go to stderr. The exit code counts the verdicts that differ from the
// report's.
//
// Syntax:   build/core/benchmarks/benchmark_neur_net_contr_reach_acc [ACC-safe-distance ...]
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
using cora::bench::box_poly;
using cora::bench::loop_options;
using cora::bench::make_verify;

using Vars = std::vector<Expr>;

// The plant ------------------------------------------------------------------------------------

/// example_neuralNet_reach_02_ACC.m: both cars with the air drag u_f x^2 on the velocity.
NonlinearSys aux_acc() {
    constexpr double u_f = 0.0001, a_lead = -2;
    return {[=](const Vars &x, const Vars &u) {
                return Vars{x[1], x[2], -2.0 * x[2] + 2.0 * a_lead - u_f * pow(x[1], 2),
                            x[4], x[5], -2.0 * x[5] + 2.0 * u[0] - u_f * pow(x[3], 2)};
            },
            6, 1};
}

// The instances ----------------------------------------------------------------------------------

std::vector<AinncsInstance> aux_instances() {
    const std::optional<Zonotope> none = std::nullopt;
    std::vector<AinncsInstance> v;
    // ACC: the safe distance D_rel >= 10 + 1.4 v_ego is one halfspace
    v.push_back(
        {"ACC", "safe-distance", Verdict::verified, 3.176,
         make_verify(
             aux_acc(), "acc", 0.1, box_poly({90, 32, 0, 10, 30, 0}, {110, 32.2, 0, 11, 30.2, 0}),
             none, 5.0, loop_options(0.1, 2, 1, 10),
             {Specification::safe_set(Polytope(Tensor::from_data({-1, 0, 0, 1, 1.4, 0}, {1, 6}),
                                               Tensor::from_data({-10.0}, {1, 1})))})});
    return v;
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main(int argc, char **argv) { return cora::bench::run_instances(argc, argv, aux_instances); }

// ---------------------------------------  END OF CODE  ---------------------------------------- //
