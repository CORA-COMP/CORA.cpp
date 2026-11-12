// benchmark_neur_net_contr_reach_tora - TORA, remain (ARCH-COMP 2026)
//
// CORA: examples/contDynamics/neurNetContrSys/example_neuralNet_reach_03_TORA.m
//
// TORA with the ReLU controller: the state stays in [-2, 2]^4 for 20 s. The network is
// examples/benchmarks/data/networks/<name>.nn.json and .nn.bin that
// matlab/benchmarks/export_network_data.m writes from the script; the plant is written out as the
// model file writes it. The line of the scripts, 'benchmark,instance,verdict,time', goes to stdout
// (the time of verify, simulations included), the expected verdict and time of the ARCH-COMP 2026
// report (CORA, Table 1) go to stderr. The exit code counts the verdicts that differ from the
// report's.
//
// Syntax:   build/core/benchmarks/benchmark_neur_net_contr_reach_tora [TORA-remain ...]
// Outputs:  one line 'benchmark,instance,verdict,time' per instance on stdout
// See also: benchmark_neur_net.h
//
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#include "benchmark_neur_net.h"
#include "benchmark_neur_net_tora.h"
#include "benchmark_run.h"
#include "cora/cora.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using namespace cora;
using cora::bench::AinncsInstance;
using cora::bench::box_poly;
using cora::bench::box_polytope;
using cora::bench::loop_options;
using cora::bench::make_verify;

// The instances ----------------------------------------------------------------------------------

std::vector<AinncsInstance> aux_instances() {
    const std::optional<Zonotope> none = std::nullopt;
    std::vector<AinncsInstance> v;
    // TORA, remain
    v.push_back(
        {"TORA", "remain", Verdict::verified, 8.998,
         make_verify(cora::bench::tora_plant(1.0, -10.0), "tora_relu", 1.0,
                     box_poly({0.6, -0.7, -0.4, 0.5}, {0.7, -0.6, -0.3, 0.6}), none, 20.0,
                     loop_options(0.1, 3, 4, 200, 10, 20),
                     {Specification::safe_set(box_polytope({-2, -2, -2, -2}, {2, 2, 2, 2}))}),
         "time_step 0.1 (script 0.25)"});
    return v;
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main(int argc, char **argv) { return cora::bench::run_instances(argc, argv, aux_instances); }

// ---------------------------------------  END OF CODE  ---------------------------------------- //
