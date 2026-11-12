// benchmark_neur_net_contr_reach_tora_heterogeneous_sigmoid - TORA heterogeneous sigmoid
//
// TORA with the sigmoid controller. The network is examples/benchmarks/data/networks/<name>.nn.json
// and .nn.bin that matlab/benchmarks/export_network_data.m writes from the script; the plant is
// written out as the model file writes it. The line of the scripts,
// 'benchmark,instance,verdict,time', goes to stdout (the time of verify, simulations included), the
// expected verdict and time of the ARCH-COMP 2026 report (CORA, Table 1) go to stderr. The exit
// code counts the verdicts that differ from the report's.
//
// Syntax:   benchmark_neur_net_contr_reach_tora_heterogeneous_sigmoid [instance ...]
// Outputs:  one line 'benchmark,instance,verdict,time' per instance on stdout
// See also: benchmark_neur_net.h
//
// CORA: contDynamics/neurNetContrSys/example_neuralNet_reach_03_TORA_heterogeneousSigmoid.m
// CORA authors: Niklas Kochdumper, Tobias Ladner
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
using cora::bench::kInf;
using cora::bench::loop_options;
using cora::bench::make_verify;

// The instances ----------------------------------------------------------------------------------

std::vector<AinncsInstance> aux_instances() {
    const std::optional<Zonotope> none = std::nullopt;
    std::vector<AinncsInstance> v;
    // the goal set at the end
    v.push_back({"TORA", "reach-sigmoid", Verdict::verified, 4.730,
                 make_verify(cora::bench::tora_plant(11.0, 0.0), "tora_sigmoid", 0.5,
                             box_poly({-0.77, -0.45, 0.51, -0.3}, {-0.75, -0.43, 0.54, -0.28}),
                             none, 5.0, loop_options(0.025, 3, 4, 20, 10, 50),
                             {Specification::safe_set(box_polytope({-0.1, -0.9, -kInf, -kInf},
                                                                   {0.2, -0.6, kInf, kInf})),
                              std::make_pair(5.0, 5.0)}),
                 "time_step 0.025 (script 0.1)"});
    return v;
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main(int argc, char **argv) { return cora::bench::run_instances(argc, argv, aux_instances); }

// ---------------------------------------  END OF CODE  ---------------------------------------- //
