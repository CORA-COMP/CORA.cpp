// benchmark_neur_net_contr_reach_double_pendulum_more_robust - Double Pendulum, more robust
//
// Double pendulum, more robust controller: falsified by a simulation. The network is
// examples/benchmarks/data/networks/<name>.nn.json and .nn.bin that
// matlab/benchmarks/export_network_data.m writes from the script; the plant is written out as the
// model file writes it. The line of the scripts, 'benchmark,instance,verdict,time', goes to stdout
// (the time of verify, simulations included), the expected verdict and time of the ARCH-COMP 2026
// report (CORA, Table 1) go to stderr. The exit code counts the verdicts that differ from the
// report's.
//
// Syntax:   benchmark_neur_net_contr_reach_double_pendulum_more_robust [instance ...]
// Outputs:  one line 'benchmark,instance,verdict,time' per instance on stdout
// See also: benchmark_neur_net.h
//
// CORA: contDynamics/neurNetContrSys/example_neuralNet_reach_05_doublePendulum_moreRobust.m
// CORA authors: Niklas Kochdumper, Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#include "benchmark_neur_net.h"
#include "benchmark_neur_net_double_pendulum.h"
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
    v.push_back({"Double_Pendulum", "more-robust", Verdict::falsified, 5.589,
                 make_verify(cora::bench::double_pendulum_plant(), "double_pendulum_more_robust",
                             0.02, box_poly({1, 1, 1, 1}, {1.3, 1.3, 1.3, 1.3}), none, 0.4,
                             loop_options(0.02, 2, 1, 10),
                             {Specification::safe_set(
                                 box_polytope({-1.5, -1.5, -1.5, -1.5}, {1.5, 1.5, 1.5, 1.5}))})});
    return v;
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main(int argc, char **argv) { return cora::bench::run_instances(argc, argv, aux_instances); }

// ---------------------------------------  END OF CODE  ---------------------------------------- //
