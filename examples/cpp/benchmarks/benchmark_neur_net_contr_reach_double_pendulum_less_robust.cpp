// benchmark_neur_net_contr_reach_double_pendulum_less_robust - Double Pendulum, less robust
//
// Double pendulum, less robust controller: the script is UNKNOWN, the report's tools split the
// initial set. The network is examples/benchmarks/data/networks/<name>.nn.json and .nn.bin that
// matlab/benchmarks/export_network_data.m writes from the script; the plant is written out as the
// model file writes it. The line of the scripts, 'benchmark,instance,verdict,time', goes to stdout
// (the time of verify, simulations included), the expected verdict and time of the ARCH-COMP 2026
// report (CORA, Table 1) go to stderr. The exit code counts the verdicts that differ from the
// report's.
//
// Syntax:   benchmark_neur_net_contr_reach_double_pendulum_less_robust [instance ...]
// Outputs:  one line 'benchmark,instance,verdict,time' per instance on stdout
// See also: benchmark_neur_net.h
//
// CORA: contDynamics/neurNetContrSys/example_neuralNet_reach_05_doublePendulum_lessRobust.m
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
using cora::bench::box_polytope;
using cora::bench::box_zonotope;
using cora::bench::loop_options;
using cora::bench::make_verify;

// The instances ----------------------------------------------------------------------------------

std::vector<AinncsInstance> aux_instances() {
    const std::optional<Zonotope> none = std::nullopt;
    std::vector<AinncsInstance> v;
    // Double Pendulum less-robust: UNKNOWN by the script, verified by the report's splitting tools
    for (const bool split : {false, true}) {
        v.push_back(
            {"Double_Pendulum", split ? "less-robust" : "less-robust-script",
             split ? Verdict::verified : Verdict::unknown, split ? 67.334 : 5.604,
             make_verify(
                 cora::bench::double_pendulum_plant(), "double_pendulum_less_robust", 0.05,
                 box_zonotope({1, 1, 1, 1}, {1.3, 1.3, 1.3, 1.3}), none, 1.0,
                 loop_options(cora::bench::error_bound() == alg::ErrorBound::loop ? 0.01 : 0.005, 2,
                              4, 20),
                 {Specification::safe_set(box_polytope({-1.7, -1.7, -1.7, -1.7}, {2, 2, 2, 2}))},
                 {.depth = split ? 14 : 0}),
             split ? "splitting of the initial set (CROWN-Reach: 5 x 5 x 3 x 3 uniform pieces); "
                     "the expected time is CROWN-Reach's, CORA's script is UNKNOWN in 5.604 s; "
                     "time_step 0.01 as the script (0.005 under the region step, whose check fails "
                     "at 0.01)"
                   : "the script, without splitting; time_step 0.01 as the script (0.005 under the "
                     "region step)"});
    }
    return v;
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main(int argc, char **argv) { return cora::bench::run_instances(argc, argv, aux_instances); }

// ---------------------------------------  END OF CODE  ---------------------------------------- //
