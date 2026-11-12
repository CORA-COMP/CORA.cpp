// benchmark_neur_net_contr_reach_nav - NAV (ARCH-COMP 2026)
//
// CORA: examples/contDynamics/neurNetContrSys/example_neuralNet_reach_11_NAV.m
//
// Navigation, standard (point-trained network) and robust (set-trained): the goal at t = 6, the
// obstacle all the time. The network is examples/benchmarks/data/networks/<name>.nn.json and
// .nn.bin that matlab/benchmarks/export_network_data.m writes from the script; the plant is written
// out as the model file writes it. The line of the scripts, 'benchmark,instance,verdict,time', goes
// to stdout (the time of verify, simulations included), the expected verdict and time of the
// ARCH-COMP 2026 report (CORA, Table 1) go to stderr. The exit code counts the verdicts that differ
// from the report's.
//
// Syntax:   build/core/benchmarks/benchmark_neur_net_contr_reach_nav [NAV-robust ...]
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
using cora::bench::kInf;
using cora::bench::loop_options;
using cora::bench::make_verify;

using Vars = std::vector<Expr>;

// The plant ------------------------------------------------------------------------------------

/// f(x, u) = (x3 cos x4, x3 sin x4, u1, u2) of the navigation task.
NonlinearSys aux_nav() {
    return {[](const Vars &x, const Vars &u) {
                return Vars{x[2] * cos(x[3]), x[2] * sin(x[3]), u[0] + 0.0 * x[0],
                            u[1] + 0.0 * x[0]};
            },
            4, 2};
}

// The instances ----------------------------------------------------------------------------------

std::vector<AinncsInstance> aux_instances() {
    const std::optional<Zonotope> none = std::nullopt;
    std::vector<AinncsInstance> v;
    // NAV: the goal at t = 6, the obstacle all the time
    for (const bool point : {true, false}) {
        const TimedSpecifications specs{
            {Specification::safe_set(
                 box_polytope({-0.5, -0.5, -kInf, -kInf}, {0.5, 0.5, kInf, kInf})),
             std::make_pair(6.0, 6.0)},
            {Specification::unsafe_set(box_polytope({1, 1, -kInf, -kInf}, {2, 2, kInf, kInf}))}};
        v.push_back(
            {"NAV", point ? "standard" : "robust", Verdict::verified, point ? 411.641 : 2.271,
             make_verify(aux_nav(), point ? "nav_point" : "nav_set", 0.2,
                         box_zonotope({2.9, 2.9, 0, 0}, {3.1, 3.1, 0, 0}), none, 6.0,
                         loop_options(point ? 0.05 : 0.1, 2, 1, 10, 0, 0, PolyMethod::regression),
                         specs, {.depth = 9})});
    }
    return v;
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main(int argc, char **argv) { return cora::bench::run_instances(argc, argv, aux_instances); }

// ---------------------------------------  END OF CODE  ---------------------------------------- //
