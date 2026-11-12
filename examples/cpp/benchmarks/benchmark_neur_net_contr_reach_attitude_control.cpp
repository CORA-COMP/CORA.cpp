// benchmark_neur_net_contr_reach_attitude_control - Attitude Control
//
// Attitude control: the unsafe set at t = 3 must not be reached. The network is
// examples/benchmarks/data/networks/<name>.nn.json and .nn.bin that
// matlab/benchmarks/export_network_data.m writes from the script; the plant is written out as the
// model file writes it. The line of the scripts, 'benchmark,instance,verdict,time', goes to stdout
// (the time of verify, simulations included), the expected verdict and time of the ARCH-COMP 2026
// report (CORA, Table 1) go to stderr. The exit code counts the verdicts that differ from the
// report's.
//
// Syntax:   benchmark_neur_net_contr_reach_attitude_control [instance ...]
// Outputs:  one line 'benchmark,instance,verdict,time' per instance on stdout
// See also: benchmark_neur_net.h
//
// CORA: contDynamics/neurNetContrSys/example_neuralNet_reach_08_attitudeControl.m
// CORA authors: Tobias Ladner
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
using cora::bench::loop_options;
using cora::bench::make_verify;

using Vars = std::vector<Expr>;

// The plant ------------------------------------------------------------------------------------

/// examples/contDynamics/neurNetContrSys/dynamics/dynamics_attitudeControl.m
NonlinearSys aux_attitude() {
    return {[](const Vars &x, const Vars &u) {
                const Expr &w1 = x[0], &w2 = x[1], &w3 = x[2], &p1 = x[3], &p2 = x[4], &p3 = x[5];
                const Expr p = pow(p1, 2) + pow(p2, 2) + pow(p3, 2);
                return Vars{0.25 * (u[0] + w2 * w3),
                            0.5 * (u[1] - 3.0 * w1 * w3),
                            u[2] + 2.0 * w1 * w2,
                            0.5 * (w2 * (p - p3) + w3 * (p + p2) + w1 * (p + 1.0)),
                            0.5 * (w1 * (p + p3) + w3 * (p - p1) + w2 * (p + 1.0)),
                            0.5 * (w1 * (p - p2) + w2 * (p + p1) + w3 * (p + 1.0))};
            },
            6, 3};
}

// The instances ----------------------------------------------------------------------------------

std::vector<AinncsInstance> aux_instances() {
    const std::optional<Zonotope> none = std::nullopt;
    std::vector<AinncsInstance> v;
    // Attitude Control: the unsafe set at t = 3 must not be reached
    v.push_back(
        {"Attitude_Control", "avoid", Verdict::verified, 5.226,
         make_verify(aux_attitude(), "attitude_control", 0.1,
                     box_poly({-0.45, -0.55, 0.65, -0.75, 0.85, -0.65},
                              {-0.44, -0.54, 0.66, -0.74, 0.86, -0.64}),
                     none, 3.0, loop_options(0.1, 2, 1, 10),
                     {Specification::unsafe_set(box_polytope({-0.2, -0.5, 0, -0.7, 0.7, -0.4},
                                                             {0, -0.4, 0.2, -0.6, 0.8, -0.2})),
                      std::make_pair(3.0, 3.0)})});
    return v;
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main(int argc, char **argv) { return cora::bench::run_instances(argc, argv, aux_instances); }

// ---------------------------------------  END OF CODE  ---------------------------------------- //
