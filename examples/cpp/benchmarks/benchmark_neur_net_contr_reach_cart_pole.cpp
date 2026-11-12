// benchmark_neur_net_contr_reach_cart_pole - CartPole
//
// Cart pole: x1, x3, x4 in [-0.001, 0.001] for t in [8, 10], from the script's initial set and from
// the report's (0.4 times the radius of the script's). The network is
// examples/benchmarks/data/networks/<name>.nn.json and .nn.bin that
// matlab/benchmarks/export_network_data.m writes from the script; the plant is written out as the
// model file writes it. The line of the scripts, 'benchmark,instance,verdict,time', goes to stdout
// (the time of verify, simulations included), the expected verdict and time of the ARCH-COMP 2026
// report (CORA, Table 1) go to stderr. The exit code counts the verdicts that differ from the
// report's.
//
// Syntax:   benchmark_neur_net_contr_reach_cart_pole [instance ...]
// Outputs:  one line 'benchmark,instance,verdict,time' per instance on stdout
// See also: benchmark_neur_net.h
//
// CORA: contDynamics/neurNetContrSys/example_neuralNet_reach_12_cartPole.m
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
using cora::bench::kInf;
using cora::bench::loop_options;
using cora::bench::make_verify;

using Vars = std::vector<Expr>;

// The plant ------------------------------------------------------------------------------------

/// The cart pole: x1' = x2, x2' = 2 u, x3' = x4, x4' = (0.08 0.41 (9.8 sin x3 - 2 u cos x3) -
/// 0.0021 x4) / 0.0105.
NonlinearSys aux_cart_pole() {
    return {[](const Vars &x, const Vars &u) {
                return Vars{
                    x[1] + 0.0 * x[0], 2.0 * u[0], x[3] + 0.0 * x[0],
                    (0.08 * 0.41 * (9.8 * sin(x[2]) - 2.0 * u[0] * cos(x[2])) - 0.0021 * x[3]) /
                        0.0105};
            },
            4, 1};
}

// The instances ----------------------------------------------------------------------------------

std::vector<AinncsInstance> aux_instances() {
    const std::optional<Zonotope> none = std::nullopt;
    std::vector<AinncsInstance> v;
    // CartPole: x1, x3, x4 in [-0.001, 0.001] for t in [8, 10]
    for (const bool report : {false, true}) {
        // the script's set has 0.4 times the radius of the report's
        const double r = report ? 1.0 : 0.4;
        const std::vector<double> lo{-0.1 * r, -0.05 * r, -0.1 * r, -0.05 * r},
            hi{0.1 * r, 0.05 * r, 0.1 * r, 0.05 * r};
        v.push_back(
            {"CartPole", report ? "reach-report" : "reach", Verdict::verified, 227.216,
             make_verify(aux_cart_pole(), "cartpole", 0.02, box_poly(lo, hi), none, 10.0,
                         loop_options(0.02, 2, 1, 50, 0, 0, PolyMethod::regression),
                         TimedSpecification{
                             Specification::safe_set(box_polytope({-0.001, -kInf, -0.001, -0.001},
                                                                  {0.001, kInf, 0.001, 0.001})),
                             std::make_pair(8.0, 10.0)},
                         {.depth = report ? 16 : 10}),
             report
                 ? "the report's initial set (no tool ran it), split to depth 16 (CORA's time is "
                   "for the smaller set)"
                 : "the smaller initial set of the script (0.4 times the radius of the report's)"});
    }
    return v;
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main(int argc, char **argv) { return cora::bench::run_instances(argc, argv, aux_instances); }

// ---------------------------------------  END OF CODE  ---------------------------------------- //
