// example_nonlinear_reach_01_van_der_pol - reachability of the van der Pol oscillator
//
// The dynamics is written once on symbolic states; the system differentiates it, and reach
// linearizes it step by step (CORA's algorithm "lin"). The reachable set follows the limit cycle
// for one period; simulations from the initial set must stay inside it.
//
// Syntax:   build/examples/cpp/example_nonlinear_reach_01_vanDerPol
// Outputs:  the run time, the widest enclosure, and the figure as an SVG file
// See also: example_linear_reach_01_5dim, the Python example of the same name (which plots)
//
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/nonlinear_sys/nonlinear_sys.h"
#include "cora/global/plot/plot.h"
#include "cora/global/rng.h"

#include <algorithm>
#include <chrono>
#include <iostream>


using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    // Parameters ------------------------------------------------------------------------------

    const double t_final = 6.74;
    const Zonotope R0(Tensor({1.4, 2.3}), Tensor({{0.05, 0.0}, {0.0, 0.05}}));

    // Reachability Settings -------------------------------------------------------------------

    const double time_step = 0.005;
    const int taylor_terms = 4;
    const int zonotope_order = 50;

    // System Dynamics -------------------------------------------------------------------------

    const double mu = 1;
    const NonlinearSys vdp(
        [mu](const std::vector<Expr> &x) {
            return std::vector<Expr>{x[1], mu * (1 - x[0] * x[0]) * x[1] - x[0]};
        },
        2);

    // Reachability Analysis -------------------------------------------------------------------

    const auto timer_val = std::chrono::steady_clock::now();
    const ReachSet R = vdp.reach({.initial_set = R0, .t_final = t_final},
                                 alg::Linearization{.time_step = time_step,
                                                    .taylor_terms = taylor_terms,
                                                    .zonotope_order = zonotope_order});
    const std::chrono::duration<double> computation_time =
        std::chrono::steady_clock::now() - timer_val;

    std::cout << "computation time of reachable set: " << computation_time.count() << " s\n";

    // Simulation ------------------------------------------------------------------------------

    cora::Rng rng(0);
    const std::vector<Tensor> traj = vdp.simulate_random(
        {.initial_set = R0, .t_final = t_final}, alg::RungeKutta4{.time_step = time_step}, 10, rng);

    // Evaluation ------------------------------------------------------------------------------

    double widest = 0;
    for (const Zonotope &Z : R.time_interval) {
        const Interval box = Z.interval();
        const std::vector<double> lo = box.inf().data(), hi = box.sup().data();
        widest = std::max({widest, hi[0] - lo[0], hi[1] - lo[1]});
    }
    std::cout << "steps: " << R.time_interval.size() << ", widest enclosure: " << widest << "\n";

    // Visualization ---------------------------------------------------------------------------

    use_cora_colors("CORA:contDynamics");
    plot(R, {0, 1}, {.label = "reachable set"});
    plot(R0, {0, 1}, {.label = "initial set"});
    plot(traj, {0, 1}, {.label = "simulations"});
    figure().title = "van der Pol oscillator";
    figure().save("example_nonlinear_reach_01_vanDerPol.svg");
    std::cout << "figure written to example_nonlinear_reach_01_vanDerPol.svg\n";

    // example completed
    return 0;
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
