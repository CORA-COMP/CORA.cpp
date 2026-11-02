// example_linear_reach_01_5dim - reachability of a five-dimensional linear system
//
// The system of CORA's example of the same name, without its uncertain inputs (not implemented
// yet): the reachable set, and simulations that must stay inside it.
//
// Syntax:   build/examples/cpp/example_linear_reach_01_5dim
// Outputs:  the run time, the box around the last enclosure, and the figure as an SVG file
// See also: example_linear_reach_02_algorithms, the Python example of the same name (which plots)

#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/global/rng.h"
#include "cora/global/plot/plot.h"

#include <chrono>
#include <iostream>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

using namespace cora;

int main() {
    // Parameters ------------------------------------------------------------------------------

    const double t_final = 5;
    const Zonotope R0(Tensor({1, 1, 1, 1, 1}), Tensor({{0.1, 0, 0, 0, 0},
                                                       {0, 0.1, 0, 0, 0},
                                                       {0, 0, 0.1, 0, 0},
                                                       {0, 0, 0, 0.1, 0},
                                                       {0, 0, 0, 0, 0.1}}));

    // Reachability Settings -------------------------------------------------------------------

    const double time_step = 0.02;
    const int taylor_terms = 4;

    // System Dynamics -------------------------------------------------------------------------

    const LinearSys five_dim_sys(Tensor({{-1, -4, 0, 0, 0},
                                       {4, -1, 0, 0, 0},
                                       {0, 0, -3, 1, 0},
                                       {0, 0, -1, -3, 0},
                                       {0, 0, 0, 0, -2}}));

    // Reachability Analysis -------------------------------------------------------------------

    const auto timer_val = std::chrono::steady_clock::now();
    const ReachSet R = five_dim_sys.reach({.initial_set = R0, .t_final = t_final}, alg::Standard{.time_step = time_step, .taylor_terms = taylor_terms});
    const std::chrono::duration<double> computation_time = std::chrono::steady_clock::now() - timer_val;

    std::cout << "computation time of reachable set: " << computation_time.count() << " s\n";

    // Simulation ------------------------------------------------------------------------------

    cora::Rng rng(0);
    const std::vector<Tensor> traj = five_dim_sys.simulate_random({.initial_set = R0, .t_final = t_final}, alg::Exact{.time_step = time_step}, 25, rng);

    // Evaluation ------------------------------------------------------------------------------

    const Interval box = R.time_interval.back().interval();
    std::cout << "steps: " << R.time_interval.size() << ", simulated time points: " << traj.size() << "\n"
              << "the last enclosure lies in the box\n  lower " << box.inf() << "\n  upper "
              << box.sup() << "\n";

    // Visualization ---------------------------------------------------------------------------

    // The reachable set, the initial set and the simulations in the first two dimensions.
    use_cora_colors("CORA:contDynamics");
    plot(R, {0, 1}, {.label = "reachable set"});
    plot(R0, {0, 1}, {.label = "initial set"});
    plot(traj, {0, 1}, {.label = "simulations"});
    figure().title = "five-dimensional system";
    figure().save("example_linear_reach_01_5dim.svg");
    std::cout << "figure written to example_linear_reach_01_5dim.svg\n";

    // example completed
    return 0;
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
