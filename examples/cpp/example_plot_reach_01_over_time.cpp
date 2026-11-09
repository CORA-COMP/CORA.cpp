// example_plot_reach_01_over_time - a reachable set in the plane and over time
//
// The reachable set of a damped oscillator in the phase plane (one region, the union of the
// sets of all steps), one state over time, simulations, and an unsafe set.
//
// Syntax:   build/examples/cpp/example_plot_reach_01_over_time
// Outputs:  example_plot_reach_01_over_time_plane.svg and _x1.svg
// See also: example_plot_reach_01_over_time of the Python examples, example_plot_tikz_01_group
//
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/cont_set/polytope/polytope.h"
#include "cora/global/plot/plot.h"
#include "cora/global/rng.h"

#include <iostream>


using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    // Parameters ------------------------------------------------------------------------------

    const double t_final = 6, time_step = 0.1;
    const Zonotope X0(Tensor({1, 0}), Tensor({{0.1, 0}, {0, 0.1}}));
    const LinearSys sys(Tensor({{-0.1, 1}, {-1, -0.1}}));

    // Reachability Analysis -------------------------------------------------------------------

    const ReachSet R = sys.reach({.initial_set = X0, .t_final = t_final},
                                 alg::Standard{.time_step = time_step, .taylor_terms = 6});
    cora::Rng rng(0);
    const std::vector<Tensor> x = sys.simulate_random({.initial_set = X0, .t_final = t_final},
                                                      alg::Exact{.time_step = time_step}, 10, rng);

    // Visualization ---------------------------------------------------------------------------

    // The phase plane: the unsafe set, the reachable set, the initial set and the simulations.
    use_cora_colors("CORA:contDynamics");
    Figure plane;
    plot(plane, Specification::unsafe_set(Polytope(Tensor({{-1.0, 0.0}}), Tensor({{0.5}}))), {0, 1},
         {.label = "unsafe set"});
    plot(plane, R, {0, 1}, {.label = "reachable set"});
    plot(plane, X0, {0, 1}, {.label = "initial set"});
    plot(plane, x, {0, 1}, {.label = "simulations"});
    plane.save("example_plot_reach_01_over_time_plane.svg");

    // The first state over time: boxes over the time intervals, and the simulations.
    Figure over_time;
    plot_over_time(over_time, R, 0, {.label = "reachable set"});
    plot_over_time(over_time, x, 0, time_step, {.label = "simulations"});
    over_time.save("example_plot_reach_01_over_time_x1.svg");
    std::cout << "steps: " << R.time_interval.size() << "\n";

    // example completed
    return 0;
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
