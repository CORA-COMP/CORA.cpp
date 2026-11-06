// example_linear_reach_08_inputs - reachability of a linear system with an input and an output
//
// A mass-spring-damper x' = A x + B u, y = C x: a force u(t) in U acts on the mass, the position
// is measured. The reachable states, the output sets, and simulations with constant forces.
//
// Syntax:   build/examples/cpp/example_linear_reach_08_inputs
// Outputs:  the run time, the range of the output, and the figure as an SVG file
// See also: example_linear_reach_01_5dim, example_linear_simulate_01_random
//
// Authors:       Claude Sonnet 5.5
// Written:       30-September-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/linear_sys/linear_sys.h"
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

    const double t_final = 5;
    const Zonotope R0(Tensor({0, 0}), Tensor({{0.2, 0}, {0, 0.2}})); // position, velocity
    const Zonotope U(Tensor({0.5}), Tensor({{0.25}}));               // force in [0.25, 0.75]

    // Reachability Settings -------------------------------------------------------------------

    const double time_step = 0.05;
    const int taylor_terms = 6;
    const int zonotope_order = 20;

    // System Dynamics -------------------------------------------------------------------------

    const Tensor A({{0, 1}, {-4, -0.4}});
    const Tensor B({{0}, {1}});
    const Tensor C({{1, 0}}); // the position
    const LinearSys sys(A, B, C);

    // Reachability Analysis -------------------------------------------------------------------

    const auto timer_val = std::chrono::steady_clock::now();
    const ReachSet R = sys.reach({.initial_set = R0, .input_set = U, .t_final = t_final},
                                 alg::Standard{.time_step = time_step,
                                               .taylor_terms = taylor_terms,
                                               .zonotope_order = zonotope_order});
    const ReachSet Y = sys.output_set(R);
    const std::chrono::duration<double> computation_time =
        std::chrono::steady_clock::now() - timer_val;

    std::cout << "computation time of reachable set: " << computation_time.count() << " s\n";

    // Simulation ------------------------------------------------------------------------------

    cora::Rng rng(0);
    const std::vector<Tensor> traj =
        sys.simulate_random({.initial_set = R0, .input_set = U, .t_final = t_final},
                            alg::Exact{.time_step = time_step}, 25, rng);

    // Evaluation ------------------------------------------------------------------------------

    // The position over all steps, from the output sets.
    double lower = 1e9, upper = -1e9;
    for (const Zonotope &y : Y.time_interval) {
        const Interval box = y.interval();
        lower = std::min(lower, box.inf().data()[0]);
        upper = std::max(upper, box.sup().data()[0]);
    }
    std::cout << "the output y stays in [" << lower << ", " << upper << "]\n";

    // Visualization ---------------------------------------------------------------------------

    use_cora_colors("CORA:contDynamics");
    plot(R, {0, 1}, {.label = "reachable set"});
    plot(R0, {0, 1}, {.label = "initial set"});
    plot(traj, {0, 1}, {.label = "simulations"});
    figure().title = "mass-spring-damper with a force";
    figure().save("example_linear_reach_08_inputs.svg");
    std::cout << "figure written to example_linear_reach_08_inputs.svg\n";

    // example completed
    return 0;
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
