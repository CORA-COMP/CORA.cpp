// example_linear_reach_01_5dim - reachability of a five-dimensional linear system
//
// The system of CORA's example of the same name, without its uncertain inputs (not implemented
// yet): the reachable set, and simulations that must stay inside it.
//
// Syntax:   build/examples/cpp/example_linear_reach_01_5dim
// Outputs:  the run time, the box around the last enclosure, and the figure as an SVG file
// See also: example_linear_reach_02_algorithms, the Python example of the same name (which plots)

#include "contDynamics/linearSys/linearSys.h"
#include "global/rng.h"
#include "plot/plot.h"

#include <chrono>
#include <iostream>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

using namespace cora::ct;

int main() {
    // Parameters ------------------------------------------------------------------------------

    const double tFinal = 5;
    const Zonotope R0(Tensor({1, 1, 1, 1, 1}), Tensor({{0.1, 0, 0, 0, 0},
                                                       {0, 0.1, 0, 0, 0},
                                                       {0, 0, 0.1, 0, 0},
                                                       {0, 0, 0, 0.1, 0},
                                                       {0, 0, 0, 0, 0.1}}));

    // Reachability Settings -------------------------------------------------------------------

    const double timeStep = 0.02;
    const int taylorTerms = 4;

    // System Dynamics -------------------------------------------------------------------------

    const LinearSys fiveDimSys(Tensor({{-1, -4, 0, 0, 0},
                                       {4, -1, 0, 0, 0},
                                       {0, 0, -3, 1, 0},
                                       {0, 0, -1, -3, 0},
                                       {0, 0, 0, 0, -2}}));

    // Reachability Analysis -------------------------------------------------------------------

    const auto timerVal = std::chrono::steady_clock::now();
    const Reach R = fiveDimSys.reach(R0, timeStep, tFinal, taylorTerms);
    const std::chrono::duration<double> tComp = std::chrono::steady_clock::now() - timerVal;

    std::cout << "computation time of reachable set: " << tComp.count() << " s\n";

    // Simulation ------------------------------------------------------------------------------

    cora::Rng rng(0);
    const std::vector<Tensor> traj = fiveDimSys.simulateRandom(R0, 25, timeStep, tFinal, rng);

    // Evaluation ------------------------------------------------------------------------------

    const Interval box = R.timeInt.back().interval();
    std::cout << "steps: " << R.timeInt.size() << ", simulated time points: " << traj.size() << "\n"
              << "the last enclosure lies in the box\n  lower " << box.inf << "\n  upper "
              << box.sup << "\n";

    // Visualization ---------------------------------------------------------------------------

    // The reachable set, the initial set and the simulations in the first two dimensions.
    plot(R, {0, 1}, {.label = "reachable set"});
    plot(R0, {0, 1}, {.label = "initial set", .color = "CORA:simulations", .facecolor = "CORA:initialSet"});
    plot(traj, {0, 1}, {.label = "simulations"});
    figure().title = "five-dimensional system";
    figure().save("example_linear_reach_01_5dim.svg");
    std::cout << "figure written to example_linear_reach_01_5dim.svg\n";

    // example completed
    return 0;
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
