// example_nonlinear_reach_01_vanDerPol - reachability of the van der Pol oscillator
//
// The dynamics is written once on symbolic states; the system differentiates it, and reach
// linearizes it step by step (CORA's algorithm "lin"). The reachable set follows the limit cycle
// for one period; simulations from the initial set must stay inside it.
//
// Syntax:   build/examples/cpp/example_nonlinear_reach_01_vanDerPol
// Outputs:  the run time, the widest enclosure, and the figure as an SVG file
// See also: example_linear_reach_01_5dim, the Python example of the same name (which plots)

#include "contDynamics/nonlinearSys/nonlinearSys.h"
#include "global/rng.h"
#include "global/plot/plot.h"

#include <algorithm>
#include <chrono>
#include <iostream>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

using namespace cora::ct;

int main() {
    // Parameters ------------------------------------------------------------------------------

    const double tFinal = 6.74;
    const Zonotope R0(Tensor({1.4, 2.3}), Tensor({{0.05, 0.0}, {0.0, 0.05}}));

    // Reachability Settings -------------------------------------------------------------------

    const double timeStep = 0.005;
    const int taylorTerms = 4;
    const int zonotopeOrder = 50;

    // System Dynamics -------------------------------------------------------------------------

    const double mu = 1;
    const NonlinearSys vdp(
        [mu](const std::vector<Expr> &x) {
            return std::vector<Expr>{x[1], mu * (1 - x[0] * x[0]) * x[1] - x[0]};
        },
        2);

    // Reachability Analysis -------------------------------------------------------------------

    const auto timerVal = std::chrono::steady_clock::now();
    const Reach R = vdp.reach(R0, timeStep, tFinal, taylorTerms, zonotopeOrder);
    const std::chrono::duration<double> tComp = std::chrono::steady_clock::now() - timerVal;

    std::cout << "computation time of reachable set: " << tComp.count() << " s\n";

    // Simulation ------------------------------------------------------------------------------

    cora::Rng rng(0);
    const std::vector<Tensor> traj = vdp.simulateRandom(R0, 10, timeStep, tFinal, rng);

    // Evaluation ------------------------------------------------------------------------------

    double widest = 0;
    for (const Zonotope &Z : R.timeInt) {
        const Interval box = Z.interval();
        const std::vector<double> lo = box.inf.data(), hi = box.sup.data();
        widest = std::max({widest, hi[0] - lo[0], hi[1] - lo[1]});
    }
    std::cout << "steps: " << R.timeInt.size() << ", widest enclosure: " << widest << "\n";

    // Visualization ---------------------------------------------------------------------------

    useCORAcolors("CORA:contDynamics");
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
