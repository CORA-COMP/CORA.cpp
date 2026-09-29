// example_linear_simulate_01_random - simulations from random points of the initial set
//
// Random points of the initial set, run through the dynamics, must stay inside the reachable set.
// Extreme points (corners of the generator cube) are where a set that is too small shows first.
//
// Syntax:   build/examples/cpp/example_linear_simulate_01_random
// Outputs:  the size of the simulation, and whether it stays in the reachable set
// See also: example_linear_reach_01_5dim

#include "contDynamics/linearSys/linearSys.h"
#include "global/rng.h"

#include <algorithm>
#include <iostream>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

using namespace cora::ct;

int main() {
    // Parameters ------------------------------------------------------------------------------

    const double tFinal = 1.0;
    const Zonotope R0(Tensor({1.0, 0.0}), Tensor({{0.1, 0.0}, {0.0, 0.1}}));
    cora::Rng rng(1); // a seed makes the points repeatable

    // System Dynamics -------------------------------------------------------------------------

    const LinearSys sys(Tensor({{-0.1, 1.0}, {-1.0, -0.1}}));

    // Reachability Analysis -------------------------------------------------------------------

    const double timeStep = 0.1;
    const Reach R = sys.reach(R0, timeStep, tFinal, 8);

    // Simulation ------------------------------------------------------------------------------

    // traj[k] holds the 20 points at time k * timeStep / 2; the columns are the trajectories.
    const std::vector<Tensor> traj = sys.simulateRandom(R0, 20, timeStep / 2, tFinal, rng);

    // The corners of the generator cube as start points, and their trajectories.
    const Tensor corners = R0.randPoint(20, rng, "extreme");
    const std::vector<Tensor> trajExtreme = sys.simulate(corners, timeStep / 2, tFinal);

    // Verification ----------------------------------------------------------------------------

    // A point is in the reachable set of its step if d'x is at most the support function there.
    const Tensor d({-1.0, 0.5});
    bool inside = true;
    for (std::size_t j = 0; j < trajExtreme.size(); ++j) {
        const std::size_t step = std::min(j / 2, R.timeInt.size() - 1); // two points per step
        const std::vector<double> along = d.transpose().matmul(trajExtreme[j]).data();
        inside &= *std::max_element(along.begin(), along.end()) <=
                  R.timeInt[step].supportFunc(d).data()[0] + 1e-9;
    }
    std::cout << "time points: " << traj.size() << ", trajectories: " << traj[0].shape()[1] << "\n"
              << "trajectories from the corners stay in the reachable set: "
              << (inside ? "yes" : "NO") << "\n";

    // example completed
    return 0;
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
