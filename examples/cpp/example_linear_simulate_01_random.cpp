// example_linear_simulate_01_random - simulations from random points of the initial set
//
// Random points of the initial set, run through the dynamics, must stay inside the reachable set.
// Extreme points (corners of the generator cube) are where a set that is too small shows first.
//
// Syntax:   build/examples/cpp/example_linear_simulate_01_random
// Outputs:  the size of the simulation, and whether it stays in the reachable set
// See also: example_linear_reach_01_5dim

#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/global/rng.h"

#include <algorithm>
#include <iostream>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

using namespace cora;

int main() {
    // Parameters ------------------------------------------------------------------------------

    const double t_final = 1.0;
    const Zonotope R0(Tensor({1.0, 0.0}), Tensor({{0.1, 0.0}, {0.0, 0.1}}));
    cora::Rng rng(1); // a seed makes the points repeatable

    // System Dynamics -------------------------------------------------------------------------

    const LinearSys sys(Tensor({{-0.1, 1.0}, {-1.0, -0.1}}));

    // Reachability Analysis -------------------------------------------------------------------

    const double time_step = 0.1;
    const ReachSet R = sys.reach({.initial_set = R0, .t_final = t_final},
                                 alg::Standard{.time_step = time_step, .taylor_terms = 8});

    // Simulation ------------------------------------------------------------------------------

    // traj[k] holds the 20 points at time k * time_step / 2; the columns are the trajectories.
    const std::vector<Tensor> traj = sys.simulate_random(
        {.initial_set = R0, .t_final = t_final}, alg::Exact{.time_step = time_step / 2}, 20, rng);

    // The corners of the generator cube as start points, and their trajectories.
    const Tensor corners = R0.rand_point(20, rng, "extreme");
    const std::vector<Tensor> traj_extreme = sys.simulate(
        {.initial_state = corners, .t_final = t_final}, alg::Exact{.time_step = time_step / 2});

    // Verification ----------------------------------------------------------------------------

    // A point is in the reachable set of its step if d'x is at most the support function there.
    const Tensor d({-1.0, 0.5});
    bool inside = true;
    for (std::size_t j = 0; j < traj_extreme.size(); ++j) {
        const std::size_t step = std::min(j / 2, R.time_interval.size() - 1); // two points per step
        const std::vector<double> along = d.transpose().matmul(traj_extreme[j]).data();
        inside &= *std::max_element(along.begin(), along.end()) <=
                  R.time_interval[step].support_function(d).data()[0] + 1e-9;
    }
    std::cout << "time points: " << traj.size() << ", trajectories: " << traj[0].shape()[1] << "\n"
              << "trajectories from the corners stay in the reachable set: "
              << (inside ? "yes" : "NO") << "\n";

    // example completed
    return 0;
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
