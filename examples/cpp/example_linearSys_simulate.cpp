// Simulations: random points of the initial set, run through the dynamics, and a check that
// they stay inside the reachable set. Extreme points (corners of the generator cube) are where
// a set that is too small shows first.
//
//   build/examples/cpp/example_linearSys_simulate

#include "contDynamics/linearSys/linearSys.h"
#include "global/rng.h"

#include <algorithm>
#include <iostream>

using namespace cora::ct;

int main() {
    const Tensor A({{-0.1, 1.0}, {-1.0, -0.1}});
    const Zonotope X0(Tensor({1.0, 0.0}), Tensor({{0.1, 0.0}, {0.0, 0.1}}));
    const LinearSys sys(A);
    cora::Rng rng(1);  // a seed makes the points repeatable

    // 20 random points of X0 (uniform in the generator cube), and 20 corners of it.
    const Tensor random = X0.rand_point(20, rng);
    const Tensor extreme = X0.rand_point(20, rng, /*extreme=*/true);

    // x[k] holds the 20 points at time k * 0.05; the columns are the trajectories.
    const std::vector<Tensor> x = sys.simulate(random, /*time_step=*/0.05, /*t_final=*/1.0);
    std::cout << "time points: " << x.size() << ", points per time point: " << x[0].shape()[1] << "\n";

    // simulate_random draws the start points itself, from any initial set.
    const std::vector<Tensor> y = sys.simulate_random(X0, 20, 0.05, 1.0, rng);

    // Every simulated point must be inside the reachable set of its step: along a direction d
    // the largest d'x is at most the support function of the enclosure.
    const Reach R = sys.reach(X0, 0.1, 1.0, 8);
    const Tensor d({-1.0, 0.5});
    bool inside = true;
    for (std::size_t j = 0; j < x.size(); ++j) {
        const std::size_t step = std::min<std::size_t>(j / 2, R.time_int.size() - 1);  // 0.05 = 0.1 / 2
        const std::vector<double> along = d.transpose().matmul(sys.simulate(extreme, 0.05, 1.0)[j]).data();
        inside &= *std::max_element(along.begin(), along.end()) <=
                  R.time_int[step].support_func(d).data()[0] + 1e-9;
    }
    std::cout << "extreme-point trajectories stay in the reachable set: " << (inside ? "yes" : "NO") << "\n";
}
