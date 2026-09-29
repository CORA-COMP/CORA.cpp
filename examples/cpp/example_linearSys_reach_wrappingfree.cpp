// The two algorithms of linearSys.reach: `Standard` encloses every step's set again,
// `WrappingFree` computes the first step's enclosure once and maps it forward. Both cover the
// same trajectories; a fast rotation with big steps, from an initial set that is not aligned
// with the axes, shows how far apart they are: `Standard` is the tighter one.
//
//   build/examples/cpp/example_linearSys_reach_wrappingfree

#include "contDynamics/linearSys/linearSys.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>

using namespace cora::ct;

int main() {
    const Tensor A({{-0.1, 3.0}, {-3.0, -0.1}});
    const Zonotope X0(Tensor({1.0, 0.5}), Tensor({{0.15, 0.05}, {-0.05, 0.1}}));
    const LinearSys sys(A);

    const Reach standard = sys.reach(X0, 0.3, 2.4, 8, Algorithm::Standard);
    const Reach wrapping_free = sys.reach(X0, 0.3, 2.4, 8, Algorithm::WrappingFree);

    // The extent of every enclosure along x1: the support function in the direction (1, 0).
    const Tensor e1({1.0, 0.0});
    std::cout << std::fixed << std::setprecision(5) << "step   standard   wrapping-free\n";
    for (std::size_t k = 0; k < standard.time_int.size(); ++k)
        std::cout << std::setw(4) << k << "   " << standard.time_int[k].support_func(e1).data()[0]
                  << "    " << wrapping_free.time_int[k].support_func(e1).data()[0] << "\n";

    // The time points are e^{A t} X0 either way.
    const std::vector<double> a = standard.time_point.back().c.data(),
                              b = wrapping_free.time_point.back().c.data();
    std::cout << "last time point, largest difference in the center: "
              << std::max(std::abs(a[0] - b[0]), std::abs(a[1] - b[1])) << "\n";
}
