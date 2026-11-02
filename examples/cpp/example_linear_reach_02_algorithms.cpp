// example_linear_reach_02_algorithms - the two algorithms of reach: standard and wrapping-free
//
// Both cover the same trajectories. "standard" encloses the set of every step again,
// "wrapping-free" encloses the first step once and maps it forward. A fast rotation with big
// steps, from an initial set that is not aligned with the axes, shows how far apart they are.
//
// Syntax:   build/examples/cpp/example_linear_reach_02_algorithms
// Outputs:  the extent of every enclosure along x1, for both algorithms
// See also: example_linear_reach_01_5dim

#include "cora/cont_dynamics/linear_sys/linear_sys.h"

#include <iomanip>
#include <iostream>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

using namespace cora;

int main() {
    // Parameters ------------------------------------------------------------------------------

    const double t_final = 2.4;
    const Zonotope R0(Tensor({1.0, 0.5}), Tensor({{0.15, 0.05}, {-0.05, 0.1}}));

    // Reachability Settings -------------------------------------------------------------------

    const double time_step = 0.3;
    const int taylor_terms = 8;

    // System Dynamics -------------------------------------------------------------------------

    const LinearSys sys(Tensor({{-0.1, 3.0}, {-3.0, -0.1}}));

    // Reachability Analysis -------------------------------------------------------------------

    const ReachSet Rstandard =
        sys.reach({.initial_set = R0, .t_final = t_final},
                  alg::Standard{.time_step = time_step, .taylor_terms = taylor_terms});
    const ReachSet RwrappingFree =
        sys.reach({.initial_set = R0, .t_final = t_final},
                  alg::WrappingFree{.time_step = time_step, .taylor_terms = taylor_terms});

    // Evaluation ------------------------------------------------------------------------------

    // The extent of an enclosure along x1 is its support function in the direction (1, 0).
    const Tensor x1({1.0, 0.0});
    std::cout << std::fixed << std::setprecision(5) << "step   standard   wrapping-free\n";
    for (std::size_t k = 0; k < Rstandard.time_interval.size(); ++k)
        std::cout << std::setw(4) << k << "   "
                  << Rstandard.time_interval[k].support_function(x1).data()[0] << "    "
                  << RwrappingFree.time_interval[k].support_function(x1).data()[0] << "\n";

    // example completed
    return 0;
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
