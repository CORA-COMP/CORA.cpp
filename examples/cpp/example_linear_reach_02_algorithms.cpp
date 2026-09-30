// example_linear_reach_02_algorithms - the two algorithms of reach: standard and wrapping-free
//
// Both cover the same trajectories. "standard" encloses the set of every step again,
// "wrapping-free" encloses the first step once and maps it forward. A fast rotation with big
// steps, from an initial set that is not aligned with the axes, shows how far apart they are.
//
// Syntax:   build/examples/cpp/example_linear_reach_02_algorithms
// Outputs:  the extent of every enclosure along x1, for both algorithms
// See also: example_linear_reach_01_5dim

#include "contDynamics/linearSys/linearSys.h"

#include <iomanip>
#include <iostream>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

using namespace cora;

int main() {
    // Parameters ------------------------------------------------------------------------------

    const double tFinal = 2.4;
    const Zonotope R0(Tensor({1.0, 0.5}), Tensor({{0.15, 0.05}, {-0.05, 0.1}}));

    // Reachability Settings -------------------------------------------------------------------

    const double timeStep = 0.3;
    const int taylorTerms = 8;

    // System Dynamics -------------------------------------------------------------------------

    const LinearSys sys(Tensor({{-0.1, 3.0}, {-3.0, -0.1}}));

    // Reachability Analysis -------------------------------------------------------------------

    const Reach Rstandard = sys.reach(R0, timeStep, tFinal, taylorTerms, Algorithm::Standard);
    const Reach RwrappingFree =
        sys.reach(R0, timeStep, tFinal, taylorTerms, Algorithm::WrappingFree);

    // Evaluation ------------------------------------------------------------------------------

    // The extent of an enclosure along x1 is its support function in the direction (1, 0).
    const Tensor x1({1.0, 0.0});
    std::cout << std::fixed << std::setprecision(5) << "step   standard   wrapping-free\n";
    for (std::size_t k = 0; k < Rstandard.timeInt.size(); ++k)
        std::cout << std::setw(4) << k << "   " << Rstandard.timeInt[k].supportFunc(x1).data()[0]
                  << "    " << RwrappingFree.timeInt[k].supportFunc(x1).data()[0] << "\n";

    // example completed
    return 0;
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
