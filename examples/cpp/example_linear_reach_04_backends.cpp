// example_linear_reach_04_backends - the same code on Eigen and on libtorch
//
// The backend is chosen once, with setBackend (or CORACPP_BACKEND from outside); every tensor is
// made on it and everything built from a tensor stays there. Nothing else in the code changes.
//
// Syntax:   build/examples/cpp/example_linear_reach_04_backends
// Outputs:  the last enclosure's box on each backend, and how much the two differ
// See also: example_linear_reach_06_gpu

#include "contDynamics/linearSys/linearSys.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

using namespace cora::ct;

/// The reachable set of a damped oscillator, on whichever backend is current.
Reach reachOscillator() {
    // Parameters --------------------------------------------------------------------------

    const Zonotope R0(Tensor({1.0, 0.0}), Tensor({{0.1, 0.0}, {0.0, 0.1}}));

    // System Dynamics ---------------------------------------------------------------------

    const LinearSys sys(Tensor({{-0.1, 1.0}, {-1.0, -0.1}}));

    // Reachability Analysis ---------------------------------------------------------------

    return sys.reach(R0, 0.1, 1.0, 8);
}

int main() {
    // Backends --------------------------------------------------------------------------------

    std::vector<std::string> backends{"eigen"};
#ifdef CORACPP_TORCH
    backends.push_back("torch");
#endif

    // Reachability Analysis -------------------------------------------------------------------

    std::vector<std::vector<double>> centers;
    for (const std::string &spec : backends) {
        setBackend(spec);
        const Reach R = reachOscillator();
        centers.push_back(R.timeInt.back().c.data());
        std::cout << spec << ": the last center is (" << centers.back()[0] << ", "
                  << centers.back()[1] << ")\n";
    }

    // Evaluation ------------------------------------------------------------------------------

    for (std::size_t i = 1; i < centers.size(); ++i)
        std::cout << "eigen and " << backends[i] << " differ by "
                  << std::max(std::abs(centers[0][0] - centers[i][0]),
                              std::abs(centers[0][1] - centers[i][1]))
                  << "\n";

    // example completed
    return 0;
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
