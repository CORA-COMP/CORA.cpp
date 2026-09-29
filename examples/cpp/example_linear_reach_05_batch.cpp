// example_linear_reach_05_batch - reachability of a batch of initial sets in one call
//
// A batch lives in the object: Zonotope::stack joins four sets into one (c of shape (4, 2, 1), G
// of shape (4, 2, 2)), and the call is the one of a single set. libtorch broadcasts the leading
// dimensions of the set and of the system, so a batch of systems works the same way.
//
// Syntax:   build/examples/cpp/example_linear_reach_05_batch
// Outputs:  the shape of the result, and whether each member equals its own single run
// See also: example_linear_reach_01_5dim, example_linear_reach_06_gpu

#include "contDynamics/linearSys/linearSys.h"
#include "tensor/torch.h"

#include <iostream>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

using namespace cora::ct;

int main() {
    setBackend("torch");

    // Parameters ------------------------------------------------------------------------------

    const double tFinal = 3.0;

    // Four initial sets: different centers, and boxes of different size.
    const std::vector<Tensor> centers = {Tensor({1.0, 0.0}), Tensor({0.0, 1.0}),
                                         Tensor({-1.0, 0.0}), Tensor({0.0, -2.0})};
    const std::vector<double> scales = {0.05, 0.1, 0.2, 0.4};
    std::vector<Zonotope> singles;
    for (std::size_t b = 0; b < centers.size(); ++b)
        singles.emplace_back(centers[b], Tensor::eye(2) * scales[b]);
    const Zonotope batchR0 = Zonotope::stack(singles);

    // Reachability Settings -------------------------------------------------------------------

    const double timeStep = 0.1;
    const int taylorTerms = 8;

    // System Dynamics -------------------------------------------------------------------------

    const LinearSys sys(Tensor({{-0.1, 1.0}, {-1.0, -0.1}})); // one system for all sets

    // Reachability Analysis -------------------------------------------------------------------

    const Reach R = sys.reach(batchR0, timeStep, tFinal, taylorTerms);

    // Evaluation ------------------------------------------------------------------------------

    // Every set of the result carries the batch: centers (4, 2, 1), generators (4, 2, m).
    const torch::Tensor last = toTorch(R.timeInt.back().c);
    std::cout << "sets: " << singles.size() << ", last centers " << last.sizes() << "\n"
              << "distance from the origin at t = 3, by set:\n"
              << last.squeeze(-1).norm(2, {-1}) << "\n";

    // Verification ----------------------------------------------------------------------------

    // Each member of the batch is what the set gives on its own.
    for (std::size_t b = 0; b < singles.size(); ++b) {
        const Reach Rb = sys.reach(singles[b], timeStep, tFinal, taylorTerms);
        const double gap = (last[b] - toTorch(Rb.timeInt.back().c)).abs().max().item<double>();
        std::cout << "set " << b << " differs from its single run by " << gap << "\n";
    }

    // example completed
    return 0;
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
