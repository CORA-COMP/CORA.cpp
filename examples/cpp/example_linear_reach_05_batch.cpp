// example_linear_reach_05_batch - reachability of a batch of initial sets in one call
//
// A batch lives in the object: one Zonotope holds four sets (c of shape (4, 2, 1), G of shape
// (4, 2, 2)), and the call is the one of a single set. libtorch broadcasts the leading
// dimensions of the set and of the system, so a batch of systems works the same way.
//
// Syntax:   build/examples/cpp/example_linear_reach_05_batch
// Outputs:  the shape of the result, and whether each member equals its own single run
// See also: example_linear_reach_01_5dim, example_linear_reach_06_gpu

#include "contDynamics/linearSys/linearSys.h"
#include "tensor/torch.h"

#include <iostream>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

using namespace cora::ct;

int main() {
    setBackend("torch");

    // Parameters ------------------------------------------------------------------------------

    const double tFinal = 3.0;

    // Four initial sets: different centers, and boxes of different size.
    const torch::Tensor c =
        torch::tensor({{{1.0}, {0.0}}, {{0.0}, {1.0}}, {{-1.0}, {0.0}}, {{0.0}, {-2.0}}},
                      torch::kDouble);
    const torch::Tensor scale = torch::tensor({0.05, 0.1, 0.2, 0.4}, torch::kDouble);
    const torch::Tensor G = scale.view({4, 1, 1}) * torch::eye(2, torch::kDouble);
    const Zonotope batchR0(fromTorch(c), fromTorch(G));

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
    std::cout << "sets: " << c.size(0) << ", last centers " << last.sizes() << "\n"
              << "distance from the origin at t = 3, by set:\n"
              << last.squeeze(-1).norm(2, {-1}) << "\n";

    // Verification ----------------------------------------------------------------------------

    // Each member of the batch is what the set gives on its own.
    for (int64_t b = 0; b < c.size(0); ++b) {
        const Zonotope R0b(fromTorch(c[b]), fromTorch(G[b]));
        const Reach Rb = sys.reach(R0b, timeStep, tFinal, taylorTerms);
        const double gap =
            (last[b] - toTorch(Rb.timeInt.back().c)).abs().max().item<double>();
        std::cout << "set " << b << " differs from its single run by " << gap << "\n";
    }

    // example completed
    return 0;
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
