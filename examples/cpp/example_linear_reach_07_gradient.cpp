// example_linear_reach_07_gradient - how the reachable set depends on the system
//
// libtorch's autograd differentiates through the whole computation. The matrix exponential can
// use a hand-written backward pass instead of autograd's own; the gradient is the same.
//
// Syntax:   build/examples/cpp/example_linear_reach_07_gradient
// Outputs:  the total width of the enclosures and its gradient with respect to A
// See also: example_linear_reach_05_batch

#include "contDynamics/linearSys/linearSys.h"
#include "global/backend/torch.h"

#include <iostream>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

using namespace cora;

/// The total width of the enclosures for the system A: a scalar to differentiate.
torch::Tensor totalWidth(const torch::Tensor &A, bool customBackward) {
    // Parameters --------------------------------------------------------------------------

    const Zonotope R0(Tensor({1.0, 0.0}), Tensor({{0.1, 0.0}, {0.0, 0.1}}));

    // Reachability Analysis ---------------------------------------------------------------

    const Reach R = LinearSys(fromTorch(A, customBackward)).reach(R0, 0.1, 1.0, 8);

    // Width -------------------------------------------------------------------------------

    torch::Tensor width = torch::zeros({}, torch::kDouble);
    for (const Zonotope &Z : R.timeInt) width = width + toTorch(Z.G).abs().sum();
    return width;
}

int main() {
    setBackend("torch");

    // System Dynamics -------------------------------------------------------------------------

    torch::Tensor A =
        torch::tensor({{-0.1, 1.0}, {-1.0, -0.1}}, torch::kDouble).requires_grad_(true);

    // Gradient --------------------------------------------------------------------------------

    torch::Tensor width = totalWidth(A, /*customBackward=*/false);
    width.backward();
    std::cout << "width " << width.item<double>() << ", d width / dA:\n" << A.grad() << "\n";

    // Custom Backward -------------------------------------------------------------------------

    torch::Tensor B = A.detach().clone().requires_grad_(true);
    totalWidth(B, /*customBackward=*/true).backward();
    std::cout << "with the custom backward, the largest difference: "
              << (A.grad() - B.grad()).abs().max().item<double>() << "\n";

    // example completed
    return 0;
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
