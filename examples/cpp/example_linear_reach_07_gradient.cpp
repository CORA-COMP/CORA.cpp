// example_linear_reach_07_gradient - how the reachable set depends on the system
//
// libtorch's autograd differentiates through the whole computation. The matrix exponential can
// use a hand-written backward pass instead of autograd's own; the gradient is the same.
//
// Syntax:   build/examples/cpp/example_linear_reach_07_gradient
// Outputs:  the total width of the enclosures and its gradient with respect to A
// See also: example_linear_reach_05_batch

#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/global/backend/torch.h"

#include <iostream>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

using namespace cora;

/// The total width of the enclosures for the system A: a scalar to differentiate.
torch::Tensor total_width(const torch::Tensor &A, bool custom_backward) {
    // Parameters --------------------------------------------------------------------------

    const Zonotope R0(Tensor({1.0, 0.0}), Tensor({{0.1, 0.0}, {0.0, 0.1}}));

    // Reachability Analysis ---------------------------------------------------------------

    const ReachSet R = LinearSys(from_torch(A, custom_backward))
                           .reach({.initial_set = R0, .t_final = 1.0},
                                  alg::Standard{.time_step = 0.1, .taylor_terms = 8});

    // Width -------------------------------------------------------------------------------

    torch::Tensor width = torch::zeros({}, torch::kDouble);
    for (const Zonotope &Z : R.time_interval) width = width + to_torch(Z.generators()).abs().sum();
    return width;
}

int main() {
    const BackendScope backend_scope("torch");

    // System Dynamics -------------------------------------------------------------------------

    torch::Tensor A =
        torch::tensor({{-0.1, 1.0}, {-1.0, -0.1}}, torch::kDouble).requires_grad_(true);

    // Gradient --------------------------------------------------------------------------------

    torch::Tensor width = total_width(A, /*custom_backward=*/false);
    width.backward();
    std::cout << "width " << width.item<double>() << ", d width / dA:\n" << A.grad() << "\n";

    // Custom Backward -------------------------------------------------------------------------

    torch::Tensor B = A.detach().clone().requires_grad_(true);
    total_width(B, /*custom_backward=*/true).backward();
    std::cout << "with the custom backward, the largest difference: "
              << (A.grad() - B.grad()).abs().max().item<double>() << "\n";

    // example completed
    return 0;
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
