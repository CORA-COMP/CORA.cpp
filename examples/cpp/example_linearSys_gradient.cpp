// Automatic differentiation: how the size of the reachable set depends on the system, with
// libtorch's autograd through the whole computation.
//
//   build/examples/cpp/example_linearSys_gradient

#include "contDynamics/linearSys/linearSys.h"
#include "tensor/torch.h"

#include <iostream>

using namespace cora::ct;

/// The total width of the enclosures for the system `A`: a scalar to differentiate.
torch::Tensor total_width(const torch::Tensor &A, bool custom_backward) {
    const Zonotope X0(Tensor({1.0, 0.0}), Tensor({{0.1, 0.0}, {0.0, 0.1}}));
    const Reach R = LinearSys(from_torch(A, custom_backward)).reach(X0, 0.1, 1.0, 8);
    torch::Tensor width = torch::zeros({}, torch::kDouble);
    for (const Zonotope &Z : R.time_int) width = width + to_torch(Z.G).abs().sum();
    return width;
}

int main() {
    set_backend("torch");

    torch::Tensor A = torch::tensor({{-0.1, 1.0}, {-1.0, -0.1}}, torch::kDouble).requires_grad_(true);
    torch::Tensor width = total_width(A, /*custom_backward=*/false);
    width.backward();
    std::cout << "width " << width.item<double>() << ", d width / dA:\n" << A.grad() << "\n";

    // The matrix exponential can use a hand-written backward pass instead of autograd's own;
    // the gradient is the same.
    torch::Tensor B = A.detach().clone().requires_grad_(true);
    total_width(B, /*custom_backward=*/true).backward();
    std::cout << "with the custom backward, the largest difference: "
              << (A.grad() - B.grad()).abs().max().item<double>() << "\n";
}
