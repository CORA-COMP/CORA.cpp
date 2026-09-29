// The same reachable set on libtorch, and how to hand it a libtorch tensor you already have.
//
//   build/examples/cpp/example_linearSys_reach_torch

#include "contDynamics/linearSys/linearSys.h"
#include "tensor/torch.h"

#include <iostream>

using namespace cora::ct;

int main() {
    setBackend("torch");  // the only line that differs from the Eigen example

    const Tensor A({{-0.1, 1.0}, {-1.0, -0.1}});
    const Zonotope X0(Tensor({1.0, 0.0}), Tensor({{0.1, 0.0}, {0.0, 0.1}}));
    const Reach R = LinearSys(A).reach(X0, 0.1, 1.0, 8);
    std::cout << "backend: " << backend().name() << ", tensors live on " << A.device() << "\n";

    // A torch::Tensor you already have goes in and comes out without a copy of the algorithm:
    // fromTorch wraps it, toTorch unwraps a result.
    const torch::Tensor c = torch::tensor({{1.0}, {0.0}}, torch::kDouble);
    const torch::Tensor G = 0.1 * torch::eye(2, torch::kDouble);
    const Reach again = LinearSys(fromTorch(torch::tensor({{-0.1, 1.0}, {-1.0, -0.1}}, torch::kDouble)))
                            .reach(Zonotope(fromTorch(c), fromTorch(G)), 0.1, 1.0, 8);
    std::cout << "the last center as a torch tensor:\n" << toTorch(again.timeInt.back().c) << "\n";
}
