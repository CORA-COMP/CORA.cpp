// Automatic batching: linearSys is written for one system and one set, and a batch of systems
// (or of sets, or of both) is the same call. libtorch broadcasts the leading dimensions.
//
//   build/examples/cpp/example_linearSys_reach_batch

#include "contDynamics/linearSys/linearSys.h"
#include "tensor/torch.h"

#include <iostream>

using namespace cora::ct;

int main() {
    setBackend("torch");

    // Three oscillators that differ in their damping: A has shape (3, 2, 2).
    const torch::Tensor A = torch::tensor({{{-0.05, 1.0}, {-1.0, -0.05}},
                                           {{-0.10, 1.0}, {-1.0, -0.10}},
                                           {{-0.20, 1.0}, {-1.0, -0.20}}},
                                          torch::kDouble);
    // One initial set for all of them.
    const Zonotope X0(Tensor({1.0, 0.0}), Tensor({{0.1, 0.0}, {0.0, 0.1}}));

    const Reach R = LinearSys(fromTorch(A)).reach(X0, 0.1, 3.0, 8);

    // Every set of the result carries the batch: centers (3, 2, 1), generators (3, 2, m).
    const torch::Tensor c = toTorch(R.timeInt.back().c);
    std::cout << "systems: " << A.size(0) << ", last centers " << c.sizes() << "\n"
              << "distance from the origin at t = 3 (more damping, closer):\n"
              << c.squeeze(-1).norm(2, {-1}) << "\n";

    // A batch of initial sets under one system works the same way: give c (S, n, 1), G (S, n, m).
}
