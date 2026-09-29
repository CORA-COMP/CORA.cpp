// The same reachable set on Eigen: one line names the backend, the rest is backend-free.
//
//   build/examples/cpp/example_linearSys_reach_eigen

#include "contDynamics/linearSys/linearSys.h"

#include <iostream>

using namespace cora::ct;

int main() {
    set_backend("eigen");  // the only line that differs from the libtorch example

    const Tensor A({{-0.1, 1.0}, {-1.0, -0.1}});
    const Zonotope X0(Tensor({1.0, 0.0}), Tensor({{0.1, 0.0}, {0.0, 0.1}}));
    const Reach R = LinearSys(A).reach(X0, 0.1, 1.0, 8);

    std::cout << "backend: " << backend().name() << ", tensors live on " << A.device() << "\n"
              << "generators of the last enclosure: " << R.time_int.back().G.shape()[1] << "\n";
}
