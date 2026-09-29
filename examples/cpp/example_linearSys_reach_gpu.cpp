// The reachable set on the GPU: allocate the tensors there and everything built from them
// stays there. Prints a note and exits if the machine has no CUDA device.
//
//   build/examples/cpp/example_linearSys_reach_gpu

#include "contDynamics/linearSys/linearSys.h"
#include "tensor/torch.h"

#include <iostream>

using namespace cora::ct;

int main() {
    if (!torch::cuda::is_available()) {
        std::cout << "no CUDA device: nothing to show\n";
        return 0;
    }
    setBackend("torch");

    // A device per tensor: "gpu" allocates on CUDA, "cpu" on the host.
    const Tensor A({{-0.1, 1.0}, {-1.0, -0.1}}, "gpu");
    const Zonotope X0(Tensor({1.0, 0.0}, "gpu"), Tensor({{0.1, 0.0}, {0.0, 0.1}}, "gpu"));
    const Reach R = LinearSys(A).reach(X0, 0.1, 1.0, 8);

    std::cout << "A lives on " << A.device() << ", the result on " << R.timeInt.back().c.device()
              << "\n";
    // Reading values copies them to the host; .to("cpu") moves a tensor.
    std::cout << "last center " << R.timeInt.back().c.to("cpu").data()[0] << ", "
              << R.timeInt.back().c.to("cpu").data()[1] << "\n";

    // Or make the GPU the default for every tensor: setBackend("torch:cuda") (or, without
    // recompiling, CORACPP_BACKEND=torch:cuda).
}
