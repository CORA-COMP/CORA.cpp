// example_linear_reach_06_gpu - reachability on the GPU
//
// Tensors allocated on a CUDA device stay there: everything built from them runs on it. Prints a
// note and returns if the machine has no CUDA device.
//
// Syntax:   build/examples/cpp/example_linear_reach_06_gpu
// Outputs:  where the system and the result live, and the last center
// See also: example_linear_reach_05_batch

#include "contDynamics/linearSys/linearSys.h"
#include "global/backend/torch.h"

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

using namespace cora;

int main() {
    if (!torch::cuda::is_available()) {
        std::cout << "no CUDA device: nothing to show\n";
        return 0;
    }

    setBackend("torch"); // gpu requires torch
    // A device per tensor: "gpu" allocates on CUDA, "cpu" on the host.
    const std::string device = "gpu";


    // Parameters ------------------------------------------------------------------------------

    const Zonotope R0(Tensor({1.0, 0.0}, device), Tensor({{0.1, 0.0}, {0.0, 0.1}}, device));

    // System Dynamics -------------------------------------------------------------------------

    const Tensor A({{-0.1, 1.0}, {-1.0, -0.1}}, device);
    const LinearSys sys(A);

    // Reachability Analysis -------------------------------------------------------------------

    const auto timerVal = std::chrono::steady_clock::now();
    const Reach R = sys.reach(R0, 0.1, 1.0, 8);
    const std::chrono::duration<double> tComp = std::chrono::steady_clock::now() - timerVal;

    std::cout << "computation time of reachable set: " << tComp.count() << " s\n";

    // Evaluation ------------------------------------------------------------------------------

    // Reading values copies them to the host; .to("cpu") moves a tensor.
    const std::vector<double> c = R.timeInt.back().c.to("cpu").data();
    std::cout << "A lives on " << A.device() << ", the result on " << R.timeInt.back().c.device()
              << "\nlast center " << c[0] << ", " << c[1] << "\n";

    // The GPU can also be the default for every tensor: setBackend("torch:cuda"), or, without
    // recompiling, CORACPP_BACKEND=torch:cuda.

    // example completed
    return 0;
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
