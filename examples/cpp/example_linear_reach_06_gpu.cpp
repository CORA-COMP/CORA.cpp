// example_linear_reach_06_gpu - reachability on the GPU
//
// Tensors allocated on a CUDA device stay there: everything built from them runs on it. Prints a
// note and returns if the machine has no CUDA device.
//
// Syntax:   build/examples/cpp/example_linear_reach_06_gpu
// Outputs:  where the system and the result live, and the last center
// See also: example_linear_reach_05_batch

#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/global/backend/torch.h"

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

    const BackendScope backend_scope("torch"); // gpu requires torch
    // A device per tensor: "gpu" allocates on CUDA, "cpu" on the host.
    const std::string device = "gpu";


    // Parameters ------------------------------------------------------------------------------

    const Zonotope R0(Tensor({1.0, 0.0}, device), Tensor({{0.1, 0.0}, {0.0, 0.1}}, device));

    // System Dynamics -------------------------------------------------------------------------

    const Tensor A({{-0.1, 1.0}, {-1.0, -0.1}}, device);
    const LinearSys sys(A);

    // Reachability Analysis -------------------------------------------------------------------

    const auto timer_val = std::chrono::steady_clock::now();
    const ReachSet R = sys.reach({.initial_set = R0, .t_final = 1.0},
                                 alg::Standard{.time_step = 0.1, .taylor_terms = 8});
    const std::chrono::duration<double> computation_time =
        std::chrono::steady_clock::now() - timer_val;

    std::cout << "computation time of reachable set: " << computation_time.count() << " s\n";

    // Evaluation ------------------------------------------------------------------------------

    // Reading values copies them to the host; .to("cpu") moves a tensor.
    const std::vector<double> c = R.time_interval.back().center().to("cpu").data();
    std::cout << "A lives on " << A.device() << ", the result on "
              << R.time_interval.back().center().device() << "\nlast center " << c[0] << ", "
              << c[1] << "\n";

    // The GPU can also be the default for every tensor: set_default_backend("torch:cuda") or a
    // BackendScope, or, without recompiling, CORACPP_BACKEND=torch:cuda.

    // example completed
    return 0;
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
