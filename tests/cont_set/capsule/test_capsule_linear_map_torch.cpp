// test_capsule_linear_map_torch - Capsule linear_map: gradients through torch
//
// The image is a product and the radius scales by the Frobenius norm: differentiable.
//
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "capsule_gradients.h"

using namespace cora;
using test::Inputs;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(37);
    std::vector<torch::Tensor> x = test::random_capsule_parts(3);
    x.push_back(torch::randn({2, 3}, torch::kDouble));
    test::gradcheck(
        "linear_map",
        [](const Inputs &x) { return test::parts(test::capsule(x).linear_map(x[3])); }, x);
    return test::finish("capsule linear_map (torch)");
}


// ---------------------------------------  END OF CODE  ---------------------------------------- //
