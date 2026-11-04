// test_con_zonotope_linear_map_torch - ConZonotope linear_map: gradients through torch
//
// The image is a product with the matrix: differentiable in the set and the matrix.
//
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "con_zonotope_gradients.h"

using namespace cora;
using test::Inputs;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ===========================================  MAIN  =========================================== //

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(34);
    std::vector<torch::Tensor> x = test::random_con_zonotope_parts(2, 3, 1);
    x.push_back(torch::randn({3, 2}, torch::kDouble));
    test::gradcheck(
        "linear_map", [](const Inputs &x) { return test::parts(test::set(x, 0).linear_map(x[4])); },
        x);
    return test::finish("con zonotope linear_map (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
