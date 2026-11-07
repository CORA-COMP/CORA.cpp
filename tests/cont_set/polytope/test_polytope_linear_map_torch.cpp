// test_polytope_linear_map_torch - Polytope linear_map: gradients through torch
//
// The vertices map linearly: differentiable in the matrix and the vertices. The halfspaces of
// a square matrix are mapped by its inverse.
//
// Authors:       Claude Sonnet 5.5
// Written:       07-November-2026
// Last update:   ---
// Reviewed:      ---

#include "polytope_gradients.h"

using namespace cora;
using test::Inputs;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(61);
    Rng rng(61);
    const Tensor V = test::sphere_points(rng, 3, 5);
    const torch::Tensor M = torch::randn({2, 3}, torch::kDouble);
    test::gradcheck("linear_map (vertices)",
                    [](const Inputs &x) { return test::parts(Polytope(x[0]).linear_map(x[1])); },
                    {to_torch(V), M});
    const std::vector<torch::Tensor> H = test::random_halfspaces(4, 3);
    const torch::Tensor S =
        torch::randn({3, 3}, torch::kDouble) + 2 * torch::eye(3, torch::kDouble);
    test::gradcheck(
        "linear_map (halfspaces)",
        [](const Inputs &x) { return test::parts(test::halfspaces(x).linear_map(x[2])); },
        {H[0], H[1], S});
    return test::finish("polytope linear_map (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
