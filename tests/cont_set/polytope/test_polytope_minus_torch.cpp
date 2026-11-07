// test_polytope_minus_torch - Polytope minus: gradients through torch
//
// The set shifted by -v: differentiable in the set and the point, in both forms.
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
    const std::vector<torch::Tensor> H = test::random_halfspaces(4, 3);
    const torch::Tensor v = torch::randn({3, 1}, torch::kDouble);
    test::gradcheck(
        "minus (vertices)", [](const Inputs &x) { return test::parts(Polytope(x[0]).minus(x[1])); },
        {to_torch(V), v});
    test::gradcheck(
        "minus (halfspaces)",
        [](const Inputs &x) { return test::parts(test::halfspaces(x).minus(x[2])); },
        {H[0], H[1], v});
    return test::finish("polytope minus (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
