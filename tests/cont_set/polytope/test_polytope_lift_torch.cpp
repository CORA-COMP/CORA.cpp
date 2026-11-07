// test_polytope_lift_torch - Polytope lift: gradients through torch
//
// The rows keep their columns at the new positions: differentiable. The lifted set has
// halfspaces only.
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
    const std::vector<torch::Tensor> H = test::random_halfspaces(4, 3);
    test::gradcheck(
        "lift", [](const Inputs &x) { return test::parts(test::halfspaces(x).lift(4, {3, 0, 2})); },
        H);
    return test::finish("polytope lift (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
