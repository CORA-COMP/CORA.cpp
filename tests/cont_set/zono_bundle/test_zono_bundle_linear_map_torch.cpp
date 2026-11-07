// test_zono_bundle_linear_map_torch - ZonoBundle linear_map: gradients through torch
//
// A matrix product of every member: differentiable in the matrix and the members.
//
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "zono_bundle_gradients.h"

using namespace cora;
using test::Inputs;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(71);
    // the inputs are the matrix, then the two members
    const std::vector<torch::Tensor> members = test::random_bundle_parts(2, 3, 2);
    test::gradcheck(
        "linear_map",
        [](const Inputs &x) { return test::parts(test::bundle(x, 1, 2).linear_map(x[0])); },
        test::joined({torch::rand({2, 3}, torch::kDouble) - 0.5}, members));
    return test::finish("zono_bundle linear_map (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
