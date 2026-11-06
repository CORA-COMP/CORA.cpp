// test_polytope_cart_prod_torch - Polytope cart_prod: gradients through torch
//
// The product stacks and pads: differentiable in both operands, in both forms.
//
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "polytope_gradients.h"

using namespace cora;
using test::Inputs;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(31);
    std::vector<torch::Tensor> x = test::random_halfspaces(3, 2);
    const std::vector<torch::Tensor> B = test::random_halfspaces(4, 3);
    x.insert(x.end(), B.begin(), B.end());
    test::gradcheck(
        "cart_prod",
        [](const Inputs &x) {
            return test::parts(test::halfspaces(x, 0).cart_prod(test::halfspaces(x, 2)));
        },
        x);
    test::gradcheck(
        "cart_prod, vertices",
        [](const Inputs &x) { return test::parts(Polytope(x[0]).cart_prod(Polytope(x[1]))); },
        {torch::randn({2, 3}, torch::kDouble), torch::randn({3, 2}, torch::kDouble)});
    return test::finish("polytope cart_prod (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
