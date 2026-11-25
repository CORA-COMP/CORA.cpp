// test_polytope_cart_prod_torch - Polytope cart_prod: gradients through torch
//
// The product stacks and pads: differentiable in both operands, in both forms.
//
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   19-November-2026
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
        [](const Inputs &xs) {
            return test::parts(test::halfspaces(xs, 0).cart_prod(test::halfspaces(xs, 2)));
        },
        x);
    test::gradcheck(
        "cart_prod, vertices",
        [](const Inputs &xs) { return test::parts(Polytope(xs[0]).cart_prod(Polytope(xs[1]))); },
        {torch::randn({3, 2}, torch::kDouble), torch::randn({2, 3}, torch::kDouble)});
    return test::finish("polytope cart_prod (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
