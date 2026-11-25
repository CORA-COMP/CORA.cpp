// test_ellipsoid_cart_prod_torch - Ellipsoid cart_prod: gradients through torch
//
// With a point the product is built of concatenations: differentiable in both operands. With an
// ellipsoid it is piecewise: the radii of the boxes are widened below 5e-3, which the inputs avoid.
//
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "ellipsoid_gradients.h"

using namespace cora;
using test::Inputs;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(14);
    std::vector<torch::Tensor> x = test::random_ellipsoid_factors(2);
    x.push_back(torch::rand({3}, torch::kDouble) + 0.5);
    test::gradcheck(
        "cart_prod (point)",
        [](const Inputs &xs) { return test::parts(test::ellipsoid_of(xs).cart_prod(xs[2])); }, x);
    test::gradcheck(
        "cart_prod (point first)",
        [](const Inputs &xs) {
            return test::parts(Ellipsoid::cart_prod(xs[2], test::ellipsoid_of(xs)));
        },
        x);
    std::vector<torch::Tensor> y = test::random_ellipsoid_factors(2);
    for (const torch::Tensor &t : test::random_ellipsoid_factors(3)) y.push_back(t);
    test::gradcheck(
        "cart_prod (ellipsoid)",
        [](const Inputs &xs) {
            // the gradient of the product
            return test::parts(test::ellipsoid_of(xs, 0).cart_prod(test::ellipsoid_of(xs, 2)));
        },
        y);
    return test::finish("ellipsoid cart_prod (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
