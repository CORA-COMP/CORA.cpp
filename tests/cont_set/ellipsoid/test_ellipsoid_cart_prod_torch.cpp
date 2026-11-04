// test_ellipsoid_cart_prod_torch - Ellipsoid cart_prod: gradients through torch
//
// With a point the product is built of concatenations: differentiable in both operands. With an
// ellipsoid it is piecewise: the radii of the boxes are widened below 5e-3, which the inputs avoid.

#include "ellipsoid_gradients.h"

using namespace cora;
using test::Inputs;

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(14);
    std::vector<torch::Tensor> x = test::random_ellipsoid_factors(2);
    x.push_back(torch::rand({3, 1}, torch::kDouble) + 0.5);
    test::gradcheck(
        "cart_prod (point)",
        [](const Inputs &x) { return test::parts(test::ellipsoid_of(x).cart_prod(x[2])); }, x);
    test::gradcheck(
        "cart_prod (point first)",
        [](const Inputs &x) {
            return test::parts(Ellipsoid::cart_prod(x[2], test::ellipsoid_of(x)));
        },
        x);
    std::vector<torch::Tensor> y = test::random_ellipsoid_factors(2);
    for (const torch::Tensor &t : test::random_ellipsoid_factors(3)) y.push_back(t);
    test::gradcheck(
        "cart_prod (ellipsoid)",
        [](const Inputs &x) {
            return test::parts(test::ellipsoid_of(x, 0).cart_prod(test::ellipsoid_of(x, 2)));
        },
        y);
    return test::finish("ellipsoid cart_prod (torch)");
}
