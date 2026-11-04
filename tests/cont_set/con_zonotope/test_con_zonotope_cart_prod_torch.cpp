// test_con_zonotope_cart_prod_torch - ConZonotope cart_prod: gradients through torch
//
// The product is built of concatenations: differentiable in both operands.

#include "con_zonotope_gradients.h"

using namespace cora;
using test::Inputs;

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(34);
    const std::vector<torch::Tensor> A = test::random_con_zonotope_parts(2, 3, 1),
                                     B = test::random_con_zonotope_parts(3, 2, 1);
    std::vector<torch::Tensor> x = A;
    x.insert(x.end(), B.begin(), B.end());
    test::gradcheck(
        "cart_prod",
        [](const Inputs &x) { return test::parts(test::set(x, 0).cart_prod(test::set(x, 4))); }, x);
    return test::finish("con zonotope cart_prod (torch)");
}
