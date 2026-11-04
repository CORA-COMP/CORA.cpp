// test_zonotope_cart_prod_torch - zonotope cart_prod: gradients through torch

#include "zonotope_gradients.h"

using namespace cora;
using test::Inputs;

namespace {

/// The product is built of concatenations: differentiable in both operands.
void cart_prod_gradient() {
    const std::vector<torch::Tensor> A = test::random_zonotope(2, 3),
                                     B = test::random_zonotope(3, 2);
    test::gradcheck("cart_prod",
                    [](const Inputs &x) {
                        return test::parts(Zonotope(x[0], x[1]).cart_prod(Zonotope(x[2], x[3])));
                    },
                    {A[0], A[1], B[0], B[1]});
}

} // namespace

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(33);
    cart_prod_gradient();
    return test::finish("zonotope cart_prod (torch)");
}
