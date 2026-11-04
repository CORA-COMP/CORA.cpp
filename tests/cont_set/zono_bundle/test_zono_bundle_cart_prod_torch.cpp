// test_zono_bundle_cart_prod_torch - ZonoBundle::cart_prod: gradients through torch
//
// The product is built of concatenations: differentiable in both operands.
//
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "zono_bundle_gradients.h"

using namespace cora;
using test::Inputs;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The gradient of every part of the product of two bundles with respect to both.
void cart_prod_gradient() {
    test::gradcheck(
        "cart_prod",
        [](const Inputs &x) { return test::parts(test::bundle(x, 0, 2).cart_prod(test::bundle(x, 4, 2))); },
        test::joined(test::random_bundle_parts(2, 2, 3), test::random_bundle_parts(2, 3, 2)));
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(51);
    cart_prod_gradient();
    return test::finish("zono_bundle cart_prod (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
