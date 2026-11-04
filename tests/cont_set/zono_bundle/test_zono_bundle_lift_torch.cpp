// test_zono_bundle_lift_torch - ZonoBundle::lift: gradients through torch
//
// A permutation of the rows: differentiable (a gather).
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

/// The gradient of the lifted bundle with respect to its members.
void lift_gradient() {
    test::gradcheck(
        "lift", [](const Inputs &x) { return test::parts(test::bundle(x, 0, 2).lift(3, {2, 0, 1})); },
        test::random_bundle_parts(2, 3, 2));
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(55);
    lift_gradient();
    return test::finish("zono_bundle lift (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
