// test_zono_bundle_intersect_torch - ZonoBundle::intersect: gradients through torch
//
// The result lists the members of both sets: differentiable in every one.
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

/// The gradient of every part of the intersection of a bundle with a bundle and a zonotope.
void intersect_gradient() {
    test::gradcheck(
        "intersect",
        [](const Inputs &x) {
            const ZonoBundle R = test::bundle(x, 0, 2).intersect(test::bundle(x, 4, 1));
            return test::parts(R.intersect(Zonotope(x[6], x[7])));
        },
        test::joined(test::random_bundle_parts(2, 2, 3), test::random_bundle_parts(2, 2, 3)));
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(52);
    intersect_gradient();
    return test::finish("zono_bundle intersect (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
