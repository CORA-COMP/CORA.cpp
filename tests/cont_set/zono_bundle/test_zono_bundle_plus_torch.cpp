// test_zono_bundle_plus_torch - ZonoBundle plus: gradients through torch
//
// The sum of every pair of members: differentiable in both bundles.
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
    torch::manual_seed(72);
    // two bundles of two members each
    test::gradcheck(
        "plus",
        [](const Inputs &x) { return test::parts(test::bundle(x, 0, 2).plus(test::bundle(x, 4, 2))); },
        test::joined(test::random_bundle_parts(2, 2, 2), test::random_bundle_parts(2, 2, 2)));
    return test::finish("zono_bundle plus (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
