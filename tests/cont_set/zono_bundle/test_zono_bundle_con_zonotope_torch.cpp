// test_zono_bundle_con_zonotope_torch - ZonoBundle::con_zonotope: gradients through torch
//
// The constrained zonotope is built of concatenations and sums: differentiable in the members.
//
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_zonotope/con_zonotope_gradients.h"
#include "zono_bundle_gradients.h"

using namespace cora;
using test::Inputs;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The gradient of the center, generators and constraints of the conversion.
void con_zonotope_gradient() {
    test::gradcheck(
        "con_zonotope",
        [](const Inputs &x) { return test::parts(test::bundle(x, 0, 3).con_zonotope()); },
        test::random_bundle_parts(3, 2, 3));
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(57);
    con_zonotope_gradient();
    return test::finish("zono_bundle con_zonotope (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
