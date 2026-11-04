// test_zono_bundle_interval_torch - ZonoBundle::interval: gradients through torch
//
// The box of the bundle is the intersection of the members' boxes: piecewise differentiable
// (maximum and minimum of the bounds), checked away from the kinks.
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

/// The gradient of both bounds of the box with respect to the members.
void interval_gradient() {
    test::gradcheck(
        "interval",
        [](const Inputs &x) {
            const Interval I = test::bundle(x, 0, 3).interval();
            return Inputs{I.inf(), I.sup()};
        },
        test::random_bundle_parts(3, 2, 3));
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(56);
    interval_gradient();
    return test::finish("zono_bundle interval (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
