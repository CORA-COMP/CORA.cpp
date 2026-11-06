// test_ellipsoid_interval_torch - Ellipsoid interval: gradients through torch
//
// The bounds q -+ sqrt(diag Q): differentiable in Q and q.
//
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "ellipsoid_gradients.h"

using namespace cora;
using test::Inputs;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(11);
    test::gradcheck(
        "interval",
        [](const Inputs &x) {
            const Interval I = test::ellipsoid_of(x).interval();
            return Inputs{I.inf(), I.sup()};
        },
        test::random_ellipsoid_factors(3));
    return test::finish("ellipsoid interval (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
