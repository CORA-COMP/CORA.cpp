// test_capsule_interval_torch - Capsule interval: gradients through torch
//
// The bounds are c -+ (|g| + r): piecewise differentiable (the absolute value).
//
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "capsule_gradients.h"

using namespace cora;
using test::Inputs;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(38);
    test::gradcheck(
        "interval",
        [](const Inputs &x) {
            const Interval I = test::capsule(x).interval();
            return Inputs{I.inf(), I.sup()};
        },
        test::random_capsule_parts(3));
    return test::finish("capsule interval (torch)");
}


// ---------------------------------------  END OF CODE  ---------------------------------------- //
