// test_capsule_radius_torch - Capsule radius and volume: gradients through torch
//
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "capsule_gradients.h"

using namespace cora;
using test::Inputs;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(211);
    test::gradcheck(
        "radius", [](const Inputs &x) { return Inputs{test::capsule(x).radius()}; },
        test::random_capsule_parts(3));
    test::gradcheck(
        "volume", [](const Inputs &x) { return Inputs{test::capsule(x).volume()}; },
        test::random_capsule_parts(3));
    return test::finish("capsule radius (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
