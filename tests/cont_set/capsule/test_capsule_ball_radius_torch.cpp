// test_capsule_ball_radius_torch - Capsule ball_radius and volume: gradients through torch
//
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       06-November-2026
// Last update:   19-November-2026
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
        "radius", [](const Inputs &x) { return Inputs{test::capsule(x).ball_radius()}; },
        test::random_capsule_parts(3));
    test::gradcheck(
        "volume", [](const Inputs &x) { return Inputs{test::capsule(x).volume()}; },
        test::random_capsule_parts(3));
    return test::finish("capsule radius (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
