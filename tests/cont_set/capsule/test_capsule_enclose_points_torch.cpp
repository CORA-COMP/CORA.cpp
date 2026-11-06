// test_capsule_enclose_points_torch - Capsule enclose_points: gradients through torch
//
// The center is a mean, the generator and the radius are picked by a maximum and clipped: piecewise
// differentiable, the choice held fixed.
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
    torch::manual_seed(39);
    test::gradcheck("enclose_points",
                    [](const Inputs &x) { return test::parts(Capsule::enclose_points(x[0])); },
                    {torch::randn({3, 6}, torch::kDouble)});
    return test::finish("capsule enclose_points (torch)");
}


// ---------------------------------------  END OF CODE  ---------------------------------------- //
