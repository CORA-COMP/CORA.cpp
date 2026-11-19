// test_zonotope_ball_radius_torch - zonotope ball_radius: gradients through torch
//
// radius is piecewise (the smaller of two bounds): the gradient of the one that is taken.
//
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "zonotope_gradients.h"

using namespace cora;
using test::Inputs;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(51);
    test::gradcheck(
        "radius", [](const Inputs &x) { return Inputs{Zonotope(x[0], x[1]).ball_radius()}; },
        test::random_zonotope_rows(3, 4));
    return test::finish("zonotope radius (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
