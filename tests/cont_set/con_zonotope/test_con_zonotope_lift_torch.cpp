// test_con_zonotope_lift_torch - ConZonotope lift: gradients through torch
//
// A permutation of the rows: differentiable.
//
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "con_zonotope_gradients.h"

using namespace cora;
using test::Inputs;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(34);
    test::gradcheck(
        "lift", [](const Inputs &x) { return test::parts(test::set(x, 0).lift(3, {2, 0, 1})); },
        test::random_con_zonotope_parts(3, 3, 1));
    return test::finish("con zonotope lift (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
