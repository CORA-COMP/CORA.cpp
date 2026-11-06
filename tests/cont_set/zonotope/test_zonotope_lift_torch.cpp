// test_zonotope_lift_torch - zonotope lift: gradients through torch
//
// lift is a permutation of the rows: differentiable.
//
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "zonotope_gradients.h"

using namespace cora;
using test::Inputs;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(39);
    test::gradcheck(
        "lift",
        [](const Inputs &x) { return test::parts(Zonotope(x[0], x[1]).lift(3, {2, 0, 1})); },
        test::random_zonotope(3, 4));
    return test::finish("zonotope lift (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
