// test_con_zonotope_reduce_torch - ConZonotope reduce: gradients through torch
//
// The generators kept are chosen on detached values; the result is differentiable in the set.
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
        "reduce", [](const Inputs &x) { return test::parts(test::set(x, 0).reduce(1)); },
        test::random_con_zonotope_parts(2, 5, 1));
    return test::finish("con zonotope reduce (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
