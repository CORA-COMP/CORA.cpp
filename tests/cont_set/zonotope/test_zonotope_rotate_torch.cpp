// test_zonotope_rotate_torch - zonotope rotate: gradients through torch
//
// rotate is linear in the center and the generators: differentiable.
//
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
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
    torch::manual_seed(53);
    test::gradcheck(
        "rotate",
        [](const Inputs &x) { return test::parts(Zonotope(x[0], x[1]).rotate({0, 2}, 0.7)); },
        test::random_zonotope(3, 4));
    return test::finish("zonotope rotate (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
