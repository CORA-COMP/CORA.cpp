// test_con_zonotope_project_torch - ConZonotope project: gradients through torch
//
// A selection of rows: differentiable.
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
        "project",
        [](const Inputs &x) {
            return test::parts(test::set(x, 0).project({1, 0}).as<ConZonotope>());
        },
        test::random_con_zonotope_parts(3, 3, 1));
    return test::finish("con zonotope project (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
