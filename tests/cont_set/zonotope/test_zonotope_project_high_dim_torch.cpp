// test_zonotope_project_high_dim_torch - zonotope project_high_dim: gradients through torch
//
// project_high_dim places the rows of the center and the generators: differentiable.
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
    torch::manual_seed(52);
    test::gradcheck(
        "project_high_dim",
        [](const Inputs &x) {
            return test::parts(Zonotope(x[0], x[1]).project_high_dim(5, {3, 0, 4}));
        },
        test::random_zonotope(3, 4));
    return test::finish("zonotope project_high_dim (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
