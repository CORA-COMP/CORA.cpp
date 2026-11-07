// test_polytope_support_function_torch - Polytope support_function: gradients through torch
//
// The largest dot product of the vertices: piecewise in the direction and the vertices, away from
// ties.
//
// Authors:       Claude Sonnet 5.5
// Written:       07-November-2026
// Last update:   ---
// Reviewed:      ---

#include "polytope_gradients.h"

using namespace cora;
using test::Inputs;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(61);
    Rng rng(61);
    const Tensor V = test::sphere_points(rng, 3, 6);
    const torch::Tensor d = torch::randn({3, 1}, torch::kDouble);
    test::gradcheck("support_function (vertices)",
                    [](const Inputs &x) { return Inputs{Polytope(x[0]).support_function(x[1])}; },
                    {to_torch(V), d});
    return test::finish("polytope support_function (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
