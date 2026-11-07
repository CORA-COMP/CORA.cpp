// test_polytope_project_torch - Polytope project: gradients through torch
//
// Vertices are selected by coordinate: differentiable.
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
    const Tensor V = test::sphere_points(rng, 4, 6);
    test::gradcheck(
        "project (vertices)",
        [](const Inputs &x) { return test::parts(Polytope(x[0]).project({2, 0})); },
        {to_torch(V)});
    return test::finish("polytope project (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
