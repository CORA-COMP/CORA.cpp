// test_polytope_center_torch - Polytope center: gradients through torch
//
// The mean of the extreme vertices: the choice of the vertices carries no gradient, the mean is
// differentiable in them.
//
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "polytope_gradients.h"

using namespace cora;
using test::Inputs;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(31);
    Rng rng(31);
    const Tensor sphere = test::sphere_points(rng, 3, 6);
    test::gradcheck("center", [](const Inputs &x) { return Inputs{Polytope(x[0]).center("avg")}; },
                    {to_torch(sphere)});
    return test::finish("polytope center (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
