// test_polytope_plus_torch - Polytope plus: gradients through torch
//
// The sum of two vertex sets is all pairs of vertices and a point shifts the vertices or the
// offsets: differentiable in the operands and the point. The sum of two halfspace sets is a
// host program: no gradient (a gradient request fails).
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
    // the operands: two vertex sets and a halfspace set on spheres and random rows, and a point
    const Tensor V1 = test::sphere_points(rng, 3, 5), V2 = test::sphere_points(rng, 3, 4);
    const std::vector<torch::Tensor> H = test::random_halfspaces(4, 3);
    const torch::Tensor v = torch::randn({3, 1}, torch::kDouble);
    test::gradcheck(
        "plus (vertices)",
        [](const Inputs &x) { return test::parts(Polytope(x[0]).plus(Polytope(x[1]))); },
        {to_torch(V1), to_torch(V2)});
    test::gradcheck("plus (vertices, point)",
                    [](const Inputs &x) { return test::parts(Polytope(x[0]).plus(x[1])); },
                    {to_torch(V1), v});
    test::gradcheck("plus (halfspaces, point)",
                    [](const Inputs &x) { return test::parts(test::halfspaces(x).plus(x[2])); },
                    {H[0], H[1], v});
    return test::finish("polytope plus (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
