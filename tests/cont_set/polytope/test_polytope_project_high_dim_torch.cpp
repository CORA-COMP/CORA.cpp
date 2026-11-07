// test_polytope_project_high_dim_torch - Polytope project_high_dim: gradients through torch
//
// The rows are placed at the new positions and the vertices padded with zeros: differentiable.
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
    const Tensor V = test::sphere_points(rng, 3, 5);
    const std::vector<torch::Tensor> H = test::random_halfspaces(4, 3);
    test::gradcheck(
        "project_high_dim (vertices)",
        [](const Inputs &x) { return test::parts(Polytope(x[0]).project_high_dim(5, {4, 0, 2})); },
        {to_torch(V)});
    test::gradcheck(
        "project_high_dim (halfspaces)",
        [](const Inputs &x) {
            const Polytope::Constraints C =
                test::halfspaces(x).project_high_dim(5, {4, 0, 2}).constraints();
            return Inputs{C.A, C.b, C.Ae, C.be};
        },
        H);
    return test::finish("polytope project_high_dim (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
