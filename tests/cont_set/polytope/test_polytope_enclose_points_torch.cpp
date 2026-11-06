// test_polytope_enclose_points_torch - Polytope enclose_points: gradients through torch
//
// The extreme points are chosen without gradient; the result takes its columns from the cloud.
//
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "polytope_gradients.h"

using namespace cora;
using test::Inputs;


// ===========================================  MAIN  =========================================== //

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(31);
    Rng rng(31);
    const Tensor sphere = test::sphere_points(rng, 2, 6);
    const Tensor inner = sphere.mean(-1, true) * 1.0;
    test::gradcheck("enclose_points",
                    [](const Inputs &x) { return test::parts(Polytope::enclose_points(x[0])); },
                    {to_torch(Tensor::concat({sphere, inner}, -1))});
    return test::finish("polytope enclose_points (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
