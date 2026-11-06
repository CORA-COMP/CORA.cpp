// test_capsule_convex_hull_torch - Capsule convex_hull: gradients through torch
//
// The hull is built from sums, norms and a maximum: piecewise differentiable.
//
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "capsule_gradients.h"

using namespace cora;
using test::Inputs;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(35);
    std::vector<torch::Tensor> x = test::random_capsule_parts(3);
    const std::vector<torch::Tensor> y = test::random_capsule_parts(3);
    x.insert(x.end(), y.begin(), y.end());
    test::gradcheck(
        "convex_hull",
        [](const Inputs &x) {
            return test::parts(test::capsule(x, 0).convex_hull(test::capsule(x, 3)));
        },
        x);
    return test::finish("capsule convex_hull (torch)");
}


// ---------------------------------------  END OF CODE  ---------------------------------------- //
