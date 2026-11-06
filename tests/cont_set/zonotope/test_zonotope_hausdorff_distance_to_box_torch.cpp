// test_zonotope_hausdorff_distance_to_box_torch - zonotope hausdorff_distance_to_box: gradients
//
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "zonotope_gradients.h"

using namespace cora;
using test::Inputs;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Every method is piecewise (abs, max, argmax): the gradient of the branch taken, away from the
/// kinks.
void hausdorff_distance_gradient() {
    const std::vector<torch::Tensor> Z = test::random_zonotope(2, 4);
    for (const std::string method : {"exact", "naive", "wgreedy", "wopt"})
        test::gradcheck(
            "hausdorff_distance_to_box " + method,
            [&](const Inputs &x) {
                return Inputs{Zonotope(x[0], x[1]).hausdorff_distance_to_box(method)};
            },
            Z);
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(37);
    hausdorff_distance_gradient();
    return test::finish("zonotope hausdorff_distance_to_box (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
