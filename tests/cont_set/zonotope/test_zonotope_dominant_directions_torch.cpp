// test_zonotope_dominant_directions_torch - zonotope dominant_directions: gradients through torch
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

/// The columns are generators, chosen on detached values: the gradient is one on the chosen ones.
void dominant_directions_gradient() {
    const std::vector<torch::Tensor> Z = test::random_zonotope(3, 8);
    test::gradcheck(
        "dominant_directions",
        [](const Inputs &x) { return Inputs{Zonotope(x[0], x[1]).dominant_directions()}; }, Z);
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(39);
    dominant_directions_gradient();
    return test::finish("zonotope dominant_directions (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
