// test_zonotope_box_torch - zonotope box: gradients through torch
//
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "zonotope_gradients.h"

using namespace cora;
using test::Inputs;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The generators of the box are the row sums of |G|: piecewise in G, away from zero.
void box_gradient() {
    const std::vector<torch::Tensor> Z = test::random_zonotope_rows(3, 4);
    test::gradcheck(
        "box", [](const Inputs &x) { return test::parts(Zonotope(x[0], x[1]).box()); }, Z);
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(32);
    box_gradient();
    return test::finish("zonotope box (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
