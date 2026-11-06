// test_zonotope_underapproximate_torch - zonotope underapproximate: gradients through torch
//
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "zonotope_gradients.h"

using namespace cora;
using test::Inputs;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The signs are chosen on the values (piecewise): the vertices are linear in c and G for them.
void underapproximate_gradient() {
    const std::vector<torch::Tensor> Z = test::random_zonotope(2, 3);
    const torch::Tensor S = torch::randn({2, 2}, torch::kDouble);
    test::gradcheck(
        "underapproximate",
        [](const Inputs &x) { return Inputs{Zonotope(x[0], x[1]).underapproximate(x[2])}; },
        {Z[0], Z[1], S});
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(43);
    underapproximate_gradient();
    return test::finish("zonotope underapproximate (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
