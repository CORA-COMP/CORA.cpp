// test_zonotope_compact_torch - zonotope compact: gradients through torch
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

/// The choice of generators is made on detached values: the gradient flows to the ones that stay.
void compact_gradient() {
    // differences step by 1e-6: tol 1e-3 drops the tiny generator and merges the antiparallel one
    const double tol = 1e-3;
    std::vector<torch::Tensor> Z = test::random_zonotope(3, 5);
    Z[1].index_put_({torch::indexing::Slice(), 2}, 1e-9);
    Z[1].index_put_({torch::indexing::Slice(), 4}, -2 * Z[1].index({torch::indexing::Slice(), 0}));
    for (const std::string method : {"zeros", "all"})
        test::gradcheck(
            "compact " + method,
            [&](const Inputs &x) { return test::parts(Zonotope(x[0], x[1]).compact(method, tol)); },
            Z);
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(38);
    compact_gradient();
    return test::finish("zonotope compact (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
