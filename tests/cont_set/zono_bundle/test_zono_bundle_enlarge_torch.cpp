// test_zono_bundle_enlarge_torch - ZonoBundle::enlarge: gradients through torch
//
// Each member is scaled about its center: differentiable in the members and in the factor.
//
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "zono_bundle_gradients.h"

using namespace cora;
using test::Inputs;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The gradient of the enlarged bundle with respect to its members and the factor column.
void enlarge_gradient() {
    test::gradcheck(
        "enlarge",
        [](const Inputs &x) {
            return test::parts(Set(test::bundle(x, 0, 2))->enlarge(x[4]).as<ZonoBundle>());
        },
        test::joined(test::random_bundle_parts(2, 3, 2),
                     {torch::rand({3, 1}, torch::kDouble) + 0.5}));
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(54);
    enlarge_gradient();
    return test::finish("zono_bundle enlarge (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
