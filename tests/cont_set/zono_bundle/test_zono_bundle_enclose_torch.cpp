// test_zono_bundle_enclose_torch - ZonoBundle::enclose: gradients through torch
//
// The enclosure of each pair of members is built of sums and concatenations: differentiable.
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

/// The gradient of the enclosure of two bundles, and with a map and a shift.
void enclose_gradient() {
    const std::vector<torch::Tensor> A = test::random_bundle_parts(2, 2, 3),
                                     B = test::random_bundle_parts(2, 2, 2);
    test::gradcheck(
        "enclose",
        [](const Inputs &x) { return test::parts(test::bundle(x, 0, 2).enclose(test::bundle(x, 4, 2))); },
        test::joined(A, B));
    test::gradcheck(
        "enclose (M, Zplus)",
        [](const Inputs &x) {
            return test::parts(test::bundle(x, 0, 2).enclose(x[4], Zonotope(x[5], x[6])));
        },
        test::joined(A, {torch::rand({2, 2}, torch::kDouble) + 0.5, B[0], B[1]}));
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(53);
    enclose_gradient();
    return test::finish("zono_bundle enclose (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
