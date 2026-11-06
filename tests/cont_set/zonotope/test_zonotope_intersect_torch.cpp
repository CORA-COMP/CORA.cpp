// test_zonotope_intersect_torch - zonotope intersect: gradients of the averaging through torch
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

/// The weights 1 / trace(G G') and the averaged center and generators: differentiable.
void averaging_gradient() {
    const std::vector<torch::Tensor> A = test::random_zonotope(2, 3),
                                     B = test::random_zonotope(2, 2);
    test::gradcheck("intersect (averaging)",
                    [](const Inputs &x) {
                        return test::parts(
                            Zonotope(x[0], x[1]).intersect(Zonotope(x[2], x[3]), "averaging"));
                    },
                    {A[0], A[1], B[0], B[1]});
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(36);
    averaging_gradient();
    return test::finish("zonotope intersect (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
