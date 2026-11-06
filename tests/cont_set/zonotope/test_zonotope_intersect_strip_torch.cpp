// test_zonotope_intersect_strip_torch - zonotope intersect_strip: gradients through torch
//
// intersect_strip is differentiable for a given gain and for "normGen" (matrix products, an
// inverse).
//
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "zonotope_gradients.h"

using namespace cora;
using test::Inputs;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(42);
    const std::vector<torch::Tensor> Z = test::random_zonotope(3, 4);
    const torch::Tensor C = torch::randn({2, 3}, torch::kDouble),
                        phi = torch::rand({2, 1}, torch::kDouble) + 0.5,
                        y = torch::randn({2, 1}, torch::kDouble),
                        lambda = torch::randn({3, 2}, torch::kDouble);
    test::gradcheck("intersect_strip normGen",
                    [](const Inputs &x) {
                        return test::parts(Zonotope(x[0], x[1]).intersect_strip(x[2], x[3], x[4]));
                    },
                    {Z[0], Z[1], C, phi, y});
    test::gradcheck("intersect_strip gain",
                    [](const Inputs &x) {
                        return test::parts(
                            Zonotope(x[0], x[1]).intersect_strip(x[2], x[3], x[4], x[5]));
                    },
                    {Z[0], Z[1], C, phi, y, lambda});
    // the gradient check
    return test::finish("zonotope intersect_strip (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
