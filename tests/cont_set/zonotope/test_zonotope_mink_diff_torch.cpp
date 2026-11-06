// test_zonotope_mink_diff_torch - zonotope mink_diff: gradients through torch
//
// mink_diff is piecewise: "approx" and the aligned "exact" carry gradients (the facets are chosen
// on the values); the linear-program methods compute on the host and do not.
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
    torch::manual_seed(44);
    const std::vector<torch::Tensor> M = test::random_zonotope(3, 3);
    const torch::Tensor c = torch::zeros({3, 1}, torch::kDouble);
    const torch::Tensor G = torch::diag(torch::rand({3}, torch::kDouble) + 0.1);
    test::gradcheck("mink_diff approx",
                    [](const Inputs &x) {
                        return test::parts(
                            Zonotope(x[0], x[1]).mink_diff(Zonotope(x[2], x[3] * 0.3)));
                    },
                    {M[0], M[1], c, G});
    return test::finish("zonotope mink_diff (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
