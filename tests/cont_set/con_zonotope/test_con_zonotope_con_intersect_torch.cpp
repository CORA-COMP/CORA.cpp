// test_con_zonotope_con_intersect_torch - ConZonotope con_intersect: gradients through torch
//
// The constraint rows are products with the matrix R: differentiable in both sets and R.
//
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "con_zonotope_gradients.h"

using namespace cora;
using test::Inputs;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(34);
    const std::vector<torch::Tensor> A = test::random_con_zonotope_parts(3, 3, 1),
                                     B = test::random_con_zonotope_parts(2, 2, 1);
    std::vector<torch::Tensor> x = A;
    x.insert(x.end(), B.begin(), B.end());
    x.push_back(torch::randn({2, 3}, torch::kDouble));
    test::gradcheck(
        "con_intersect",
        [](const Inputs &x) {
            return test::parts(test::set(x, 0).con_intersect(test::set(x, 4), x[8]));
        },
        x);
    return test::finish("con zonotope con_intersect (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
