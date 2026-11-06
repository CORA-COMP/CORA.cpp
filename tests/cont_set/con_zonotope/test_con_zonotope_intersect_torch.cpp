// test_con_zonotope_intersect_torch - ConZonotope intersect: gradients through torch
//
// The agreement rows are matrix products: differentiable in both operands; the zero rows and
// columns that compact drops are chosen on detached values.
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
    const std::vector<torch::Tensor> A = test::random_con_zonotope_parts(2, 3, 1),
                                     B = test::random_con_zonotope_parts(2, 2, 1);
    std::vector<torch::Tensor> x = A;
    x.insert(x.end(), B.begin(), B.end());
    test::gradcheck(
        "intersect",
        [](const Inputs &x) { return test::parts(test::set(x, 0).intersect(test::set(x, 4))); }, x);
    return test::finish("con zonotope intersect (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
