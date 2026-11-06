// test_con_zonotope_cubic_map_torch - ConZonotope cubic_map: gradients through torch
//
// The enclosure is built of products and sums: differentiable in the set and in T.
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
    std::vector<torch::Tensor> x = test::random_con_zonotope_parts(2, 3, 1);
    for (int i = 0; i < 4; ++i) x.push_back(torch::randn({2, 2}, torch::kDouble));
    test::gradcheck(
        "cubic_map",
        [](const Inputs &x) {
            return test::parts(test::set(x, 0).cubic_map({{x[4], x[5]}, {x[6], x[7]}}));
        },
        x);
    return test::finish("con zonotope cubic_map (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
