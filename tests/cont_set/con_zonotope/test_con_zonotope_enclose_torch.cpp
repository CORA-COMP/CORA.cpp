// test_con_zonotope_enclose_torch - ConZonotope enclose: gradients through torch
//
// The enclosure is built from the generators, differences and sums: differentiable.
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
    const std::vector<torch::Tensor> A = test::random_con_zonotope_parts(2, 3, 1);
    std::vector<torch::Tensor> x = A;
    x.push_back(torch::randn({2, 2}, torch::kDouble));
    x.push_back(torch::randn({2, 1}, torch::kDouble));
    test::gradcheck(
        "enclose", [](const Inputs &x) { return test::parts(test::set(x, 0).enclose(x[4], x[5])); },
        x);
    return test::finish("con zonotope enclose (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
