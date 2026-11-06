// test_zonotope_quad_map_torch - zonotope quad_map: gradients through torch
//
// quad_map is polynomial in the center and the generators: differentiable (the generators that
// are exactly zero are dropped on the values).
//
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
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
    torch::manual_seed(54);
    const std::vector<Tensor> Q{cora::from_torch(torch::randn({3, 3}, torch::kDouble)),
                                cora::from_torch(torch::randn({3, 3}, torch::kDouble))};
    test::gradcheck(
        "quad_map", [&](const Inputs &x) { return test::parts(Zonotope(x[0], x[1]).quad_map(Q)); },
        test::random_zonotope(3, 3));
    test::gradcheck("quad_map (two zonotopes)",
                    [&](const Inputs &x) {
                        return test::parts(Zonotope(x[0], x[1]).quad_map(Zonotope(x[2], x[3]), Q));
                    },
                    {test::random_zonotope(3, 2)[0], test::random_zonotope(3, 2)[1],
                     test::random_zonotope(3, 3)[0], test::random_zonotope(3, 3)[1]});
    return test::finish("zonotope quad_map (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
