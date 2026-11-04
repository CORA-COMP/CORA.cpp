// test_polytope_con_zonotope_torch - Polytope con_zonotope: gradients through torch
//
// With a given box the result is built of products and corner choices: differentiable in the
// rows, offsets and box, piecewise in the signs of the rows.

#include "polytope_gradients.h"

using namespace cora;
using test::Inputs;

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(31);
    std::vector<torch::Tensor> x = test::random_halfspaces(3, 2);
    x.push_back(-torch::rand({2, 1}, torch::kDouble) - 1.0);
    x.push_back(torch::rand({2, 1}, torch::kDouble) + 1.0);
    test::gradcheck(
        "con_zonotope",
        [](const Inputs &x) {
            const ConZonotope cz =
                test::halfspaces(x, 0).con_zonotope("exact:supportFunc", Interval(x[2], x[3]));
            return Inputs{cz.c(), cz.generators(), cz.A(), cz.b()};
        },
        x);
    return test::finish("polytope con_zonotope (torch)");
}
