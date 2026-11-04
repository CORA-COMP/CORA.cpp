// test_con_zonotope_quad_map_torch - ConZonotope quad_map: gradients through torch
//
// The enclosure is built of products and sums: differentiable in the set and in Q. A set with
// constraints is rescaled first by bounds from the host, so the check uses one without.

#include "con_zonotope_gradients.h"

using namespace cora;
using test::Inputs;

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(34);
    std::vector<torch::Tensor> x = test::random_con_zonotope_parts(2, 3, 0);
    x.push_back(torch::randn({2, 2}, torch::kDouble));
    x.push_back(torch::randn({2, 2}, torch::kDouble));
    test::gradcheck(
        "quad_map",
        [](const Inputs &x) { return test::parts(test::set(x, 0).quad_map({x[4], x[5]})); }, x);
    return test::finish("con zonotope quad_map (torch)");
}
