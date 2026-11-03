// test_con_zonotope_intersect_strip_torch - ConZonotope intersect_strip: gradients through torch
//
// For a given gain the set is built of matrix products: differentiable in the set, the strips and
// the gain.

#include "con_zonotope_gradients.h"

using namespace cora;
using test::Inputs;

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(34);
    const std::vector<torch::Tensor> A = test::random_con_zonotope_parts(3, 4, 1);
    std::vector<torch::Tensor> x = A;
    x.push_back(torch::randn({2, 3}, torch::kDouble));
    x.push_back(torch::rand({2, 1}, torch::kDouble) + 0.5);
    x.push_back(torch::randn({2, 1}, torch::kDouble));
    x.push_back(torch::randn({3, 2}, torch::kDouble));
    test::gradcheck(
        "intersect_strip",
        [](const Inputs &x) {
            return test::parts(test::set(x, 0).intersect_strip(x[4], x[5], x[6], x[7]));
        },
        x);
    return test::finish("con zonotope intersect_strip (torch)");
}
