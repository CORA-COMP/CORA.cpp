// test_polytope_intersect_torch - Polytope intersect: gradients through torch
//
// The rows of both sets stack: differentiable in both operands.

#include "polytope_gradients.h"

using namespace cora;
using test::Inputs;

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(31);
    std::vector<torch::Tensor> x = test::random_halfspaces(3, 2);
    const std::vector<torch::Tensor> B = test::random_halfspaces(2, 2);
    x.insert(x.end(), B.begin(), B.end());
    test::gradcheck(
        "intersect",
        [](const Inputs &x) {
            return test::parts(test::halfspaces(x, 0).intersect(test::halfspaces(x, 2)));
        },
        x);
    return test::finish("polytope intersect (torch)");
}
