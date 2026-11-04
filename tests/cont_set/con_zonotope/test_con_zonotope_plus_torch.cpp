// test_con_zonotope_plus_torch - ConZonotope plus: gradients through torch
//
// The sum is built of concatenations and additions: differentiable in both operands.

#include "con_zonotope_gradients.h"

using namespace cora;
using test::Inputs;

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(34);
    const std::vector<torch::Tensor> A = test::random_con_zonotope_parts(2, 3, 1),
                                     B = test::random_con_zonotope_parts(2, 2, 1);
    std::vector<torch::Tensor> x = A;
    x.insert(x.end(), B.begin(), B.end());
    test::gradcheck(
        "plus", [](const Inputs &x) { return test::parts(test::set(x, 0).plus(test::set(x, 4))); },
        x);
    return test::finish("con zonotope plus (torch)");
}
