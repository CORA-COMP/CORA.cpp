// test_zonotope_exact_plus_torch - zonotope exact_plus: gradients through torch
//
// exact_plus is differentiable: sums and copies of the inputs.

#include "zonotope_gradients.h"

using namespace cora;
using test::Inputs;

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(37);
    const std::vector<torch::Tensor> Z = test::random_zonotope(3, 4),
                                     Z2 = test::random_zonotope(3, 2);
    test::gradcheck("exact_plus",
                    [](const Inputs &x) {
                        return test::parts(Zonotope(x[0], x[1]).exact_plus(Zonotope(x[2], x[3])));
                    },
                    {Z[0], Z[1], Z2[0], Z2[1]});
    return test::finish("zonotope exact_plus (torch)");
}
