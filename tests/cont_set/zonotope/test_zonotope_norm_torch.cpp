// test_zonotope_norm_torch - zonotope norm: gradients through torch
//
// norm is piecewise: the gradient follows the entry that attains the largest magnitude.

#include "zonotope_gradients.h"

using namespace cora;
using test::Inputs;

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(40);
    for (const double type : {1.0, 2.0}) {
        test::gradcheck(
            "norm", [type](const Inputs &x) { return Inputs{Zonotope(x[0], x[1]).norm(type)}; },
            test::random_zonotope(3, 4));
    }
    return test::finish("zonotope norm (torch)");
}
