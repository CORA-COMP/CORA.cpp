// test_zonotope_rotate_torch - zonotope rotate: gradients through torch
//
// rotate is linear in the center and the generators: differentiable.

#include "zonotope_gradients.h"

using namespace cora;
using test::Inputs;

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(53);
    test::gradcheck(
        "rotate",
        [](const Inputs &x) { return test::parts(Zonotope(x[0], x[1]).rotate({0, 2}, 0.7)); },
        test::random_zonotope(3, 4));
    return test::finish("zonotope rotate (torch)");
}
