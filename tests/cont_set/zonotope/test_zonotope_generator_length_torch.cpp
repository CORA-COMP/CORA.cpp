// test_zonotope_generator_length_torch - zonotope generator_length: gradients through torch
//
// generator_length is differentiable away from zero generators.

#include "zonotope_gradients.h"

using namespace cora;
using test::Inputs;

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(38);
    test::gradcheck(
        "generator_length",
        [](const Inputs &x) { return Inputs{Zonotope(x[0], x[1]).generator_length()}; },
        test::random_zonotope(3, 4));
    return test::finish("zonotope generator_length (torch)");
}
