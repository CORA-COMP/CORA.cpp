// test_zonotope_radius_torch - zonotope radius: gradients through torch
//
// radius is piecewise (the smaller of two bounds): the gradient of the one that is taken.

#include "zonotope_gradients.h"

using namespace cora;
using test::Inputs;

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(51);
    test::gradcheck(
        "radius", [](const Inputs &x) { return Inputs{Zonotope(x[0], x[1]).radius()}; },
        test::random_zonotope(3, 4));
    return test::finish("zonotope radius (torch)");
}
