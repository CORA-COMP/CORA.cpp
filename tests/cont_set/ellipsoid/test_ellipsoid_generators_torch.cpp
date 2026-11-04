// test_ellipsoid_generators_torch - Ellipsoid generators: gradients through torch
//
// G = U sqrt(D) of the singular value decomposition: differentiable in Q for distinct singular
// values, with the sign convention of the decomposition held fixed by inputs away from ties.

#include "ellipsoid_gradients.h"

using namespace cora;
using test::Inputs;

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(13);
    test::gradcheck(
        "generators", [](const Inputs &x) { return Inputs{test::ellipsoid_of(x).generators()}; },
        test::random_ellipsoid_factors(3));
    return test::finish("ellipsoid generators (torch)");
}
