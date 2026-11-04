// test_ellipsoid_ellipsoid_norm_torch - Ellipsoid ellipsoid_norm: gradients through torch
//
// The norm of a full-rank ellipsoid sqrt(p' Q^-1 p) is differentiable in Q, q and p away from
// the origin.

#include "ellipsoid_gradients.h"

using namespace cora;
using test::Inputs;

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(12);
    std::vector<torch::Tensor> x = test::random_ellipsoid_factors(3);
    x.push_back(torch::rand({3, 1}, torch::kDouble) + 0.5);
    test::gradcheck(
        "ellipsoid_norm",
        [](const Inputs &x) { return Inputs{test::ellipsoid_of(x).ellipsoid_norm(x[2])}; }, x);
    return test::finish("ellipsoid ellipsoid_norm (torch)");
}
