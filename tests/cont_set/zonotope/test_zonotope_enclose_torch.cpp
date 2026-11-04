// test_zonotope_enclose_torch - zonotope enclose and convex_hull: gradients through torch

#include "zonotope_gradients.h"

using namespace cora;
using test::Inputs;

namespace {

/// Averages and differences of the generators: differentiable in both zonotopes, also when they
/// have different numbers of generators.
void enclose_gradient() {
    const std::vector<torch::Tensor> A = test::random_zonotope(3, 4),
                                     B = test::random_zonotope(3, 2);
    test::gradcheck("enclose",
                    [](const Inputs &x) {
                        return test::parts(Zonotope(x[0], x[1]).enclose(Zonotope(x[2], x[3])));
                    },
                    {A[0], A[1], B[0], B[1]});
    test::gradcheck("convex_hull",
                    [](const Inputs &x) {
                        return test::parts(Zonotope(x[2], x[3]).convex_hull(Zonotope(x[0], x[1])));
                    },
                    {A[0], A[1], B[0], B[1]});
}

} // namespace

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(34);
    enclose_gradient();
    return test::finish("zonotope enclose (torch)");
}
