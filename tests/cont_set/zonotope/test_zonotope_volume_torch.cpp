// test_zonotope_volume_torch - zonotope volume: gradients through torch

#include "zonotope_gradients.h"

using namespace cora;
using test::Inputs;

namespace {

/// The exact sum of |det| and the Alamo bound are differentiable in the generators.
void volume_gradient() {
    const std::vector<torch::Tensor> Z = test::random_zonotope(2, 4);
    test::gradcheck(
        "volume (exact)", [](const Inputs &x) { return Inputs{Zonotope(x[0], x[1]).volume()}; }, Z);
    test::gradcheck(
        "volume (alamo)",
        [](const Inputs &x) { return Inputs{Zonotope(x[0], x[1]).volume("alamo")}; }, Z);
}

} // namespace

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(44);
    volume_gradient();
    return test::finish("zonotope volume (torch)");
}
