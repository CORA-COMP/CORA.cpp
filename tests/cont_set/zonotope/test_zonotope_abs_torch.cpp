// test_zonotope_abs_torch - zonotope abs: gradients through torch, and a batch on torch

#include "zonotope_gradients.h"

using namespace cora;
using test::check;
using test::Inputs;

namespace {

/// abs is piecewise: the gradient is the sign of the entry, away from zero.
void abs_gradient() {
    const std::vector<torch::Tensor> Z = test::random_zonotope(3, 4);
    test::gradcheck(
        "abs", [](const Inputs &x) { return test::parts(Zonotope(x[0], x[1]).abs()); }, Z);
}

/// A batch on torch is the abs of each member.
void abs_of_a_batch() {
    const std::vector<torch::Tensor> Z = test::random_zonotope(3, 4);
    const Zonotope batch(from_torch(torch::stack({Z[0], -Z[0]})),
                         from_torch(torch::stack({Z[1], -Z[1]})));
    const torch::Tensor G = to_torch(batch.abs().generators());
    check(torch::allclose(G[0], Z[1].abs()) && torch::allclose(G[1], Z[1].abs()),
          "abs: a torch batch");
}

} // namespace

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(31);
    abs_gradient();
    abs_of_a_batch();
    return test::finish("zonotope abs (torch)");
}
