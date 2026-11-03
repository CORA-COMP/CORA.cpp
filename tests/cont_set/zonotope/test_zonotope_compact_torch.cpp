// test_zonotope_compact_torch - zonotope compact: gradients through torch

#include "zonotope_gradients.h"

using namespace cora;
using test::Inputs;

namespace {

/// The choice of generators is made on detached values: the gradient flows to the ones that stay.
void compact_gradient() {
    std::vector<torch::Tensor> Z = test::random_zonotope(3, 5);
    Z[1].index_put_({torch::indexing::Slice(), 2}, 0.0); // a zero generator is dropped
    for (const std::string method : {"zeros", "all"})
        test::gradcheck("compact " + method,
                        [&](const Inputs &x) {
                            return test::parts(Zonotope(x[0], x[1]).compact(method));
                        },
                        Z);
}

} // namespace

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(38);
    compact_gradient();
    return test::finish("zonotope compact (torch)");
}
