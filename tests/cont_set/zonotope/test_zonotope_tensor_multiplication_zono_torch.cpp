// test_zonotope_tensor_multiplication_zono_torch - tensor_multiplication_zono: gradients

#include "tensor_lists.h"
#include "zonotope_gradients.h"

using namespace cora;
using test::Inputs;

namespace {

/// The sum of the results for the center and a generator of the tensor zonotope: differentiable.
void tensor_multiplication_zono_gradient() {
    const std::vector<torch::Tensor> Z = test::random_zonotope(2, 2);
    const torch::Tensor M = torch::randn({2, 2, 2, 2}, torch::kDouble);
    const torch::Tensor M1 = torch::randn({2, 2, 2, 2}, torch::kDouble);
    test::gradcheck("tensor_multiplication_zono",
                    [](const Inputs &x) {
                        const Zonotope centered(x[0] * 0.0, x[1]);
                        return test::parts(centered.tensor_multiplication_zono(
                            x[2], {x[3]}, test::triples(2)));
                    },
                    {Z[0], Z[1], M, M1});
}

} // namespace

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(42);
    tensor_multiplication_zono_gradient();
    return test::finish("zonotope tensor_multiplication_zono (torch)");
}
