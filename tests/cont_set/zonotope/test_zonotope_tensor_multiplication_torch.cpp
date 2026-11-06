// test_zonotope_tensor_multiplication_torch - zonotope tensor_multiplication: gradients, a batch
//
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "tensor_lists.h"
#include "zonotope_gradients.h"

using namespace cora;
using test::check;
using test::Inputs;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The result is a polynomial in the generators and the tensor: differentiable in all of them.
void tensor_multiplication_gradient() {
    const std::vector<torch::Tensor> Z = test::random_zonotope(2, 2);
    const torch::Tensor M = torch::randn({2, 2, 2, 2}, torch::kDouble);
    test::gradcheck("tensor_multiplication",
                    [](const Inputs &x) {
                        const Zonotope centered(x[0] * 0.0, x[1]);
                        return test::parts(centered.tensor_multiplication(x[2], test::triples(2)));
                    },
                    {Z[0], Z[1], M});
}

/// A batch on torch is the result of each member.
void tensor_multiplication_of_a_batch() {
    const std::vector<torch::Tensor> Z = test::random_zonotope(2, 2);
    const Tensor M = from_torch(torch::randn({2, 2, 2, 2}, torch::kDouble));
    const torch::Tensor zero = torch::zeros({2, 1}, torch::kDouble);
    const Zonotope batch(from_torch(torch::stack({zero, zero})),
                         from_torch(torch::stack({Z[1], Z[1] * 0.5})));
    const torch::Tensor H = to_torch(batch.tensor_multiplication(M, test::triples(2)).generators());
    const Zonotope member(from_torch(zero), from_torch(Z[1] * 0.5));
    const torch::Tensor second =
        to_torch(member.tensor_multiplication(M, test::triples(2)).generators());
    check(torch::allclose(H[1], second), "tensor_multiplication: member 2 of a torch batch");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(41);
    tensor_multiplication_gradient();
    tensor_multiplication_of_a_batch();
    return test::finish("zonotope tensor_multiplication (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
