// test_tensor_custom_backward_torch - Tensor expm: the hand-written backward pass against
// autograd's own

#include "cora/global/backend/torch.h"
#include "cora/global/tensor/tensor.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;

namespace {

double max_diff(const torch::Tensor &a, const torch::Tensor &b) {
    return (a.to(torch::kCPU) - b.to(torch::kCPU)).abs().max().item<double>();
}

/// The hand-written backward pass of `e^A` equals autograd's own.
void custom_backward_matches_autograd() {
    torch::manual_seed(1);
    for (const int64_t n : {1, 3, 5}) {
        const torch::Tensor A0 = torch::randn({2, n, n}, torch::kDouble) * 0.7;
        const torch::Tensor weight = torch::randn({2, n, n}, torch::kDouble);
        std::vector<torch::Tensor> grads;
        for (const bool custom : {false, true}) {
            torch::Tensor A = A0.clone().requires_grad_(true);
            const torch::Tensor loss = (to_torch(from_torch(A, custom).expm()) * weight).sum();
            grads.push_back(torch::autograd::grad({loss}, {A})[0]);
        }
        check(max_diff(grads[0], grads[1]) < 1e-10,
              "custom backward of expm, n = " + std::to_string(n));
    }
}

} // namespace

int main() {
    custom_backward_matches_autograd();
    return test::finish("tensor custom_backward (libtorch)");
}
