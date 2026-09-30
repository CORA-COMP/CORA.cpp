// test_tensor_customBackward_torch - Tensor expm: the hand-written backward pass against autograd's
// own

#include "tensor/tensor.h"
#include "tensor/torch.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;

namespace {

double max_diff(const torch::Tensor &a, const torch::Tensor &b) {
    return (a.to(torch::kCPU) - b.to(torch::kCPU)).abs().max().item<double>();
}

/// The hand-written backward pass of `e^A` equals autograd's own.
void customBackward_matches_autograd() {
    torch::manual_seed(1);
    for (const int64_t n : {1, 3, 5}) {
        const torch::Tensor A0 = torch::randn({2, n, n}, torch::kDouble) * 0.7;
        const torch::Tensor weight = torch::randn({2, n, n}, torch::kDouble);
        std::vector<torch::Tensor> grads;
        for (const bool custom : {false, true}) {
            torch::Tensor A = A0.clone().requires_grad_(true);
            const torch::Tensor loss = (toTorch(fromTorch(A, custom).expm()) * weight).sum();
            grads.push_back(torch::autograd::grad({loss}, {A})[0]);
        }
        check(max_diff(grads[0], grads[1]) < 1e-10,
              "custom backward of expm, n = " + std::to_string(n));
    }
}

} // namespace

int main() {
    customBackward_matches_autograd();
    setBackend("eigen");
    return test::finish("tensor customBackward (libtorch)");
}
