// test_tensor_batch_torch - Tensor on libtorch: batch dimensions broadcast

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

void batches_broadcast() {
    set_default_backend("torch");
    const torch::Tensor batch = torch::randn({3, 2, 2}, torch::kDouble);
    const torch::Tensor single = torch::randn({2, 2}, torch::kDouble);
    const Tensor B = from_torch(batch), S = from_torch(single);

    check(B.matmul(S).shape() == std::vector<int64_t>({3, 2, 2}), "matmul broadcasts the batch");
    check((B + S).shape() == std::vector<int64_t>({3, 2, 2}), "add broadcasts the batch");
    check(max_diff(to_torch(B.matmul(S)), batch.matmul(single)) < 1e-14, "the batched product");
    check(B.matrix_transpose().shape() == std::vector<int64_t>({3, 2, 2}), "transpose keeps the batch");
    check(B.sum(-1, true).shape() == std::vector<int64_t>({3, 2, 1}), "sum_last keeps the batch");
    check(B.eye_like().shape() == std::vector<int64_t>({3, 2, 2}),
          "eye_like takes the batch shape");
    check(Tensor::concat({B, B}, -1).shape() == std::vector<int64_t>({3, 2, 4}),
          "cat_last per batch");
    check(from_torch(torch::randn({3, 2, 1}, torch::kDouble)).diag().shape() ==
              std::vector<int64_t>({3, 2, 2}),
          "diag per batch");

    // The exponential of a batch is the batch of exponentials.
    const torch::Tensor e = to_torch(B.expm());
    bool same = true;
    for (int i = 0; i < 3; ++i) same &= max_diff(e[i], torch::matrix_exp(batch[i])) < 1e-13;
    check(same, "expm per batch element");
}

} // namespace

int main() {
    batches_broadcast();
    set_default_backend("eigen");
    return test::finish("tensor batch (libtorch)");
}
