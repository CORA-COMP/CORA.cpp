// test_tensor_batch_torch - Tensor on libtorch: batch dimensions broadcast

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

void batches_broadcast() {
    setBackend("torch");
    const torch::Tensor batch = torch::randn({3, 2, 2}, torch::kDouble);
    const torch::Tensor single = torch::randn({2, 2}, torch::kDouble);
    const Tensor B = fromTorch(batch), S = fromTorch(single);

    check(B.matmul(S).shape() == std::vector<int64_t>({3, 2, 2}), "matmul broadcasts the batch");
    check((B + S).shape() == std::vector<int64_t>({3, 2, 2}), "add broadcasts the batch");
    check(max_diff(toTorch(B.matmul(S)), batch.matmul(single)) < 1e-14, "the batched product");
    check(B.transpose().shape() == std::vector<int64_t>({3, 2, 2}), "transpose keeps the batch");
    check(B.sumLast().shape() == std::vector<int64_t>({3, 2, 1}), "sumLast keeps the batch");
    check(B.eyeLike().shape() == std::vector<int64_t>({3, 2, 2}), "eyeLike takes the batch shape");
    check(Tensor::catLast({B, B}).shape() == std::vector<int64_t>({3, 2, 4}), "catLast per batch");
    check(fromTorch(torch::randn({3, 2, 1}, torch::kDouble)).diag().shape() ==
              std::vector<int64_t>({3, 2, 2}),
          "diag per batch");

    // The exponential of a batch is the batch of exponentials.
    const torch::Tensor e = toTorch(B.expm());
    bool same = true;
    for (int i = 0; i < 3; ++i) same &= max_diff(e[i], torch::matrix_exp(batch[i])) < 1e-13;
    check(same, "expm per batch element");
}

} // namespace

int main() {
    batches_broadcast();
    setBackend("eigen");
    return test::finish("tensor batch (libtorch)");
}
