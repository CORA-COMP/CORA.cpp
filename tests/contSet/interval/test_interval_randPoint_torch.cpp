// test_interval_randPoint_torch - interval randPoint over a batch of boxes

#include "contSet/interval/interval.h"
#include "global/rng.h"
#include "tensor/torch.h"
#include "testing.h"

using namespace cora::ct;
using test::check;
using test::close;

namespace {

void batched_interval_randPoint(const torch::TensorOptions &opts, const std::string &where) {
    const int64_t S = 4, n = 2, N = 300;
    cora::Rng rng(5);
    const torch::Tensor lo = torch::randn({S, n, 1}, opts),
                        hi = lo + torch::rand({S, n, 1}, opts) + 0.1;
    const Interval I(fromTorch(lo), fromTorch(hi));
    const torch::Tensor P = toTorch(I.randPoint(N, rng));
    check(P.sizes().vec() == std::vector<int64_t>({S, n, N}), where + ": batched box points");
    check((P - lo).min().item<double>() >= 0.0 && (hi - P).min().item<double>() >= 0.0,
          where + ": batched box points stay in their box");
}

} // namespace

int main() {
    test::for_each_device([](const torch::TensorOptions &opts, const std::string &where) {
        batched_interval_randPoint(opts, where);
    });
    return test::finish("interval randPoint (libtorch)");
}
