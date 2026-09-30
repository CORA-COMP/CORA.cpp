// test_linearSys_simulate_torch - linearSys simulate over batches of systems and start points

#include "contDynamics/linearSys/linearSys.h"
#include "tensor/torch.h"
#include "testing.h"

#include <algorithm>

using namespace cora;
using test::check;
using test::close;

namespace {

double max_diff(const torch::Tensor &a, const torch::Tensor &b) {
    return (a.to(torch::kCPU) - b.to(torch::kCPU)).abs().max().item<double>();
}

/// Simulating from batched points under batched systems is one call and equals per-member calls.
void batched_simulate(const torch::TensorOptions &opts, const std::string &where) {
    const int64_t S = 3, n = 3, N = 6;
    const torch::Tensor As = torch::randn({S, n, n}, opts) * 0.5,
                        x0 = torch::randn({S, n, N}, opts);
    const std::vector<Tensor> all = LinearSys(fromTorch(As)).simulate(fromTorch(x0), 0.1, 1.0);
    check(all.size() == 11 && toTorch(all[3]).sizes().vec() == std::vector<int64_t>({S, n, N}),
          where + ": batched simulate shapes");
    for (int64_t b = 0; b < S; ++b) {
        const std::vector<Tensor> one =
            LinearSys(fromTorch(As[b])).simulate(fromTorch(x0[b]), 0.1, 1.0);
        double worst = 0.0;
        for (std::size_t k = 0; k < all.size(); ++k)
            worst = std::max(worst, max_diff(toTorch(all[k])[b], toTorch(one[k])));
        check(worst < 1e-12, where + ": a batched trajectory differs from its own run, member " +
                                 std::to_string(b));
    }
    // One system, many start batches: the systems' identity broadcast keeps the shape.
    const std::vector<Tensor> shared =
        LinearSys(fromTorch(As[0])).simulate(fromTorch(x0), 0.1, 0.5);
    check(toTorch(shared.back()).sizes().vec() == std::vector<int64_t>({S, n, N}),
          where + ": one system, batched starts");
}

} // namespace

int main() {
    test::for_each_device([](const torch::TensorOptions &opts, const std::string &where) {
        batched_simulate(opts, where);
    });
    return test::finish("linearSys simulate (libtorch)");
}
