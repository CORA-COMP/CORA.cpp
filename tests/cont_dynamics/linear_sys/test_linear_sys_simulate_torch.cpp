// test_linear_sys_simulate_torch - linear_sys simulate over batches of systems and start points

#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/global/backend/torch.h"
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
    const std::vector<Tensor> all = LinearSys(from_torch(As))
                                        .simulate({.initial_state = from_torch(x0), .t_final = 1.0},
                                                  alg::Exact{.time_step = 0.1});
    check(all.size() == 11 && to_torch(all[3]).sizes().vec() == std::vector<int64_t>({S, n, N}),
          where + ": batched simulate shapes");
    for (int64_t b = 0; b < S; ++b) {
        const std::vector<Tensor> one =
            LinearSys(from_torch(As[b]))
                .simulate({.initial_state = from_torch(x0[b]), .t_final = 1.0},
                          alg::Exact{.time_step = 0.1});
        double worst = 0.0;
        for (std::size_t k = 0; k < all.size(); ++k)
            worst = std::max(worst, max_diff(to_torch(all[k])[b], to_torch(one[k])));
        check(worst < 1e-12, where + ": a batched trajectory differs from its own run, member " +
                                 std::to_string(b));
    }
    // One system, many start batches: the systems' identity broadcast keeps the shape.
    const std::vector<Tensor> shared =
        LinearSys(from_torch(As[0]))
            .simulate({.initial_state = from_torch(x0), .t_final = 0.5},
                      alg::Exact{.time_step = 0.1});
    check(to_torch(shared.back()).sizes().vec() == std::vector<int64_t>({S, n, N}),
          where + ": one system, batched starts");
}

} // namespace

int main() {
    test::for_each_device([](const torch::TensorOptions &opts, const std::string &where) {
        batched_simulate(opts, where);
    });
    return test::finish("linear_sys simulate (libtorch)");
}
