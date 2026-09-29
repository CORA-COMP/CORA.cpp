// Random points and simulations over a batch of sets, and on the GPU.

#include "contDynamics/linearSys/linearSys.h"
#include "contSet/interval/interval.h"
#include "contSet/zonotope/zonotope.h"
#include "global/rng.h"
#include "tensor/torch.h"
#include "testing.h"

using namespace cora::ct;
using test::check;

namespace {

double max_diff(const torch::Tensor &a, const torch::Tensor &b) {
    return (a.to(torch::kCPU) - b.to(torch::kCPU)).abs().max().item<double>();
}

/// Each batch member's random points lie in that member's set: along random directions the
/// best point stays under the member's support.
void batched_rand_point(const torch::TensorOptions &opts, const std::string &where) {
    const int64_t S = 3, n = 3, m = 4, N = 500;
    cora::Rng rng(3);
    const torch::Tensor c = torch::randn({S, n, 1}, opts) * 5.0, G = torch::randn({S, n, m}, opts);
    const Zonotope Z(from_torch(c), from_torch(G));
    for (const bool extreme : {false, true}) {
        const torch::Tensor P = to_torch(Z.rand_point(N, rng, extreme));
        check(P.sizes().vec() == std::vector<int64_t>({S, n, N}), where + ": batched rand_point shape");
        bool inside = true;
        for (int k = 0; k < 20; ++k) {
            const torch::Tensor d = torch::randn({n, 1}, opts);
            const torch::Tensor rho = to_torch(Z.support_func(from_torch(d))).reshape({S});
            const torch::Tensor best = std::get<0>(d.transpose(0, 1).matmul(P).reshape({S, N}).max(-1));
            inside &= (best - rho).max().item<double>() <= 1e-9;
        }
        check(inside, where + ": batched points stay in their own set");
    }

    // The same seed draws the same points, on any device.
    cora::Rng a(7), b(7);
    check(max_diff(to_torch(Z.rand_point(20, a)), to_torch(Z.rand_point(20, b))) == 0.0,
          where + ": seeded batched draws");
}

void batched_interval_rand_point(const torch::TensorOptions &opts, const std::string &where) {
    const int64_t S = 4, n = 2, N = 300;
    cora::Rng rng(5);
    const torch::Tensor lo = torch::randn({S, n, 1}, opts), hi = lo + torch::rand({S, n, 1}, opts) + 0.1;
    const Interval I(from_torch(lo), from_torch(hi));
    const torch::Tensor P = to_torch(I.rand_point(N, rng));
    check(P.sizes().vec() == std::vector<int64_t>({S, n, N}), where + ": batched box points");
    check((P - lo).min().item<double>() >= 0.0 && (hi - P).min().item<double>() >= 0.0,
          where + ": batched box points stay in their box");
}

/// Simulating from batched points under batched systems is one call and equals per-member calls.
void batched_simulate(const torch::TensorOptions &opts, const std::string &where) {
    const int64_t S = 3, n = 3, N = 6;
    const torch::Tensor As = torch::randn({S, n, n}, opts) * 0.5, x0 = torch::randn({S, n, N}, opts);
    const std::vector<Tensor> all = LinearSys(from_torch(As)).simulate(from_torch(x0), 0.1, 1.0);
    check(all.size() == 11 && to_torch(all[3]).sizes().vec() == std::vector<int64_t>({S, n, N}),
          where + ": batched simulate shapes");
    for (int64_t b = 0; b < S; ++b) {
        const std::vector<Tensor> one = LinearSys(from_torch(As[b])).simulate(from_torch(x0[b]), 0.1, 1.0);
        double worst = 0.0;
        for (std::size_t k = 0; k < all.size(); ++k)
            worst = std::max(worst, max_diff(to_torch(all[k])[b], to_torch(one[k])));
        check(worst < 1e-12, where + ": a batched trajectory differs from its own run, member " + std::to_string(b));
    }
    // One system, many start batches: the systems' identity broadcast keeps the shape.
    const std::vector<Tensor> shared = LinearSys(from_torch(As[0])).simulate(from_torch(x0), 0.1, 0.5);
    check(to_torch(shared.back()).sizes().vec() == std::vector<int64_t>({S, n, N}), where + ": one system, batched starts");
}

} // namespace

int main() {
    std::vector<torch::TensorOptions> all{torch::TensorOptions().dtype(torch::kDouble)};
    std::vector<std::string> names{"cpu"};
    if (torch::cuda::is_available()) {
        all.push_back(torch::TensorOptions().dtype(torch::kDouble).device(torch::kCUDA, 0));
        names.push_back("gpu");
    }
    set_backend("torch");
    torch::manual_seed(2);
    for (std::size_t i = 0; i < all.size(); ++i) {
        batched_rand_point(all[i], names[i]);
        batched_interval_rand_point(all[i], names[i]);
        batched_simulate(all[i], names[i]);
    }
    set_backend("eigen");
    return test::finish("zonotope (libtorch)");
}
