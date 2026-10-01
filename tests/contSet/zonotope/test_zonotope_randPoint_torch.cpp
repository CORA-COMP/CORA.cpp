// test_zonotope_randPoint_torch - zonotope randPoint over a batch of sets: each member's points lie
// in its own set

#include "contSet/zonotope/zonotope.h"
#include "global/rng.h"
#include "global/backend/torch.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;

namespace {

double max_diff(const torch::Tensor &a, const torch::Tensor &b) {
    return (a.to(torch::kCPU) - b.to(torch::kCPU)).abs().max().item<double>();
}

/// Each batch member's random points lie in that member's set: along random directions the
/// best point stays under the member's support.
void batched_randPoint(const torch::TensorOptions &opts, const std::string &where) {
    const int64_t S = 3, n = 3, m = 4, N = 500;
    cora::Rng rng(3);
    const torch::Tensor c = torch::randn({S, n, 1}, opts) * 5.0, G = torch::randn({S, n, m}, opts);
    const Zonotope Z(fromTorch(c), fromTorch(G));
    for (const bool extreme : {false, true}) {
        const torch::Tensor P = toTorch(Z.randPoint(N, rng, extreme ? "extreme" : "standard"));
        check(P.sizes().vec() == std::vector<int64_t>({S, n, N}),
              where + ": batched randPoint shape");
        bool inside = true;
        for (int k = 0; k < 20; ++k) {
            const torch::Tensor d = torch::randn({n, 1}, opts);
            const torch::Tensor rho = toTorch(Z.supportFunc(fromTorch(d))).reshape({S});
            const torch::Tensor best =
                std::get<0>(d.transpose(0, 1).matmul(P).reshape({S, N}).max(-1));
            inside &= (best - rho).max().item<double>() <= 1e-9;
        }
        check(inside, where + ": batched points stay in their own set");
    }

    // The same seed draws the same points, on any device.
    cora::Rng a(7), b(7);
    check(max_diff(toTorch(Z.randPoint(20, a)), toTorch(Z.randPoint(20, b))) == 0.0,
          where + ": seeded batched draws");
}

} // namespace

int main() {
    test::for_each_device([](const torch::TensorOptions &opts, const std::string &where) {
        batched_randPoint(opts, where);
    });
    return test::finish("zonotope randPoint (libtorch)");
}
