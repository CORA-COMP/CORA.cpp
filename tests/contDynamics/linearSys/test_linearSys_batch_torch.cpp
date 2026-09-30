// test_linearSys_batch_torch - automatic batching ("vmap"): linearSys is written for one set and
// one system, and one call over a batch of sets, of systems, or of both equals one call per member.

#include "contDynamics/linearSys/linearSys.h"
#include "tensor/torch.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

const Algorithm kAlgorithms[] = {Algorithm::Standard, Algorithm::WrappingFree};

std::string name(Algorithm a) { return a == Algorithm::Standard ? "standard" : "wrapping-free"; }

double max_diff(const torch::Tensor &a, const torch::Tensor &b) {
    return (a.to(torch::kCPU) - b.to(torch::kCPU)).abs().max().item<double>();
}

/// A set from a centre `(..., n)` and generators `(..., n, m)`.
Zonotope make_set(const torch::Tensor &c, const torch::Tensor &G) {
    return {fromTorch(c.unsqueeze(-1)), fromTorch(G)};
}

Reach run(const torch::Tensor &A, const torch::Tensor &c, const torch::Tensor &G, Algorithm a) {
    return LinearSys(fromTorch(A)).reach(make_set(c, G), 0.1, 1.0, 8, a);
}

/// Whether member `b` of `batched` is `single`, for every step, centre and generators.
bool member_is(const Reach &batched, int64_t b, const Reach &single) {
    for (std::size_t k = 0; k < single.timeInt.size(); ++k) {
        if (max_diff(toTorch(batched.timeInt[k].c)[b], toTorch(single.timeInt[k].c)) > 1e-12 ||
            max_diff(toTorch(batched.timeInt[k].G)[b], toTorch(single.timeInt[k].G)) > 1e-12)
            return false;
    }
    for (std::size_t k = 0; k < single.timePoint.size(); ++k)
        if (max_diff(toTorch(batched.timePoint[k].G)[b], toTorch(single.timePoint[k].G)) > 1e-12)
            return false;
    return true;
}

void batches(const torch::TensorOptions &opts, const std::string &where) {
    const int64_t n = 3, m = 4, S = 3;
    const torch::Tensor A = torch::randn({n, n}, opts) * 0.5;
    const torch::Tensor As = torch::randn({S, n, n}, opts) * 0.5;
    const torch::Tensor cs = torch::randn({S, n}, opts), Gs = torch::randn({S, n, m}, opts);

    for (const Algorithm algorithm : kAlgorithms) {
        const std::string what = where + ": " + name(algorithm);
        const Reach sets = run(A, cs, Gs, algorithm);
        const Reach systems = run(As, cs[0], Gs[0], algorithm);
        const Reach both = run(As, cs, Gs, algorithm);
        for (int64_t b = 0; b < S; ++b) {
            check(member_is(sets, b, run(A, cs[b], Gs[b], algorithm)),
                  what + ": a batch of sets, member " + std::to_string(b));
            check(member_is(systems, b, run(As[b], cs[0], Gs[0], algorithm)),
                  what + ": a batch of systems, member " + std::to_string(b));
            check(member_is(both, b, run(As[b], cs[b], Gs[b], algorithm)),
                  what + ": sets and systems together, member " + std::to_string(b));
        }
    }
}

/// Two batch dimensions, and a batch of one, go through unchanged.
void shapes(const torch::TensorOptions &opts, const std::string &where) {
    const int64_t n = 2, m = 3;
    const torch::Tensor A = torch::randn({n, n}, opts) * 0.5;
    const Reach grid = run(A, torch::randn({2, 3, n}, opts), torch::randn({2, 3, n, m}, opts),
                           Algorithm::Standard);
    check(toTorch(grid.timeInt[0].c).sizes().vec() == std::vector<int64_t>({2, 3, n, 1}),
          where + ": two batch dimensions");
    const Reach one =
        run(A, torch::randn({1, n}, opts), torch::randn({1, n, m}, opts), Algorithm::Standard);
    check(toTorch(one.timePoint[0].c).size(0) == 1, where + ": a batch of one");
}

} // namespace

int main() {
    std::vector<torch::TensorOptions> all{torch::TensorOptions().dtype(torch::kDouble)};
    std::vector<std::string> names{"cpu"};
    if (torch::cuda::is_available()) {
        all.push_back(torch::TensorOptions().dtype(torch::kDouble).device(torch::kCUDA, 0));
        names.push_back("gpu");
    }
    setBackend("torch");
    torch::manual_seed(7);
    for (std::size_t i = 0; i < all.size(); ++i) {
        batches(all[i], names[i]);
        shapes(all[i], names[i]);
    }
    return test::finish("linearSys batch");
}
