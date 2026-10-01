// test_zonotope_stack_torch - zonotope stack: a batch member is the zonotope it was made from

#include "contSet/zonotope/zonotope.h"
#include "global/rng.h"
#include "global/backend/torch.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

/// Member b of the batch has the center and the support function of the b-th zonotope, and
/// generators beyond its own are zero.
void stack_keeps_each_member() {
    cora::Rng rng(6);
    const int64_t n = 3;
    const std::vector<Zonotope> Zs = {Zonotope::generateRandom(n, 2, rng),
                                      Zonotope::generateRandom(n, 5, rng),
                                      Zonotope::generateRandom(n, 3, rng)};
    const Zonotope S = Zonotope::stack(Zs);
    check(S.c.shape() == std::vector<int64_t>({3, n, 1}), "stack: center shape");
    check(S.G.shape() == std::vector<int64_t>({3, n, 5}), "stack: padded to the most generators");
    const torch::Tensor c = toTorch(S.c), G = toTorch(S.G);
    for (int64_t b = 0; b < 3; ++b) {
        check(torch::allclose(c[b], toTorch(Zs[b].c)), "stack: member center");
        const int64_t m = Zs[b].G.shape().back();
        check(torch::allclose(G[b].slice(-1, 0, m), toTorch(Zs[b].G)), "stack: member generators");
        check(G[b].slice(-1, m).abs().sum().item<double>() == 0.0, "stack: padding is zero");
        const torch::Tensor d = torch::randn({n, 1}, torch::kDouble);
        check(torch::allclose(toTorch(S.supportFunc(fromTorch(d)))[b],
                              toTorch(Zs[b].supportFunc(fromTorch(d)))),
              "stack: member support function");
    }
}

void stack_of_nothing_is_described() {
    check(test::throws([] { Zonotope::stack({}); }), "stack: an empty list throws");
}

} // namespace

int main() {
    setBackend("torch");
    stack_keeps_each_member();
    stack_of_nothing_is_described();
    return test::finish("zonotope stack (torch)");
}
