// test_zonotope_stack_torch - zonotope stack: a batch member is the zonotope it was made from
//
// Authors:       Tobias Ladner, Claude Sonnet 5.5, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   04-November-2026
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/backend/torch.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Member b of the batch has the center and the support function of the b-th zonotope, and
/// generators beyond its own are zero.
void stack_keeps_each_member() {
    cora::Rng rng(6);
    const int64_t n = 3;
    const std::vector<Zonotope> Zs = {Zonotope::generate_random(rng, n, std::nullopt, 2),
                                      Zonotope::generate_random(rng, n, std::nullopt, 5),
                                      Zonotope::generate_random(rng, n, std::nullopt, 3)};
    const Zonotope S = Zonotope::stack(Zs);
    check(S.center().shape() == std::vector<int64_t>({3, n, 1}), "stack: center shape");
    check(S.generators().shape() == std::vector<int64_t>({3, n, 5}),
          "stack: padded to the most generators");
    const torch::Tensor c = to_torch(S.center()), G = to_torch(S.generators());
    for (int64_t b = 0; b < 3; ++b) {
        check(torch::allclose(c[b], to_torch(Zs[b].center())), "stack: member center");
        const int64_t m = Zs[b].generators().shape().back();
        check(torch::allclose(G[b].slice(-1, 0, m), to_torch(Zs[b].generators())),
              "stack: member generators");
        check(G[b].slice(-1, m).abs().sum().item<double>() == 0.0, "stack: padding is zero");
        const torch::Tensor d = torch::randn({n, 1}, torch::kDouble);
        check(torch::allclose(to_torch(S.support_function(from_torch(d)))[b],
                              to_torch(Zs[b].support_function(from_torch(d)))),
              "stack: member support function");
    }
}

void stack_of_nothing_is_described() {
    check(test::throws([] { Zonotope::stack({}); }), "stack: an empty list throws");
}

/// An empty list is refused; with the checks off the stack is the same.
void checks_and_their_switch() {
    check(test::raises([] { Zonotope::stack({}); }, Error::wrong_value), "stack: an empty list");
    const Zonotope S = Zonotope(Tensor({1.0, 0.0}), Tensor({{0.5, 0.0}, {0.0, 0.5}}));
    const Tensor on = Zonotope::stack({S, S}).center();
    ChecksScope off(false);
    check(test::close(Zonotope::stack({S, S}).center(), on, 0),
          "stack: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    const BackendScope backend_scope("torch");
    stack_keeps_each_member();
    stack_of_nothing_is_described();
    checks_and_their_switch();
    return test::finish("zonotope stack (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
