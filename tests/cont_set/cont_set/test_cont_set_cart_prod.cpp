// test_cont_set_cart_prod - the Cartesian product of two sets: any dimensions; no pair yet (noops)
// implementation raises noops

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/registry.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

/// Sets of different dimensions are fine for a product; no pair implements it yet.
void an_unknown_pair_is_noops() {
    const Zonotope Z = Zonotope(Tensor({1.0, 0.0}), Tensor({{0.5, 0.0}, {0.0, 0.5}})),
                   Z3 = Zonotope(Tensor({1.0, 0.0, 0.0}),
                                 Tensor({{0.5, 0.0}, {0.0, 0.5}, {0.0, 0.0}}));
    check(test::raises([&] { cart_prod(Z, Z3); }, Error::noops), "dimensions 2 and 3: noops");
    ChecksScope off(false);
    check(test::raises([&] { cart_prod(Z3, Z); }, Error::noops), "the checks off");
}

} // namespace

int main() {
    an_unknown_pair_is_noops();
    return test::finish("cont_set cart_prod");
}
