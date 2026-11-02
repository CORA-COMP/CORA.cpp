// test_cont_set_plus - plus of two sets: the registry picks the pair, operands in CORA's order

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/registry.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

/// Zonotope + Interval in either order is a zonotope; dimensions are checked.
void the_registry_dispatches() {
    const Zonotope Z = Zonotope(Tensor({1.0, 0.0}), Tensor({{0.5, 0.0}, {0.0, 0.5}}));
    const Interval I = Interval(test::column({0, -1}), test::column({1, 1}));
    const Set a = plus(Z, I), b = plus(I, Z);
    check(a.is<Zonotope>() && b.is<Zonotope>(), "a zonotope either way");
    check(test::close(a.get().center(), b.get().center()), "commutative");
    check(test::close((Set(Z) + Set(I)).get().center(), a.get().center()), "the operator +");
    const Zonotope Z3 =
        Zonotope(Tensor({1.0, 0.0, 0.0}), Tensor({{0.5, 0.0}, {0.0, 0.5}, {0.0, 0.0}}));
    check(test::raises([&] { plus(Z, Z3); }, Error::dimension_mismatch), "dimensions 2 and 3");
    ChecksScope off(false);
    check(test::close(plus(Z, I).get().center(), a.get().center(), 0), "the checks off");
}

} // namespace

int main() {
    the_registry_dispatches();
    return test::finish("cont_set plus");
}
