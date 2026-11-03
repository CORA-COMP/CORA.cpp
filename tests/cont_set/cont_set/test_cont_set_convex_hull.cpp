// test_cont_set_convex_hull - the convex hull of two sets: a pair without an implementation raises
// noops
//
// CORA: contSet/@contSet/convHull.m

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/registry.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"
#include "unregistered_set.h"

using namespace cora;
using test::check;

namespace {

/// A pair without an entry is noops, with the checks on and off; the dimensions are checked.
void an_unknown_pair_is_noops() {
    const Zonotope Z = Zonotope(Tensor({1.0, 0.0}), Tensor({{0.5, 0.0}, {0.0, 0.5}})),
                   Z3 = Zonotope(Tensor({1.0, 0.0, 0.0}),
                                 Tensor({{0.5, 0.0}, {0.0, 0.5}, {0.0, 0.0}}));
    const test::UnregisteredSet U;
    check(test::raises([&] { convex_hull(Z, Z3); }, Error::dimension_mismatch),
          "dimensions 2 and 3");
    check(test::raises([&] { convex_hull(Z, U); }, Error::noops), "zonotope and unregistered set");
    ChecksScope off(false);
    check(test::raises([&] { convex_hull(U, Z); }, Error::noops), "the checks off");
}

} // namespace

int main() {
    an_unknown_pair_is_noops();
    return test::finish("cont_set convex_hull");
}
