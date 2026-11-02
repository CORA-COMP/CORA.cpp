// test_cont_set_intersect - the intersection of two sets (CORA's and): no pair yet raises noops
// implementation raises noops
//
// CORA: contSet/@contSet/and.m

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/registry.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

/// No pair implements it yet: noops, with the checks on and off; the dimensions are checked.
void an_unknown_pair_is_noops() {
    const Zonotope Z = Zonotope(Tensor({1.0, 0.0}), Tensor({{0.5, 0.0}, {0.0, 0.5}})),
                   Z3 = Zonotope(Tensor({1.0, 0.0, 0.0}),
                                 Tensor({{0.5, 0.0}, {0.0, 0.5}, {0.0, 0.0}}));
    const Interval I = Interval(test::column({0, -1}), test::column({1, 1}));
    check(test::raises([&] { intersect(Z, Z3); }, Error::dimension_mismatch), "dimensions 2 and 3");
    check(test::raises([&] { intersect(Z, I); }, Error::noops), "zonotope and interval");
    ChecksScope off(false);
    check(test::raises([&] { intersect(I, Z); }, Error::noops), "the checks off");
}

} // namespace

int main() {
    an_unknown_pair_is_noops();
    return test::finish("cont_set intersect");
}
