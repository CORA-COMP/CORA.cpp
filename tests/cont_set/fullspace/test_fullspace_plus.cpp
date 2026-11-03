// test_fullspace_plus - fullspace plus: the sum with any set is the whole space
//
// CORA: contSet/@fullspace/plus.m

#include "cora/cont_set/fullspace/fullspace.h"
#include "cora/cont_set/registry.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::column;

namespace {

void plus_is_the_whole_space(const std::string &b) {
    const Fullspace F(2);
    const Zonotope Z(column({1.0, 2.0}), Tensor::from_data({1, 0, 0, 1}, {2, 2}));
    const Interval I(column({0.0, 1.0}), column({2.0, 3.0}));
    check(F.plus(Z).dim() == 2 && F.plus(I).dim() == 2 && F.plus(Fullspace(2)).dim() == 2,
          b + ": with a zonotope, a box and the space");
    check(F.plus(Interval(column({1.0, 1.0}), column({1.0, 1.0}))).dim() == 2,
          b + ": a point is [p, p]");
    check(plus(Z, F).is<Fullspace>() && (Set(I) + Set(F)).is<Fullspace>(),
          b + ": through the registry and the operator");
}

/// R^0, another dimension and an empty interval (an emptySet in CORA) are refused.
void checks_and_their_switch() {
    const Fullspace F(2);
    check(test::raises([&] { Fullspace(0).plus(Fullspace(0)); }, Error::not_supported), "plus: R^0");
    check(test::raises([&] { F.plus(Fullspace(3)); }, Error::dimension_mismatch), "plus: dimension");
    check(test::raises([&] { F.plus(Interval::empty(2)); }, Error::not_supported),
          "plus: an empty interval");
    ChecksScope off(false);
    check(F.plus(Fullspace(2)).dim() == 2, "plus: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend(plus_is_the_whole_space);
    checks_and_their_switch();
    return test::finish("fullspace plus");
}
