// test_fullspace_plus - fullspace plus: the sum with any set is the whole space, with an empty set empty
//
// CORA: contSet/@fullspace/plus.m

#include "cora/cont_set/empty_set/empty_set.h"
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
    check(F.plus(Z)->dim() == 2 && F.plus(I)->dim() == 2 && F.plus(Fullspace(2))->dim() == 2,
          b + ": with a zonotope, a box and the space");
    check(F.plus(Interval(column({1.0, 1.0}), column({1.0, 1.0})))->dim() == 2,
          b + ": a point is [p, p]");
    check(plus(Z, F).is<Fullspace>() && (Set(I) + Set(F)).is<Fullspace>(),
          b + ": through the registry and the operator");
    check(F.plus(Fullspace(2)).is<Fullspace>(), b + ": the result is a Fullspace");
}

/// An empty set makes the sum empty, whatever its class (CORA hands the sum to the empty set).
void the_sum_with_an_empty_set_is_empty(const std::string &b) {
    const Fullspace F(2);
    check(F.plus(EmptySet(2)).is<EmptySet>() && F.plus(EmptySet(2)).as<EmptySet>().dim() == 2,
          b + ": with an EmptySet");
    check(F.plus(Interval::empty(2)).is<EmptySet>(), b + ": with an empty interval");
    check(Fullspace(0).plus(EmptySet(0)).is<EmptySet>(), b + ": in R^0 with an EmptySet");
    check(plus(F, EmptySet(2)).is<EmptySet>() && plus(EmptySet(2), F).is<EmptySet>(),
          b + ": through the registry, both orders");
}

/// R^0 and another dimension are refused.
void checks_and_their_switch() {
    const Fullspace F(2);
    check(test::raises([&] { Fullspace(0).plus(Fullspace(0)); }, Error::not_supported), "plus: R^0");
    check(test::raises([&] { F.plus(Fullspace(3)); }, Error::dimension_mismatch), "plus: dimension");
    check(test::raises([&] { F.plus(EmptySet(3)); }, Error::dimension_mismatch),
          "plus: an empty set of another dimension");
    ChecksScope off(false);
    check(F.plus(Fullspace(2))->dim() == 2, "plus: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend(plus_is_the_whole_space);
    test::for_each_backend(the_sum_with_an_empty_set_is_empty);
    checks_and_their_switch();
    return test::finish("fullspace plus");
}
