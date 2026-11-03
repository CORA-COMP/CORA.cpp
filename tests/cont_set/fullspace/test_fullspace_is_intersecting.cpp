// test_fullspace_is_intersecting - fullspace is_intersecting: the whole space meets every nonempty set
//
// CORA: contSet/@fullspace/isIntersecting_.m

#include "cora/cont_set/empty_set/empty_set.h"
#include "cora/cont_set/fullspace/fullspace.h"
#include "cora/cont_set/registry.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::column;

namespace {

void is_intersecting_unless_empty(const std::string &b) {
    const Fullspace F(2);
    const Zonotope Z(column({1.0, 2.0}), Tensor::from_data({1, 0, 0, 1}, {2, 2}));
    check(F.is_intersecting(Z) && F.is_intersecting(Fullspace(2)), b + ": a zonotope, the space");
    check(F.is_intersecting(Interval(column({3.0, 3.0}), column({3.0, 3.0}))), b + ": a point");
    check(!F.is_intersecting(Interval::empty(2)), b + ": the empty interval");
    check(!F.is_intersecting(EmptySet(2)), b + ": the empty set");
    check(is_intersecting(F, Z) && is_intersecting(Z, F), b + ": through the registry");
}

/// R^0, another dimension and a negative tol are refused.
void checks_and_their_switch() {
    const Fullspace F(2);
    check(test::raises([] { Fullspace(0).is_intersecting(Fullspace(0)); }, Error::not_supported),
          "is_intersecting: R^0");
    check(test::raises([&] { F.is_intersecting(Fullspace(3)); }, Error::dimension_mismatch),
          "is_intersecting: dimension");
    check(test::raises([&] { F.is_intersecting(Fullspace(2), -1.0); }, Error::wrong_value),
          "is_intersecting: negative tol");
    ChecksScope off(false);
    check(F.is_intersecting(Fullspace(2)), "is_intersecting: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend(is_intersecting_unless_empty);
    checks_and_their_switch();
    return test::finish("fullspace is_intersecting");
}
