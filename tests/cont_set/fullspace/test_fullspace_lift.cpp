// test_fullspace_lift - fullspace lift: the whole space in more dimensions
//
// CORA: contSet/@fullspace/lift_.m

#include "cora/cont_set/fullspace/fullspace.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

void lift_is_the_space_of_the_new_dimension() {
    check(Fullspace(2).lift(5, {4, 0}).dim() == 5, "lift: R^5");
    check(Fullspace(2).lift(2, {1, 0}).dim() == 2, "lift: N = n");
    check(Fullspace(2).lift(3, {1, 1}).dim() == 3, "lift: a repeated position, as the exact lift");
    check(Fullspace(0).lift(3, {}).dim() == 3, "lift: from R^0");
}

/// A space that is too small, a wrong number or a range of positions are refused.
void checks_and_their_switch() {
    const Fullspace F(2);
    check(test::raises([&] { F.lift(1, {0, 0}); }, Error::wrong_value), "lift: N below n");
    check(test::raises([&] { F.lift(3, {0}); }, Error::wrong_value), "lift: one position for two");
    check(test::raises([&] { F.lift(3, {0, 3}); }, Error::wrong_value), "lift: position 3 of 3");
    check(test::raises([&] { F.lift(3, {-1, 0}); }, Error::wrong_value), "lift: a negative position");
    ChecksScope off(false);
    check(F.lift(3, {2, 0}).dim() == 3, "lift: the same with the checks off");
}

} // namespace

int main() {
    lift_is_the_space_of_the_new_dimension();
    checks_and_their_switch();
    return test::finish("fullspace lift");
}
