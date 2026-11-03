// test_empty_set_lift - EmptySet lift: the empty set of the larger dimension
//
// CORA: contSet/@emptySet/lift_.m

#include "cora/cont_set/empty_set/empty_set.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

void the_lifted_set_has_the_new_dimension(const std::string &b) {
    check(EmptySet(2).lift(5, {0, 3}).dim() == 5, b + ": from 2 to 5 dimensions");
    check(EmptySet(0).lift(3, {}).dim() == 3, b + ": from dimension 0");
    check(EmptySet(2).lift(2, {1, 0}).dim() == 2, b + ": a permutation");
}

/// A smaller space, a wrong number of coordinates, or a coordinate outside it are refused.
void checks_and_their_switch() {
    const EmptySet E(2);
    check(test::raises([&] { E.lift(1, {0, 0}); }, Error::wrong_value), "lift: N below dim");
    check(test::raises([&] { E.lift(4, {0}); }, Error::wrong_value), "lift: one coordinate of two");
    check(test::raises([&] { E.lift(4, {0, 4}); }, Error::wrong_value), "lift: coordinate 4 of 4");
    check(test::raises([&] { E.lift(4, {-1, 0}); }, Error::wrong_value), "lift: coordinate -1");
    ChecksScope off(false);
    check(E.lift(4, {0, 3}).dim() == 4, "lift: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend(the_lifted_set_has_the_new_dimension);
    checks_and_their_switch();
    return test::finish("empty_set lift");
}
