// test_empty_set_empty_set - the EmptySet constructor: the dimension, and what it refuses
//
// CORA: contSet/@emptySet/dim.m
// CORA: contSet/@emptySet/emptySet.m

#include "cora/cont_set/empty_set/empty_set.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

void the_constructor_keeps_the_dimension(const std::string &b) {
    const EmptySet E(3);
    check(E.dim() == 3 && E.class_id() == ClassId::empty_set, b + ": dimension and class");
    check(E.class_name() == "EmptySet", b + ": the class name");
    check(EmptySet(0).dim() == 0, b + ": dimension 0 is allowed");
    const EmptySet &copy(E);
    check(copy.dim() == 3, b + ": the copy constructor, as CORA's emptySet(O)");
    const Set S(E);
    check(S.is<EmptySet>() && S.as<EmptySet>().dim() == 3, b + ": as a Set");
}

/// A negative dimension is refused; with the checks off the set is the same.
void checks_and_their_switch() {
    check(test::raises([] { EmptySet(-1); }, Error::wrong_input_in_constructor),
          "EmptySet: dimension -1");
    ChecksScope off(false);
    check(EmptySet(2).dim() == 2, "EmptySet: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend(the_constructor_keeps_the_dimension);
    checks_and_their_switch();
    return test::finish("empty_set constructor");
}
