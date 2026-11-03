// test_fullspace_inf - fullspace inf: the whole space of n dimensions, and the constructor
//
// CORA: contSet/@fullspace/Inf.m
// CORA: contSet/@fullspace/dim.m
// CORA: contSet/@fullspace/fullspace.m

#include "cora/cont_set/fullspace/fullspace.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

void inf_and_the_constructor_hold_the_dimension() {
    check(Fullspace::inf(3).dim() == 3 && Fullspace(3).dim() == 3, "inf: the dimension is n");
    check(Fullspace::inf(0).dim() == 0 && Fullspace(0).dim() == 0, "inf: R^0 is allowed");
    check(Fullspace(2).class_id() == ClassId::fullspace, "the class id");
    check(Fullspace(2).display() == "fullspace\n- dimension: 2\n", "display names the dimension");
}

/// A negative dimension is refused, in both; with the checks off the call goes through.
void checks_and_their_switch() {
    check(test::raises([] { Fullspace::inf(-1); }, Error::wrong_value), "inf: a negative n");
    check(test::raises([] { Fullspace(-2); }, Error::wrong_input_in_constructor),
          "Fullspace: a negative n");
    ChecksScope off(false);
    check(Fullspace::inf(4).dim() == 4, "inf: the same with the checks off");
}

} // namespace

int main() {
    inf_and_the_constructor_hold_the_dimension();
    checks_and_their_switch();
    return test::finish("fullspace inf");
}
