// test_fullspace_is_bounded - fullspace is_bounded: only R^0 is bounded
//
// CORA: contSet/@fullspace/isBounded.m

#include "cora/cont_set/fullspace/fullspace.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

void is_bounded_only_in_r0() {
    check(!Fullspace(1).is_bounded() && !Fullspace(5).is_bounded(), "is_bounded: R^n is not");
    check(Fullspace(0).is_bounded(), "is_bounded: R^0 is");
}

/// is_bounded has no input to refuse; with the checks off the answer is the same.
void checks_and_their_switch() {
    ChecksScope off(false);
    check(!Fullspace(2).is_bounded(), "is_bounded: the same with the checks off");
}

} // namespace

int main() {
    is_bounded_only_in_r0();
    checks_and_their_switch();
    return test::finish("fullspace is_bounded");
}
