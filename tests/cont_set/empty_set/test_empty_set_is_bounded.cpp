// test_empty_set_is_bounded - EmptySet is_bounded: always
//
// CORA: contSet/@emptySet/isBounded.m

#include "cora/cont_set/empty_set/empty_set.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

void the_empty_set_is_bounded(const std::string &b) {
    check(EmptySet(3).is_bounded() && EmptySet(0).is_bounded(), b + ": in any dimension");
}

/// is_bounded takes no input to refuse; with the checks off the answer is the same.
void checks_and_their_switch() {
    ChecksScope off(false);
    check(EmptySet(2).is_bounded(), "is_bounded: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend(the_empty_set_is_bounded);
    checks_and_their_switch();
    return test::finish("empty_set is_bounded");
}
