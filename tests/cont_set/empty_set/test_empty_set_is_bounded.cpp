// test_empty_set_is_bounded - EmptySet is_bounded: always
//
// CORA: contSet/@emptySet/isBounded.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/empty_set/empty_set.h"
#include "testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

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


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(the_empty_set_is_bounded);
    checks_and_their_switch();
    return test::finish("empty_set is_bounded");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
