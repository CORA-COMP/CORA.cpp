// test_empty_set_complement - EmptySet complement: the whole space
//
// CORA: contSet/@emptySet/not.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/empty_set/empty_set.h"
#include "cora/cont_set/fullspace/fullspace.h"
#include "testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void the_complement_is_the_whole_space(const std::string &b) {
    check(EmptySet(3).complement().dim() == 3 && EmptySet(0).complement().dim() == 0,
          b + ": the dimension is kept");
    check(Fullspace(2).complement().complement().dim() == 2, b + ": twice gives the set back");
    check(EmptySet(2).complement().complement().isequal(EmptySet(2)), b + ": the empty set again");
}

/// complement takes no input to refuse; with the checks off the answer is the same.
void checks_and_their_switch() {
    ChecksScope off(false);
    check(EmptySet(2).complement().dim() == 2, "complement: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(the_complement_is_the_whole_space);
    checks_and_their_switch();
    return test::finish("empty_set complement");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
