// test_empty_set_empty - EmptySet::empty: the empty set of n dimensions
//
// CORA: contSet/@emptySet/empty.m
// CORA authors: Mark Wetzlinger
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

void empty_makes_the_set_of_n_dimensions(const std::string &b) {
    check(EmptySet::empty(4).dim() == 4, b + ": 4 dimensions");
    check(EmptySet::empty().dim() == 0, b + ": the default is dimension 0, as CORA's nargin == 0");
}

/// A negative dimension is refused; with the checks off the result is the same.
void checks_and_their_switch() {
    check(test::raises([] { EmptySet::empty(-2); }, Error::wrong_value), "empty: dimension -2");
    ChecksScope off(false);
    check(EmptySet::empty(2).dim() == 2, "empty: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(empty_makes_the_set_of_n_dimensions);
    checks_and_their_switch();
    return test::finish("empty_set empty");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
