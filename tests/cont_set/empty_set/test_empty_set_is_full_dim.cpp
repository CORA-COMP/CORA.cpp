// test_empty_set_is_full_dim - EmptySet is_full_dim: never
//
// CORA: contSet/@emptySet/isFullDim.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   04-November-2026
// Reviewed:      ---

#include "cora/cont_set/empty_set/empty_set.h"
#include "testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void the_empty_set_has_no_interior(const std::string &b) {
    check(!EmptySet(3).is_full_dim() && !EmptySet(0).is_full_dim(), b + ": in any dimension");
    check(!EmptySet(3).is_full_dim(1e-3), b + ": with a tolerance");
    check(test::raises([] { EmptySet(3).is_full_dim(-1.0); }, Error::wrong_value),
          b + ": a negative tolerance");
}

/// With the checks off the answer is the same.
void checks_and_their_switch() {
    ChecksScope off(false);
    check(!EmptySet(2).is_full_dim(), "is_full_dim: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(the_empty_set_has_no_interior);
    checks_and_their_switch();
    return test::finish("empty_set is_full_dim");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
