// test_fullspace_is_full_dim - fullspace is_full_dim: the whole space has an interior
//
// CORA: contSet/@fullspace/isFullDim.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   04-November-2026
// Reviewed:      ---

#include "cora/cont_set/fullspace/fullspace.h"
#include "testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void is_full_dim_always() {
    check(Fullspace(1).is_full_dim() && Fullspace(4).is_full_dim() && Fullspace(0).is_full_dim(),
          "is_full_dim: in every dimension");
    check(Fullspace(2).is_full_dim(1e-3), "is_full_dim: with a tolerance");
    check(test::raises([] { Fullspace(2).is_full_dim(-1.0); }, Error::wrong_value),
          "is_full_dim: a negative tolerance");
}

/// With the checks off the answer is the same.
void checks_and_their_switch() {
    ChecksScope off(false);
    check(Fullspace(2).is_full_dim(), "is_full_dim: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    is_full_dim_always();
    checks_and_their_switch();
    return test::finish("fullspace is_full_dim");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
