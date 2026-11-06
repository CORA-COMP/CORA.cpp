// test_empty_set_compact - EmptySet::compact_impl: the empty set is returned as it is
//
// CORA: contSet/@contSet/compact.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/empty_set/empty_set.h"
#include "cora/cont_set/set.h"
#include "testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    const EmptySet S(3);
    // CORA returns before it reads the method and tol: any of them is accepted
    const Set R = static_cast<const ContSet &>(S).compact("anything", -1.0);
    check(R.is<EmptySet>() && R->dim() == S.dim(), "EmptySet: the same class and dimension");
    ChecksScope off(false);
    check(static_cast<const ContSet &>(S).compact().is<EmptySet>(), "the checks off");
    return test::finish("empty_set compact");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
