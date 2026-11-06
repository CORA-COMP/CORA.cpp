// test_fullspace_compact - Fullspace::compact_impl: the whole space is returned as it is
//
// CORA: contSet/@contSet/compact.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/fullspace/fullspace.h"
#include "cora/cont_set/set.h"
#include "testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    const Fullspace S(3);
    // CORA returns before it reads the method and tol: any of them is accepted
    const Set R = static_cast<const ContSet &>(S).compact("anything", -1.0);
    check(R.is<Fullspace>() && R->dim() == S.dim(), "Fullspace: the same class and dimension");
    ChecksScope off(false);
    check(static_cast<const ContSet &>(S).compact().is<Fullspace>(), "the checks off");
    return test::finish("fullspace compact");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
