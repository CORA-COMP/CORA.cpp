// no_indep - a polynomial zonotope without its independent generators, as CORA's noIndep
//
// The dependent part over the factors alone: a subset of the set.
//
// Syntax:   S = pZ.no_indep();
// Inputs:   -
// Outputs:  S - the set with GI of no rows
// See also: only_id
//
// CORA: contSet/@polyZonotope/noIndep.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

PolyZonotope PolyZonotope::no_indep() const {
    detail::Entry entry;
    return {c_, G_, GI_.zeros_like({0, dim()}), E_, id_, detail::Verbatim{}};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
