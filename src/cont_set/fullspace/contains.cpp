// contains - whether the whole space contains a set, as CORA's fullspace.contains_
//
// Syntax:   bool b = F.contains(S);   bool b = F.contains(S, tol);
// Inputs:   S - a set of the dimension of F (points p: the interval [p, p])
//           tol - checked and otherwise unused; default: the repository tolerance
// Outputs:  b - true; Error::not_supported in R^0, as CORA
// See also: Interval::contains
//
// CORA: contSet/@fullspace/contains_.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/fullspace/fullspace.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool Fullspace::contains(const ContSet &S, Tol tol) const {
    detail::Entry entry;
    static_cast<void>(resolve_tol(entry, tol, "contains"));
    CORA_CHECK(S.dim() == n_, Error::dimension_mismatch, [&] {
        return "contains: the sets differ in dimension, " + std::to_string(n_) + " and " +
               std::to_string(S.dim());
    });
    CORA_CHECK(n_ > 0, Error::not_supported, [] {
        return std::string("contains: the containment check for R^0 is not supported");
    });
    return true;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
