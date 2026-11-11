// compact - a polynomial zonotope without zero generators and unused factors, as CORA's compact_
//
// The constructor already merges equal exponents and drops zero generators; compact also removes
// the factors that no generator uses (CORA's methods "states" and "exponentMatrix"), and moves
// the generators of exponent 0 to the center.
//
// Syntax:   S = pZ.compact();
// Inputs:   -
// Outputs:  S - the same set in canonical form
// See also: reduce
//
// CORA: contSet/@polyZonotope/compact_.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"
#include "cora/global/rng.h"
#include "global/format.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

PolyZonotope PolyZonotope::compact() const {
    detail::Entry entry;
    PolyZonotope folded = priv_fold_constants(*this);
    Exponents E = folded.exponents();
    std::vector<int64_t> id = folded.ids();
    priv_drop_unused_ids(E, id);
    return {folded.center(), folded.generators(), folded.independent_generators(), std::move(E),
            std::move(id)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
