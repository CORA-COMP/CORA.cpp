// compact - a polynomial zonotope without zero generators and unused factors, as CORA's compact_
//
// The constructor already merges equal exponents and drops zero generators; compact also removes
// the factors that no generator uses (CORA's methods "states" and "exponentMatrix"), and moves
// the generators of exponent 0 to the center.
//
// Syntax:   S = pZ.compact();   S = pZ.compact("all", tol);
// Inputs:   -
// Outputs:  S - the same set in canonical form
// See also: reduce
//
// CORA: contSet/@polyZonotope/compact_.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       11-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"
#include "cora/global/rng.h"
#include "global/format.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

PolyZonotope PolyZonotope::compact(const std::optional<std::string> &method, Tol tol) const {
    detail::Entry entry;
    static_cast<void>(resolve_tol(entry, tol, c_.dtype(), "compact"));
    CORA_CHECK(!method || *method == "all", Error::no_specific_alg, [&] {
        return "compact: a polynomial zonotope has the method 'all' only, got '" + *method + "'";
    });
    PolyZonotope folded = priv_fold_constants(*this);
    Exponents E = folded.exponents();
    std::vector<int64_t> id = folded.ids();
    priv_drop_unused_ids(E, id);
    return {folded.center(), folded.generators(), folded.independent_generators(), std::move(E),
            std::move(id)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
