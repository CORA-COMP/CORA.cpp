// compact - fewer rows and generators for the same constrained zonotope, as CORA's compact_
//
// "zeros" and "all" drop the constraint rows 0 = 0 and the generators that are zero in G and in
// A; "aligned" leaves the set as it is.
//
// Syntax:   cZres = cz.compact();   cZres = cz.compact("all", tol);
// Inputs:   method - "zeros" (default), "all" or "aligned"
//           tol - checked only (CORA's compact_ drops exact zeros); default: the repository one
// Outputs:  cZres - the set without them; a batch drops what every member drops
// See also: Zonotope::compact, intersect
//
// CORA: contSet/@conZonotope/compact_.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Tobias Ladner, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_zonotope/private/priv.h"
#include "cora/global/tensor/tolerance.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

ConZonotope ConZonotope::compact(const std::optional<std::string> &method_arg, Tol tol_arg) const {
    detail::Entry entry;
    const std::string method = method_arg.value_or("zeros");
    static_cast<void>(resolve_tol(entry, tol_arg, c_.dtype(), "compact"));
    CORA_CHECK(
        method == "zeros" || method == "all" || method == "aligned", Error::wrong_value, [&] {
            return "compact: the method must be 'zeros', 'all' or 'aligned', got '" + method + "'";
        });
    // 'all' is 'zeros' in CORA
    if (method == "aligned") return *this;
    return priv_compact(*this);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
