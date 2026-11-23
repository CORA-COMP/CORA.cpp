// empty - the empty polynomial zonotope, as CORA's polyZonotope.empty
//
// CORA's empty set has a center without columns. A polynomial zonotope has no empty member here
// (CORALean: its wrapper OptEmpty holds the empty set), so the call is refused.
//
// Syntax:   S = PolyZonotope::empty(n);
// Inputs:   n - dimension, n >= 0 (default 0 in CORA)
// Outputs:  S - Error::not_supported
// See also: origin
//
// CORA: contSet/@polyZonotope/empty.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

CORA_FROM_NUMBERS PolyZonotope PolyZonotope::empty(int64_t n) {
    detail::Entry entry;
    CORA_CHECK(n >= 0, Error::wrong_value, [&] {
        return "empty: the dimension must be nonnegative, got " + std::to_string(n);
    });
    return detail::refuse_conversion<PolyZonotope>(
        "PolyZonotope::empty: a polynomial zonotope has no empty member (CORA's zeros(n, 0) "
        "center); use EmptySet");
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
