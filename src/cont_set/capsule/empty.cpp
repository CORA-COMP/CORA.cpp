// empty - the empty capsule of n dimensions, as CORA's capsule.empty
//
// Syntax:   Capsule C = Capsule::empty(n);   bool none = C.is_empty();
// Inputs:   n - dimension, n >= 0 (default 0)
// Outputs:  C - the origin with the radius -1, which no point is within; CORA's has a center
//           without columns and no radius. center() of it has no columns (n, 0), as in CORA.
//           is_empty() is true for a negative radius (of a batch: every member)
// See also: origin, Capsule::is_empty
//
// CORA: contSet/@capsule/empty.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/capsule/private/priv.h"
#include "cora/cont_set/capsule/capsule.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

CORA_FROM_NUMBERS Capsule Capsule::empty(int64_t n) {
    detail::Entry entry;
    CORA_CHECK(n >= 0, Error::wrong_value, [&] {
        return "empty: the dimension must be nonnegative, got " + std::to_string(n);
    });
    return {Unchecked{}, Tensor::zeros({n, 1}), Tensor::zeros({n, 1}),
            Tensor::from_data({-1.0}, {1, 1})};
}

bool Capsule::is_empty() const { return priv_all(priv_empty_mask(r_)); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
