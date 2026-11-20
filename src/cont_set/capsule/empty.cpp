// empty - the empty capsule of n dimensions, as CORA's capsule.empty
//
// CORA's empty capsule has a center without columns and no radius. Here it is flagged in
// empty_mask(), as every set; is_empty() is true when every member of a batch is flagged.
//
// Syntax:   Capsule C = Capsule::empty(n);   bool none = C.is_empty();
// Inputs:   n - dimension, n >= 0 (default 0)
// Outputs:  C - the origin with the radius 0, flagged empty
// See also: origin, Capsule::is_empty
//
// CORA: contSet/@capsule/empty.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       06-November-2026
// Last update:   20-November-2026
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
    return {Unchecked{}, Tensor::zeros({n, 1}), Tensor::zeros({n, 1}), Tensor::zeros({1, 1}),
            EmptyMask::all_members()};
}

bool Capsule::is_empty() const { return empty_.all(); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
