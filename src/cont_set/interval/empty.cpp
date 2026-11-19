// empty - the empty interval, as CORA's interval.empty
//
// Syntax:   Interval I = Interval::empty(n);
// Inputs:   n - dimension, n >= 0
// Outputs:  I - bounds of the shape (n), flagged empty in empty_mask(); is_empty() is true
// See also: unbounded, Interval::is_empty
//
// CORA: contSet/@interval/empty.m
// CORA authors: Mark Wetzlinger
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

CORA_FROM_NUMBERS Interval Interval::empty(int64_t n) {
    detail::Entry entry;
    CORA_CHECK(n >= 0, Error::wrong_value, [&] {
        return "empty: the dimension must be nonnegative, got " + std::to_string(n);
    });
    // the bounds keep the dimension; the mask says the set is empty
    const Tensor zero = Tensor::zeros({n});
    return {zero, zero, 1, EmptyMask::all_members()};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
