// is_bounded - whether a box has no infinite bound, as CORA's interval.isBounded
//
// NaN is not infinite. For a batch, true if every member is bounded.
//
// Syntax:   bool b = I.is_bounded();
// Inputs:   -
// Outputs:  b - true for an empty interval, else true if no bound is +-Inf
// See also: unbounded
//
// CORA: contSet/@interval/isBounded.m
// CORA authors: Mark Wetzlinger, Tobias Ladner
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/interval/private/priv_present.h"
#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Mask Interval::is_bounded() const {
    detail::Entry entry;
    const Tensor infinite = inf_.isinf().maximum(sup_.isinf());
    // no infinite bound, and an empty member is bounded
    const Tensor finite = priv_all_entries(infinite.ones_like() - infinite, *this);
    const Tensor present = priv_present(*this);
    return Mask(finite.maximum(present.ones_like() - present));
}

bool Interval::is_bounded_impl() const { return is_bounded().all(); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
