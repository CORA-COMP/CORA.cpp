// reduce - the order reduction of a box, as CORA's interval.reduce
//
// Syntax:   Interval R = I.reduce();   Interval R = I.reduce(order);
// Inputs:   order - the zonotope order to reduce to (at least 1); a box has nothing to reduce
// Outputs:  R - the box itself: CORA's reduce of an interval returns it unchanged for every method
//           and order
// See also: Zonotope::reduce
//
// CORA: contSet/@interval/reduce.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::reduce(int order) const {
    detail::Entry entry;
    CORA_CHECK(order >= 1, Error::wrong_value, [&] {
        return "reduce: the zonotope order must be at least 1, got " + std::to_string(order);
    });
    return *this;
}

Set Interval::reduce_impl(int order) const { return reduce(order); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
