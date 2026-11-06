// supremum - the upper bound of a box, as CORA's interval.supremum
//
// Syntax:   Tensor u = I.supremum();
// Inputs:   -
// Outputs:  u - sup, of the shape of the bounds
// See also: Interval::sup, infimum
//
// CORA: contSet/@interval/supremum.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Interval::supremum() const {
    detail::Entry entry;
    return sup_;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
