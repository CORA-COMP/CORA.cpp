// is_bounded - whether a box has no infinite bound, as CORA's interval.isBounded
//
// Syntax:   bool b = I.is_bounded();
// Inputs:   -
// Outputs:  b - true for an empty interval, else true if no bound is +-Inf (NaN is not infinite);
//           for a batch, true if every member is bounded
// See also: unbounded
//
// CORA: contSet/@interval/isBounded.m

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool Interval::is_bounded() const {
    detail::Entry entry;
    if (is_empty()) return true;
    return inf_.isinf().maximum(sup_.isinf()).max().item() == 0;
}

bool Interval::is_bounded_impl() const { return is_bounded(); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
