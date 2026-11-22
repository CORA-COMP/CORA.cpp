// norm - the norm of the empty set, as CORA's contSet.norm returns it
//
// CORA has no emptySet.norm_; its norm catches the failure and returns -Inf for the empty set.
//
// Syntax:   Tensor v = static_cast<const ContSet &>(E).norm();
// Inputs:   type, method - as ContSet::norm; neither changes the value
// Outputs:  v - -Inf, a number (): the supremum of the norms over no point
// See also: radius, Interval::norm
//
// CORA: contSet/@contSet/norm.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/empty_set/empty_set.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

CORA_FROM_NUMBERS Tensor EmptySet::norm_impl(Tensor::Norm, const std::string &) const {
    return Tensor::zeros(Shape{}).full_like(Shape{}, -std::numeric_limits<double>::infinity());
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
