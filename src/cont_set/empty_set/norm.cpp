// norm - the norm of the empty set, as CORA's contSet.norm returns it
//
// Syntax:   Tensor v = static_cast<const ContSet &>(E).norm();
// Inputs:   type, mode - as ContSet::norm; neither changes the value
// Outputs:  v - -Inf (1, 1): the supremum of the norms over no point. CORA has no emptySet.norm_;
//           its norm catches the failure and returns -Inf for the empty set
// See also: radius, Interval::norm
//
// CORA: contSet/@contSet/norm.m

#include "cora/cont_set/empty_set/empty_set.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

CORA_FROM_NUMBERS Tensor EmptySet::norm_impl(Tensor::Norm, const std::string &) const {
    return Tensor::zeros({1, 1}).full_like({1, 1}, -std::numeric_limits<double>::infinity());
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
