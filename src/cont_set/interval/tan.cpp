// tan - the tangent of a box, as CORA's interval.tan
//
// Else (-Inf, Inf); nothing rounds outward.
//
// Syntax:   Interval R = I.tan();
// Inputs:   -
// Outputs:  R - [tan(inf), tan(sup)] where the entry is narrower than pi and has no pole
// See also: tanh, sin
//
// CORA: contSet/@interval/tan.m
// CORA authors: Daniel Althoff, Dmitry Grebenyuk, Matthias Althoff, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"

#include <limits>
#include <numbers>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::tan() const {
    detail::Entry entry;
    constexpr double kInf = std::numeric_limits<double>::infinity();
    const Tensor lo = inf_.tan(), hi = sup_.tan();
    const Tensor no_pole = (sup_ - inf_)
                               .less(inf_.full_like(inf_.shape(), std::numbers::pi))
                               .multiply(lo.less_equal(hi));
    return {Tensor::where(no_pole, lo, inf_.full_like(inf_.shape(), -kInf)),
            Tensor::where(no_pole, hi, inf_.full_like(inf_.shape(), kInf))};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
