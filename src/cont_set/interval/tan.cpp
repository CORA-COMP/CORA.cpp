// tan - the tangent of a box, as CORA's interval.tan
//
// Syntax:   Interval R = I.tan();
// Inputs:   -
// Outputs:  R - [tan(inf), tan(sup)] where the entry is narrower than pi and tan(inf) <= tan(sup)
//           (no pole inside), else (-Inf, Inf); nothing rounds outward
// See also: tanh, sin
//
// CORA: contSet/@interval/tan.m

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
