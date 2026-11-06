// cosh - the hyperbolic cosine of a box, as CORA's interval.cosh
//
// Syntax:   Interval R = I.cosh();
// Inputs:   -
// Outputs:  R - [cosh(sup), cosh(inf)] if sup <= 0, [cosh(inf), cosh(sup)] if inf >= 0, else
//           [1, cosh(max(|inf|, |sup|))]
// See also: cos, exp
//
// CORA: contSet/@interval/cosh.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::cosh() const {
    detail::Entry entry;
    const Tensor left = sup_.less_equal(sup_.zeros_like()),
                 right = inf_.greater_equal(inf_.zeros_like());
    const Tensor at_inf = inf_.cosh(), at_sup = sup_.cosh();
    const Tensor lower =
        Tensor::where(left, at_sup, Tensor::where(right, at_inf, inf_.ones_like()));
    const Tensor larger = inf_.abs().maximum(sup_.abs()).cosh();
    const Tensor upper = Tensor::where(left, at_inf, Tensor::where(right, at_sup, larger));
    return {lower, upper};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
