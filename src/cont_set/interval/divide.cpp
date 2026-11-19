// divide - a box divided by a number, as CORA's interval.mrdivide with a scalar denominator
//
// Error::out_of_domain if a quotient is NaN (0/0 or Inf/Inf).
//
// Syntax:   Interval R = I.divide(d);   Interval R = I / d;
// Inputs:   d - the denominator, a number
// Outputs:  R - [min(inf/d, sup/d), max(inf/d, sup/d)]; (-Inf, Inf) for d = 0
// See also: times, Tensor::divide
//
// CORA: contSet/@interval/mrdivide.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::divide(double d) const {
    detail::Entry entry;
    const double kInf = std::numeric_limits<double>::infinity();
    if (d == 0)
        return with(inf_.full_like(inf_.shape(), -kInf), sup_.full_like(sup_.shape(), kInf));
    const Tensor q1 = inf_ / d, q2 = sup_ / d;
    CORA_CHECK(q1.isnan().maximum(q2.isnan()).max().item() == 0, Error::out_of_domain, [] {
        return std::string("divide: a quotient is NaN (validDomain inf ~= 0 && sup ~= 0)");
    });
    return with(q1.minimum(q2), q1.maximum(q2));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
