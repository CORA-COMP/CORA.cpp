// dirac - the Dirac delta (or a derivative) of a box, as CORA's interval.dirac
//
// Syntax:   Interval R = I.dirac();   Interval R = I.dirac(n);
// Inputs:   n - order of the derivative, n >= 0 (default 0)
// Outputs:  R - elementwise [0, 0] where 0 is not in the box; where it is (s = sign(inf) +
//           sign(sup), so |s| <= 1): n = 0 gives [0, Inf]; n = 1 gives [-Inf, Inf] for s = 0,
//           [-Inf, 0] if 0 is the lower end (s = 1) and [0, Inf] if it is the upper end (s = -1);
//           n >= 2 gives [-Inf, Inf]
// See also: abs
//
// CORA: contSet/@interval/dirac.m

#include "cora/cont_set/interval/interval.h"
#include "global/format.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// `value` where the mask is 1, else 0.
Tensor aux_where(const Tensor &mask, double value) {
    return Tensor::where(mask, mask.full_like(mask.shape(), value), mask.zeros_like());
}

} // namespace


// ===========================================  MAIN  =========================================== //

Interval Interval::dirac(int64_t n) const {
    detail::Entry entry;
    CORA_CHECK(n >= 0, Error::wrong_value, [&] {
        return "dirac: the order of the derivative must be nonnegative, got " + std::to_string(n);
    });
    const double kInf = std::numeric_limits<double>::infinity();
    const Tensor s = inf_.sign() + sup_.sign();
    const Tensor zero = s.zeros_like();
    const Tensor contains_zero = s.abs().less_equal(s.ones_like()), mid = s.equal(zero);
    const Tensor lower_end = s.equal(s.ones_like()), upper_end = s.equal(-s.ones_like());
    if (n == 0) return {zero, aux_where(contains_zero, kInf)};
    if (n == 1)
        return {aux_where(mid.maximum(lower_end), -kInf), aux_where(mid.maximum(upper_end), kInf)};
    return {aux_where(contains_zero, -kInf), aux_where(contains_zero, kInf)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
