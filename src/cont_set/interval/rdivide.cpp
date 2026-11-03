// rdivide - the entrywise quotient of boxes and numbers, as CORA's interval.rdivide
//
// Syntax:   Interval R = I.rdivide(d);   Interval R = Interval::rdivide(n, D);
//           Interval R = I.rdivide(D);
// Inputs:   d - numbers that broadcast to the bounds;  n - numbers, one or of the shape of the
//           bounds of D;  D - the denominator, a box
// Outputs:  R - the hull of the quotients, in the shape of the broadcast operands. I ./ d is
//           [min(inf/d, sup/d), max(...)] and (-Inf, Inf) where d = 0; n ./ D is the hull of
//           n/inf and n/sup, (-Inf, Inf) where D contains 0 in its interior, and the ray
//           beyond n/inf where sup = 0 (CORA mistakes it for the opposite ray when n < 0);
//           I ./ D is I .* (1 ./ D). Error::out_of_domain where D = [0, 0] or a quotient is NaN
// See also: divide, times
//
// CORA: contSet/@interval/rdivide.m

#include "cont_set/interval/private/priv_broadcastable.h"
#include "cont_set/interval/private/priv_check_nan.h"
#include "cont_set/interval/private/priv_times.h"
#include "cora/cont_set/interval/interval.h"
#include "global/format.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

/// A mask of x (1 where it holds): x compared with the number s.
Tensor aux_compare(const Tensor &x, Tensor::Binary op, double s) { return x.scalar_op(op, s); }

/// The number s with the shape of x.
Tensor aux_fill(const Tensor &x, double s) { return x.full_like(x.shape(), s); }

/// Whether the numbers n are one or have the shape of the matrix of the bounds.
bool aux_fits(const Tensor &n, const Tensor &bounds) {
    if (n.ndim() < 2) return false;
    const bool number = n.size(-1) == 1 && n.size(-2) == 1;
    return number || (n.size(-1) == bounds.size(-1) && n.size(-2) == bounds.size(-2));
}

} // namespace


// ===========================================  MAIN  =========================================== //

Interval Interval::rdivide(const Tensor &denominator) const {
    detail::Entry entry;
    CORA_CHECK(priv_broadcastable(inf_.shape(), denominator.shape()), Error::special_error, [&] {
        return "rdivide: the input size is wrong, got the bounds " + shape_text(inf_.shape()) +
               " and the denominator " + shape_text(denominator.shape());
    });
    const Tensor q1 = inf_.divide(denominator), q2 = sup_.divide(denominator);
    const Tensor zero = aux_compare(denominator, Tensor::Binary::equal, 0);
    const Tensor lo = Tensor::where(zero, aux_fill(q1, -kInf), q1.minimum(q2));
    const Tensor hi = Tensor::where(zero, aux_fill(q1, kInf), q1.maximum(q2));
    priv_check_nan("rdivide", lo, hi, "inf ~= 0 && sup ~= 0");
    return {lo, hi};
}

Interval Interval::rdivide(const Tensor &numerator, const Interval &denominator) {
    detail::Entry entry;
    const Tensor &a = denominator.inf(), &b = denominator.sup();
    CORA_CHECK(aux_fits(numerator, a), Error::special_error, [&] {
        return "rdivide: the input size is wrong, got the numerator " +
               shape_text(numerator.shape()) + " and the denominator " + shape_text(a.shape());
    });
    // a denominator [0, 0] has no quotient (CORA's NaN, which it then overwrites)
    if (aux_compare(a, Tensor::Binary::equal, 0)
            .multiply(aux_compare(b, Tensor::Binary::equal, 0))
            .max()
            .item() != 0)
        CORA_FAIL(Error::out_of_domain, "rdivide: the denominator [0, 0] has no quotient");
    const Tensor q1 = numerator.divide(b), q2 = numerator.divide(a);
    const Tensor positive = aux_compare(numerator, Tensor::Binary::greater, 0);
    const Tensor negative = aux_compare(numerator, Tensor::Binary::less, 0);
    const Tensor zero = aux_compare(numerator, Tensor::Binary::equal, 0);
    const Tensor crossing = aux_compare(a, Tensor::Binary::less, 0)
                                .multiply(aux_compare(b, Tensor::Binary::greater, 0));
    // sup = 0: n/x runs from n/inf to the infinity of the sign of -n
    const Tensor ends_at_zero = aux_compare(b, Tensor::Binary::equal, 0);
    Tensor lo = q1.minimum(q2), hi = q1.maximum(q2);
    lo = Tensor::where(ends_at_zero, Tensor::where(positive, aux_fill(lo, -kInf), q2), lo);
    hi = Tensor::where(ends_at_zero, Tensor::where(negative, aux_fill(hi, kInf), q2), hi);
    // 0 / x is 0 off the pole
    lo = Tensor::where(zero, aux_fill(lo, 0), lo);
    hi = Tensor::where(zero, aux_fill(hi, 0), hi);
    lo = Tensor::where(crossing, aux_fill(lo, -kInf), lo);
    hi = Tensor::where(crossing, aux_fill(hi, kInf), hi);
    priv_check_nan("rdivide", lo, hi, "inf ~= 0 && sup ~= 0");
    return {lo, hi};
}

Interval Interval::rdivide(const Interval &denominator) const {
    detail::Entry entry;
    CORA_CHECK(priv_broadcastable(inf_.shape(), denominator.inf().shape()), Error::special_error,
               [&] {
                   return "rdivide: the input size is wrong, got " + shape_text(inf_.shape()) +
                          " and " + shape_text(denominator.inf().shape());
               });
    const Interval y = rdivide(denominator.inf().ones_like(), denominator);
    const auto [lo, hi] = priv_times(inf_, sup_, y.inf(), y.sup());
    priv_check_nan("rdivide", lo, hi, "inf ~= 0 && sup ~= 0");
    return {lo, hi};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
