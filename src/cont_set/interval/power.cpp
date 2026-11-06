// power - the elementwise power of a box, as CORA's interval.power
//
// For an integer e >= 0 the hull of inf^e and sup^e, with lower bound 0 for an even e > 0 over a
// box with inf < 0 < sup; the powers are products of e factors (up to numerics::kPowerProduct),
// in the order of CORALean. A negative e is the power -e of 1 ./ I. Other e: [inf^e, sup^e]. A
// box exponent takes the hull of the four corners. Error::out_of_domain if a bound is NaN (a
// negative base with a non-integer exponent).
//
// Syntax:   Interval R = I.power(e);   Interval R = I.power(E);   Interval R = I.power(E2);
//           Interval R = Interval::power(b, E2);
// Inputs:   e - a number
//           E - numbers of the shape of the bounds (or one box with a matrix of exponents)
//           E2 - a box of exponents
//           b - numbers, the base
// Outputs:  R - the image of x^e
// See also: rdivide, matrix_power
//
// CORA: contSet/@interval/power.m
// CORA authors: Dmitry Grebenyuk
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/interval/private/priv_broadcastable.h"
#include "cont_set/interval/private/priv_check_nan.h"
#include "cora/cont_set/interval/interval.h"
#include "cora/global/tensor/tolerance.h"
#include "global/format.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using Bounds = std::pair<Tensor, Tensor>;

/// A mask of x (1 where it holds): x compared with the number s.
Tensor aux_compare(const Tensor &x, Tensor::Binary op, double s) { return x.scalar_op(op, s); }

/// The number s with the shape of x.
Tensor aux_fill(const Tensor &x, double s) { return x.full_like(x.shape(), s); }

/// Whether the numbers x are one or have the shape of the matrix of the bounds.
bool aux_fits(const Tensor &x, const Tensor &bounds) {
    if (x.ndim() < 2) return false;
    const bool number = x.size(-1) == 1 && x.size(-2) == 1;
    return number || (x.size(-1) == bounds.size(-1) && x.size(-2) == bounds.size(-2));
}

/// x^k for integers k >= 0: the product 1 * x * ... * x of k factors, as CORALean's powUp, up to
/// numerics::kPowerProduct factors and above that by pow.
Tensor aux_integer_power(const Tensor &x, const Tensor &k) {
    Tensor product = x.add(k).ones_like();
    const double most = k.max().item();
    const int64_t steps =
        most > numerics::kPowerProduct ? numerics::kPowerProduct : static_cast<int64_t>(most);
    for (int64_t j = 1; j <= steps; ++j)
        product = Tensor::where(aux_compare(k, Tensor::Binary::greater_equal, double(j)),
                                product.multiply(x), product);
    if (most <= numerics::kPowerProduct) return product;
    return Tensor::where(aux_compare(k, Tensor::Binary::greater, numerics::kPowerProduct), x.pow(k),
                         product);
}

/// The bounds of [a, b]^e for numbers e: CORA's branches by the sign and the integrality of e.
Bounds aux_numeric_power(const Tensor &a, const Tensor &b, const Tensor &e) {
    // a negative exponent is the positive one of the reciprocal (1 where it is not used)
    const Tensor negative = aux_compare(e, Tensor::Binary::less, 0);
    const Tensor one = a.add(b).ones_like();
    const Interval inverse = Interval::rdivide(
        one, Interval(Tensor::where(negative, a, one), Tensor::where(negative, b, one)));
    const Tensor lo = Tensor::where(negative, inverse.inf(), a);
    const Tensor hi = Tensor::where(negative, inverse.sup(), b);
    const Tensor exponent = e.abs();
    const Tensor rounded = exponent.round();
    const Tensor integral =
        aux_compare(rounded.subtract(exponent).abs(), Tensor::Binary::less_equal, machine_eps());
    // integer exponents: the hull of the two end powers, 0 below an even power over a sign change
    const Tensor k = Tensor::where(integral, rounded, aux_fill(rounded, 0));
    const Tensor p1 = aux_integer_power(lo, k), p2 = aux_integer_power(hi, k);
    const Tensor even = ((k / 2.0).floor() * 2.0).equal(k);
    const Tensor changes_sign = aux_compare(lo, Tensor::Binary::less, 0)
                                    .multiply(aux_compare(hi, Tensor::Binary::greater, 0));
    const Tensor flat =
        even.multiply(aux_compare(k, Tensor::Binary::not_equal, 0)).multiply(changes_sign);
    const Tensor int_lo = Tensor::where(flat, aux_fill(p1, 0), p1.minimum(p2));
    return {Tensor::where(integral, int_lo, lo.pow(exponent)),
            Tensor::where(integral, p1.maximum(p2), hi.pow(exponent))};
}

/// The hull of the powers a^e for numbers a and the bounds of exponents (NaN for a < 0).
Bounds aux_hull_power(const std::vector<Tensor> &bases, const std::vector<Tensor> &exponents) {
    const double quiet_nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<Tensor> values;
    for (const Tensor &a : bases)
        for (const Tensor &e : exponents)
            values.push_back(Tensor::where(aux_compare(a, Tensor::Binary::less, 0),
                                           aux_fill(a.pow(e), quiet_nan), a.pow(e)));
    Tensor lo = values[0], hi = values[0];
    for (std::size_t i = 1; i < values.size(); ++i) {
        lo = lo.minimum(values[i]);
        hi = hi.maximum(values[i]);
    }
    return {lo, hi};
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Numbers as exponents ----------------------------------------------------------------------------

Interval Interval::power(double exponent) const {
    detail::Entry entry;
    return power(inf_.full_like({1, 1}, exponent));
}

Interval Interval::power(const Tensor &exponent) const {
    detail::Entry entry;
    const bool scalar_base = inf_.size(-1) == 1 && inf_.size(-2) == 1;
    CORA_CHECK(scalar_base || aux_fits(exponent, inf_), Error::dimension_mismatch, [&] {
        return "power: the exponents " + shape_text(exponent.shape()) +
               " must be one number or match the bounds " + shape_text(inf_.shape());
    });
    if (inf_.numel() == 0) return *this;
    const auto [lo, hi] = aux_numeric_power(inf_, sup_, exponent);
    priv_check_nan("power", lo, hi, "base >= 0");
    return {lo, hi};
}

// Boxes as exponents ------------------------------------------------------------------------------

Interval Interval::power(const Tensor &base, const Interval &exponent) {
    detail::Entry entry;
    // the base is one number or has the shape of the exponents
    CORA_CHECK(aux_fits(base, exponent.inf()) ||
                   (exponent.inf().size(-1) == 1 && exponent.inf().size(-2) == 1),
               Error::dimension_mismatch, [&] {
                   return "power: the base " + shape_text(base.shape()) +
                          " must be one number or match the exponents " +
                          shape_text(exponent.inf().shape());
               });
    // the hull of the powers at the two ends of the exponent
    const auto [lo, hi] = aux_hull_power({base}, {exponent.inf(), exponent.sup()});
    priv_check_nan("power", lo, hi, "base >= 0");
    return {lo, hi};
}

Interval Interval::power(const Interval &exponent) const {
    detail::Entry entry;
    CORA_CHECK(priv_broadcastable(inf_.shape(), exponent.inf().shape()), Error::dimension_mismatch,
               [&] {
                   return "power: the bounds " + shape_text(inf_.shape()) + " and the exponents " +
                          shape_text(exponent.inf().shape()) + " must have one shape";
               });
    const auto [lo, hi] = aux_hull_power({inf_, sup_}, {exponent.inf(), exponent.sup()});
    priv_check_nan("power", lo, hi, "base >= 0");
    return {lo, hi};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
