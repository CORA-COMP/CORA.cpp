// times - the product of Taylor models, as CORA's taylm.times and mtimes
//
// (p + r)(q + s) = pq + p s + q r + r s: the terms of pq up to max_order stay, the higher ones,
// p s and q r (p, q bounded by the monomial bounds) and r s go to the remainder.
//
// Syntax:   T = A.times(B);  T = A.times(s);
// Inputs:   B - a model of the variables of A, or a number s
// Outputs:  T - the model
// See also: plus, power
//
// CORA: contSet/@taylm/times.m
// CORA: contSet/@taylm/mtimes.m
// CORA authors: Dmitry Grebenyuk, Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/taylm/taylm.h"
#include "global/format.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using Bounds = std::array<double, 2>;

/// The product of two intervals: the hull of the corner products (0 * Inf is 0).
Bounds aux_product(const Bounds &a, const Bounds &b) {
    double lo = 0, hi = 0;
    bool first = true;
    for (const double x : a)
        for (const double y : b) {
            const double v = (x == 0.0 || y == 0.0) ? 0.0 : x * y;
            lo = first ? v : std::min(lo, v);
            hi = first ? v : std::max(hi, v);
            first = false;
        }
    return {lo, hi};
}

/// The degree of a monomial.
int aux_degree(const Taylm::Monomial &m) {
    int d = 0;
    for (const int e : m) d += e;
    return d;
}

/// The range of the monomial over [-1, 1]^k: 1 for the constant, [0, 1] if all exponents are even,
/// else [-1, 1].
Bounds aux_monomial_range(const Taylm::Monomial &m) {
    // the constant monomial is the number 1
    if (aux_degree(m) == 0) return {1.0, 1.0};
    for (const int e : m)
        if (e % 2 == 1) return {-1.0, 1.0};
    return {0.0, 1.0};
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Products ----------------------------------------------------------------------------------------

Taylm Taylm::times(const Taylm &S) const {
    detail::Entry entry;
    CORA_CHECK(vars_ == S.vars_, Error::dimension_mismatch, [&] {
        return "times: the models have " + std::to_string(vars_) + " and " +
               std::to_string(S.vars_) + " variables";
    });
    Taylm out(vars_, max_order_);
    // the cross terms of the remainders with the bounds of the other polynomial part
    const Bounds p = polynomial_bounds(), q = S.polynomial_bounds();
    const Bounds r = {rem_lo_, rem_hi_}, s = {S.rem_lo_, S.rem_hi_};
    Bounds rest = aux_product(p, s);
    for (const Bounds &b : {aux_product(q, r), aux_product(r, s)}) {
        rest[0] += b[0];
        rest[1] += b[1];
    }
    // the polynomial product, with the terms above max_order bounded and moved to the remainder
    for (const auto &[m1, c1] : terms_)
        for (const auto &[m2, c2] : S.terms_) {
            Monomial m(m1);
            for (std::size_t i = 0; i < m.size(); ++i) m[i] += m2[i];
            if (aux_degree(m) <= max_order_) {
                out.terms_[m] += c1 * c2;
            } else {
                const Bounds b = aux_product(aux_monomial_range(m), {c1 * c2, c1 * c2});
                rest[0] += b[0];
                rest[1] += b[1];
            }
        }
    out.rem_lo_ = rest[0];
    out.rem_hi_ = rest[1];
    return out;
}

Taylm Taylm::times(double s) const {
    detail::Entry entry;
    Taylm out = *this;
    for (auto &term : out.terms_) term.second *= s;
    // s [lo, hi] is [s lo, s hi] or its swap
    const Bounds b = aux_product({rem_lo_, rem_hi_}, {s, s});
    out.rem_lo_ = b[0];
    out.rem_hi_ = b[1];
    return out;
}

// The bound of the polynomial part ----------------------------------------------------------------

std::array<double, 2> Taylm::polynomial_bounds() const {
    detail::Entry entry;
    Bounds sum = {0.0, 0.0};
    for (const auto &[m, c] : terms_) {
        const Bounds b = aux_product(aux_monomial_range(m), {c, c});
        sum[0] += b[0];
        sum[1] += b[1];
    }
    return sum;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
