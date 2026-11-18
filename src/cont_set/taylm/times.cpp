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
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       11-November-2026
// Last update:   17-November-2026
// Reviewed:      ---

#include "cora/cont_set/taylm/taylm.h"
#include "global/format.h"
#include "global/scalar_interval.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using Bounds = detail::ScalarInterval;

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
    const auto [p_lo, p_hi] = polynomial_bounds();
    const auto [q_lo, q_hi] = S.polynomial_bounds();
    const Bounds p = {p_lo, p_hi}, q = {q_lo, q_hi};
    const Bounds r = {rem_lo_, rem_hi_}, s = {S.rem_lo_, S.rem_hi_};
    Bounds rest = detail::times_zero_inf(p, s);
    for (const Bounds &b : {detail::times_zero_inf(q, r), detail::times_zero_inf(r, s)})
        rest = rest + b;
    // the polynomial product, with the terms above max_order bounded and moved to the remainder
    for (const auto &[m1, c1] : terms_)
        for (const auto &[m2, c2] : S.terms_) {
            Monomial m(m1);
            for (std::size_t i = 0; i < m.size(); ++i) m[i] += m2[i];
            if (aux_degree(m) <= max_order_) {
                out.terms_[m] += c1 * c2;
            } else {
                rest = rest + detail::times_zero_inf(aux_monomial_range(m), {c1 * c2, c1 * c2});
            }
        }
    out.rem_lo_ = rest.lo;
    out.rem_hi_ = rest.hi;
    return out;
}

Taylm Taylm::times(double s) const {
    detail::Entry entry;
    Taylm out = *this;
    for (auto &term : out.terms_) term.second *= s;
    // s [lo, hi] is [s lo, s hi] or its swap
    const Bounds b = detail::times_zero_inf({rem_lo_, rem_hi_}, {s, s});
    out.rem_lo_ = b.lo;
    out.rem_hi_ = b.hi;
    return out;
}

// The bound of the polynomial part ----------------------------------------------------------------

std::array<double, 2> Taylm::polynomial_bounds() const {
    detail::Entry entry;
    Bounds sum = {0.0, 0.0};
    for (const auto &[m, c] : terms_) {
        sum = sum + detail::times_zero_inf(aux_monomial_range(m), {c, c});
    }
    return {sum.lo, sum.hi};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
