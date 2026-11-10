// interval_tight - the box of a constrained zonotope that keeps the constraints, in floating point
//
// CORA's interval solves two linear programs per coordinate. Here the factors are bounded by
// `passes` passes of Fourier-Motzkin contraction (Koller et al., ICLR 2026, Prop. 3) on the rows of
// the constraints from the unit box: each row contracts the box of the previous pass alone, and the
// contractions intersect. The box is c + G J, with J the bounds of the factors, plus the rounding
// error. Every operation is one rounded step in CORALean's order, so the result is bit-identical
// to its `ConZonotope.intervalTight`; it is never tighter than CORA's exact ranges.
//
// Syntax:   Interval I = cz.interval_tight(passes);   Interval I = cut.interval_tight(passes);
// Inputs:   passes - the number of contraction passes (0: the box of the zonotope)
// Outputs:  I - a box that holds the set; empty if a pass proves the constraints infeasible
// See also: interval, cut, approximate_bounds
//
// CORA.cpp only: CORALean's float definition replaces CORA's linear programs
// Authors:       Claude Sonnet 5.5
// Written:       10-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_zonotope/private/priv.h"
#include "global/format.h"

#include <cstring>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using Vec = std::vector<double>;

// Floats and rows ---------------------------------------------------------------------------------

/// CORALean's `maximum` and `minimum` of two floats (a NaN is kept).
double aux_max(double a, double b) { return a <= b ? b : (b < a ? a : a + b); }
double aux_min(double a, double b) { return a <= b ? a : (b < a ? b : a + b); }

/// The rows a x <= r on the h factors, row-major.
struct Rows {
    int64_t h = 0;
    Vec a, r;
    std::size_t count() const { return r.size(); }
};

/// The values of a (rows, cols) matrix on the host; none for zero rows.
Vec aux_host(const Tensor &T) { return T.shape()[0] == 0 ? Vec{} : T.data(); }

/// The rows that absorbRounding makes of the constraints: C x <= d, then A x <= b and -A x <= -b.
/// The right-hand side is d + 0 (the slack and the error box of a float set are zero).
Rows aux_rows(const ConZonotope::Cut &cut) {
    Rows rows;
    rows.h = cut.G.shape()[1];
    const Vec C = aux_host(cut.C), d = aux_host(cut.d), A = aux_host(cut.A), b = aux_host(cut.b);
    rows.a = C;
    for (const double x : d) rows.r.push_back(x + 0.0);
    // an equality is its two halves
    rows.a.insert(rows.a.end(), A.begin(), A.end());
    for (const double x : b) rows.r.push_back(x + 0.0);
    for (const double x : A) rows.a.push_back(-x);
    for (const double x : b) rows.r.push_back(-x + 0.0);
    return rows;
}

// One pass ----------------------------------------------------------------------------------------

/// The sum of the other terms of the row at the box, rounded up: the room that a_j x_j has.
double aux_rest(const double *a, const Vec &lo, const Vec &hi, int64_t h, int64_t j) {
    double acc = 0.0;
    for (int64_t k = h; k-- > 0;) {
        const double term = k == j ? 0.0 : aux_max((-a[k]) * hi[k], (-a[k]) * lo[k]);
        acc = term + acc;
    }
    return acc;
}

/// One pass: the box [lo, hi] contracted by each row alone, the contractions intersected.
void aux_pass(const Rows &rows, Vec &lo, Vec &hi) {
    const Vec from_lo = lo, from_hi = hi;
    const int64_t h = rows.h;
    for (std::size_t i = 0; i < rows.count(); ++i) {
        const double *a = rows.a.data() + std::ptrdiff_t(i) * h;
        for (int64_t j = 0; j < h; ++j) {
            const std::size_t jj = std::size_t(j);
            // a negative coefficient bounds the factor from below, a positive one from above
            double cut_lo = from_lo[jj], cut_hi = from_hi[jj];
            if (a[j] < 0.0) {
                const double c = rows.r[i] + aux_rest(a, from_lo, from_hi, h, j);
                cut_lo = aux_max(from_lo[jj], -(c / (-a[j])));
            } else if (0.0 < a[j]) {
                const double c = rows.r[i] + aux_rest(a, from_lo, from_hi, h, j);
                cut_hi = aux_min(from_hi[jj], c / a[j]);
            }
            lo[jj] = aux_max(lo[jj], cut_lo);
            hi[jj] = aux_min(hi[jj], cut_hi);
        }
    }
}

/// Whether two vectors agree in every bit.
bool aux_same(const Vec &a, const Vec &b) {
    return a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

} // namespace


// ===========================================  MAIN  =========================================== //

Interval ConZonotope::Cut::interval_tight(int passes) const {
    detail::Entry entry;
    const int64_t n = c.shape()[0], h = G.shape()[1];
    const Rows rows = aux_rows(*this);
    // the passes from the unit box; a pass that changes nothing is the last
    Vec lo(std::size_t(h), -1.0), hi(std::size_t(h), 1.0);
    for (int k = 0; k < passes; ++k) {
        const Vec before_lo = lo, before_hi = hi;
        aux_pass(rows, lo, hi);
        if (aux_same(lo, before_lo) && aux_same(hi, before_hi)) break;
    }
    // a factor without a value: the constraints have no point
    for (std::size_t j = 0; j < lo.size(); ++j)
        if (lo[j] > hi[j]) return Interval::empty(n);
    // the box of G J, then the center and the error box (zero): [c, c] + [G J]
    Vec inf(std::size_t(n), 0.0), sup(std::size_t(n), 0.0);
    if (h > 0) {
        const Interval GJ = Interval(Tensor::like(c, lo, {h, 1}), Tensor::like(c, hi, {h, 1}))
                                .zonotope()
                                .linear_map(G)
                                .interval();
        inf = GJ.inf().data();
        sup = GJ.sup().data();
    }
    // the center is added to both ends
    const Vec center = c.data();
    for (std::size_t i = 0; i < center.size(); ++i) {
        inf[i] += center[i] + 0.0;
        sup[i] += center[i] + 0.0;
    }
    return {Tensor::like(c, inf, {n, 1}), Tensor::like(c, sup, {n, 1})};
}

Interval ConZonotope::interval_tight(int passes) const {
    detail::Entry entry;
    CORA_CHECK(c_.ndim() == 2, Error::not_supported,
               [] { return std::string("interval_tight: a single set, not a batch"); });
    CORA_CHECK(passes >= 0, Error::wrong_value, [&] {
        return "interval_tight: passes must be nonnegative, got " + std::to_string(passes);
    });
    return Cut{c_, G_, A_, b_, priv_zeros(c_, 0, G_.shape()[1]), priv_zeros(c_, 0, 1)}
        .interval_tight(passes);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
