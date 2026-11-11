// tighten_domain - a box cut to the points that can solve f = 0, as CORA's levelSet.tightenDomain
//
// The forward-backward contractor: the expression is evaluated over the box (forward), then the
// value 0 is pushed back through every operation (backward): a + b = t gives a in t - range(b),
// a * b = t gives a in t / range(b) where range(b) excludes 0, an even power a^n = t gives the
// two roots, and so on down to the variables, whose boxes are cut. Each operation narrows only
// where the inverse is sound: sin, cos, tan, a real power other than the square root, and a
// divisor are left as they are (CORA's backward tan intersects with [-1, 1], which cuts away
// solutions, docs/cora-differences.md). A root is widened by one ulp against rounding. All
// constraints must be '=='; with several, the box is contracted by each in turn. CORA raises
// emptySet when the contraction empties the box, here the result is none. The methods split,
// taylor and linear are not ported.
//
// Syntax:   std::optional<Interval> J = ls.tighten_domain(I);
// Inputs:   I - the box (n, 1)
//           type - "forwardBackward"
// Outputs:  J - a box in I that holds every solution in I of all f_i = 0; none if there is none
// See also: contains, is_intersecting
//
// CORA: contSet/@levelSet/tightenDomain.m
// CORA: specification/syntaxTree.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger, Zhuoling Li
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/level_set/level_set.h"
#include "global/format.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Interval arithmetic on doubles ------------------------------------------------------------------

constexpr double kInf = std::numeric_limits<double>::infinity();

/// A closed interval of doubles; empty if lo > hi.
struct Iv {
    double lo, hi;
    bool empty() const { return lo > hi; }
};

Iv aux_meet(const Iv &a, const Iv &b) { return {std::max(a.lo, b.lo), std::min(a.hi, b.hi)}; }
Iv aux_whole() { return {-kInf, kInf}; }
Iv aux_neg(const Iv &a) { return {-a.hi, -a.lo}; }
Iv aux_add(const Iv &a, const Iv &b) { return {a.lo + b.lo, a.hi + b.hi}; }
Iv aux_sub(const Iv &a, const Iv &b) { return {a.lo - b.hi, a.hi - b.lo}; }

/// The hull of the corner products, a corner 0 * Inf taken as 0.
Iv aux_mul(const Iv &a, const Iv &b) {
    double lo = kInf, hi = -kInf;
    for (const double x : {a.lo, a.hi})
        for (const double y : {b.lo, b.hi}) {
            const double v = (x == 0.0 || y == 0.0) ? 0.0 : x * y;
            lo = std::min(lo, v);
            hi = std::max(hi, v);
        }
    return {lo, hi};
}

/// a / b for a divisor that excludes 0.
Iv aux_div(const Iv &a, const Iv &b) { return aux_mul(a, {1.0 / b.hi, 1.0 / b.lo}); }

bool aux_has_zero(const Iv &a) { return a.lo <= 0 && a.hi >= 0; }

/// Widens by one ulp, against the rounding of a root or a logarithm.
Iv aux_widen(const Iv &a) { return {std::nextafter(a.lo, -kInf), std::nextafter(a.hi, kInf)}; }

/// The signed n-th root of x for odd n.
double aux_root(double x, int n) {
    return std::isinf(x) ? x : std::copysign(std::pow(std::abs(x), 1.0 / n), x);
}

/// The box as the ranges of the variables of an expression.
CORA_FROM_NUMBERS std::vector<Range> aux_ranges(const std::vector<Iv> &box) {
    std::vector<Range> out;
    out.reserve(box.size());
    for (const Iv &b : box)
        out.emplace_back(Tensor::from_data({b.lo}, {1, 1}), Tensor::from_data({b.hi}, {1, 1}));
    return out;
}

// The contractor ----------------------------------------------------------------------------------

/// One expression over a box: its forward values and the box it cuts.
class Contractor {
  public:
    Contractor(const Expr &f, const std::vector<Iv> &box)
        : f_(f), box_(box), cut_(box), ranges_(aux_ranges(box)) {}

    /// Pushes f = 0 back to the variables; false if that shows that no point satisfies it.
    bool run() {
        if (!backward(f_, {0.0, 0.0})) return false;
        box_ = cut_;
        return true;
    }

    const std::vector<Iv> &box() const { return box_; }

  private:
    /// The range of the subtree over the original box (once per subtree).
    Iv forward(const Expr &e) {
        if (const auto it = forward_.find(e.id()); it != forward_.end()) return it->second;
        const Range r = e.enclose(ranges_);
        return forward_.emplace(e.id(), Iv{r.lo.data()[0], r.hi.data()[0]}).first->second;
    }

    /// Cuts the subtree to the values in `target`; false if it has none.
    bool backward(const Expr &e, const Iv &target) {
        // only the values the subtree takes over the box can reach the target
        const Iv t = aux_meet(target, forward(e));
        if (t.empty()) return false;
        const ExprView v = e.view();
        // each operation inverts as far as it soundly can
        switch (v.op) {
        case ExprOp::var: {
            Iv &x = cut_[std::size_t(v.index)];
            x = aux_meet(x, t);
            return !x.empty();
        }
        case ExprOp::add:
            return backward(v.a, aux_sub(t, forward(v.b))) &&
                   backward(v.b, aux_sub(t, forward(v.a)));
        case ExprOp::sub:
            return backward(v.a, aux_add(t, forward(v.b))) &&
                   backward(v.b, aux_sub(forward(v.a), t));
        case ExprOp::neg:
            return backward(v.a, aux_neg(t));
        case ExprOp::mul:
            return backward(v.a,
                            aux_has_zero(forward(v.b)) ? aux_whole() : aux_div(t, forward(v.b))) &&
                   backward(v.b,
                            aux_has_zero(forward(v.a)) ? aux_whole() : aux_div(t, forward(v.a)));
        case ExprOp::div:
            // a = t * b; the divisor is left as it is
            return backward(v.a, aux_mul(t, forward(v.b)));
        case ExprOp::pow:
            return backward_power(v, t);
        case ExprOp::rpow:
            return v.value == 0.5 ? backward_sqrt(v, t) : true;
        case ExprOp::exp:
            return t.hi > 0 &&
                   backward(v.a, aux_widen({std::log(std::max(t.lo, 0.0)), std::log(t.hi)}));
        default:
            // constants, sin, cos and tan: nothing to narrow
            return true;
        }
    }

    /// a^n = t for an integer n >= 1: the roots, two of them for an even n.
    bool backward_power(const ExprView &v, const Iv &t) {
        if (v.index < 1) return true;
        if (v.index % 2 == 1)
            return backward(v.a, aux_widen({aux_root(t.lo, v.index), aux_root(t.hi, v.index)}));
        const Iv nonneg = aux_meet(t, {0.0, kInf});
        if (nonneg.empty()) return false;
        const Iv r =
            aux_widen({std::pow(nonneg.lo, 1.0 / v.index), std::pow(nonneg.hi, 1.0 / v.index)});
        // the operand is in [r.lo, r.hi] or in its mirror; take the hull of what the range allows
        const Iv a = forward(v.a);
        const Iv pos = aux_meet(a, r), neg = aux_meet(a, aux_neg(r));
        if (pos.empty() && neg.empty()) return false;
        if (pos.empty()) return backward(v.a, neg);
        if (neg.empty()) return backward(v.a, pos);
        return backward(v.a, {neg.lo, pos.hi});
    }

    /// sqrt(a) = t: a in [t.lo^2, t.hi^2] for t >= 0.
    bool backward_sqrt(const ExprView &v, const Iv &t) {
        const Iv nonneg = aux_meet(t, {0.0, kInf});
        if (nonneg.empty()) return false;
        return backward(v.a, aux_widen({nonneg.lo * nonneg.lo, nonneg.hi * nonneg.hi}));
    }

    Expr f_;
    std::vector<Iv> box_, cut_;
    std::vector<Range> ranges_;
    std::unordered_map<const void *, Iv> forward_;
};

} // namespace


// ===========================================  MAIN  =========================================== //

// The checks and the contraction ------------------------------------------------------------------

std::optional<Interval> LevelSet::tighten_domain(const Interval &I, const std::string &type) const {
    detail::Entry entry;
    CORA_CHECK(type == "forwardBackward" || type == "split" || type == "taylor" || type == "linear",
               Error::wrong_value, [&] {
                   return "tighten_domain: the type must be forwardBackward, split, taylor or "
                          "linear, got '" +
                          type + "'";
               });
    CORA_CHECK(type == "forwardBackward", Error::not_supported, [&] {
        return "tighten_domain: the contractor '" + type + "' is not ported, only forwardBackward";
    });
    CORA_CHECK(std::all_of(ops_.begin(), ops_.end(), [](CompOp op) { return op == CompOp::eq; }),
               Error::noops,
               [] { return std::string("tighten_domain: all constraints must be equalities"); });
    CORA_CHECK(!I.is_empty() && I.inf().ndim() == 2 &&
                   I.inf().shape() == std::vector<int64_t>({dim(), 1}),
               Error::dimension_mismatch, [&] {
                   return "tighten_domain: the box must be a nonempty column of " +
                          std::to_string(dim()) + " states";
               });
    // the box as host intervals
    const std::vector<double> lo = I.inf().data(), hi = I.sup().data();
    std::vector<Iv> box;
    box.reserve(lo.size());
    for (std::size_t i = 0; i < lo.size(); ++i) box.push_back({lo[i], hi[i]});
    // each constraint cuts the box the previous ones left
    for (const Expr &f : eq_) {
        Contractor c(f, box);
        if (!c.run()) return std::nullopt;
        box = c.box();
    }
    std::vector<double> new_lo, new_hi;
    for (const Iv &b : box) {
        new_lo.push_back(b.lo);
        new_hi.push_back(b.hi);
    }
    return Interval(Tensor::like(I.inf(), new_lo, {dim(), 1}),
                    Tensor::like(I.inf(), new_hi, {dim(), 1}));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
