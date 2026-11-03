// minkowski_difference - the Minkowski difference of a box and a set, as CORA's interval.minkDiff
//
// Syntax:   Interval R = I.minkowski_difference(S);   Interval R = I.minkowski_difference(p);
//           Interval R = I.minkowski_difference(S, "inner");
// Inputs:   S - an interval, a point (a column), or another set;  type - "exact" or "inner", for a
//           set other than a box (default: the support-function bounds, as "inner")
// Outputs:  R - {x | x + S in I}: [inf - inf_S, sup - sup_S] for a box, [inf - p, sup - p] for a
//           point, and for another set [inf + h(-e_i), sup - h(e_i)] with h its support function
//           in the coordinate directions e_i. The empty interval (bounds without columns) if a
//           bound would cross (CORA gives []); for a batch all members must agree on that.
//           Error::no_exact_alg for type "exact" and another set
// See also: minus, plus
//
// CORA: contSet/@interval/minkDiff.m

#include "cont_set/interval/private/priv_empty.h"
#include "cora/cont_set/interval/interval.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The box [lower, upper], or the empty interval if a bound crosses (all members alike).
Interval aux_box_or_empty(const Tensor &lower, const Tensor &upper) {
    const Tensor crossed = lower.greater(upper).any(-2, true).any(-1, true);
    if (crossed.max().item() == 0) return {lower, upper};
    if (crossed.min().item() == 0)
        CORA_FAIL(Error::not_supported,
                  "minkowski_difference: the members of a batch must all be empty or none");
    return priv_empty(lower);
}

} // namespace


// ===========================================  MAIN  =========================================== //

Interval Interval::minkowski_difference(const Interval &S) const {
    detail::Entry entry;
    CORA_CHECK(S.dim() == dim(), Error::dimension_mismatch, [&] {
        return "minkowski_difference: the intervals differ in dimension, " +
               std::to_string(dim()) + " and " + std::to_string(S.dim());
    });
    return aux_box_or_empty(inf_ - S.inf(), sup_ - S.sup());
}

Interval Interval::minkowski_difference(const Tensor &p) const {
    detail::Entry entry;
    CORA_CHECK(is_column(p, dim()), Error::dimension_mismatch, [&] {
        return "minkowski_difference: the point must be a column of the dimension " +
               std::to_string(dim()) + " of the interval, got the shape " + shape_text(p.shape());
    });
    return minus(p);
}

Interval Interval::minkowski_difference(const ContSet &S,
                                        const std::optional<std::string> &type) const {
    detail::Entry entry;
    // CORA raises for these as for the other operands
    CORA_CHECK(S.dim() == dim(), Error::dimension_mismatch, [&] {
        return "minkowski_difference: the sets differ in dimension, " + std::to_string(dim()) +
               " and " + std::to_string(S.dim());
    });
    CORA_CHECK(!type || *type == "exact" || *type == "inner", Error::wrong_value, [&] {
        return "minkowski_difference: type must be 'exact' or 'inner', got '" + *type + "'";
    });
    // there is no exact algorithm for a set that is no box
    CORA_CHECK(!type || *type != "exact", Error::no_exact_alg, [&] {
        return "minkowski_difference: there is no exact algorithm for a " + S.class_name();
    });
    // the support function of S along +-e_i bounds coordinate i of the difference
    const Tensor axes = inf_.eye_like(dim());
    std::vector<Tensor> up, down;
    for (int64_t i = 0; i < dim(); ++i) {
        const Tensor e = axes.slice(-1, i, i + 1);
        up.push_back(S.support_function(e));
        down.push_back(S.support_function(-e));
    }
    return aux_box_or_empty(inf_ + Tensor::concat(down, -2), sup_ - Tensor::concat(up, -2));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
