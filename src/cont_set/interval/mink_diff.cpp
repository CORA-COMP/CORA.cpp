// mink_diff - the Minkowski difference of a box and a set, as CORA's interval.minkDiff
//
// For a box [inf - inf_S, sup - sup_S], for a point [inf - p, sup - p], and for another set [inf
// + h(-e_i), sup - h(e_i)] with h its support function in the coordinate directions e_i; the
// default method uses the support-function bounds. The empty interval (bounds without columns) if a
// bound would cross (CORA gives []); for a batch all members must agree on that.
// Error::no_exact_alg for method "exact" and another set.
//
// Syntax:   Interval R = I.mink_diff(S);   Interval R = I.mink_diff(p);
//           Interval R = I.mink_diff(S, "inner");
// Inputs:   S - an interval, a point (a column), or another set
//           method - "exact" or "inner", for a set other than a box (default: "inner")
// Outputs:  R - {x | x + S in I}
// See also: minus, plus
//
// CORA: contSet/@interval/minkDiff.m
// CORA authors: Niklas Kochdumper
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/interval/private/priv_empty.h"
#include "cora/cont_set/interval/interval.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The interval [lower, upper] of the shape of I, or the empty one if a bound crosses (all
/// members alike).
Interval aux_box_or_empty(const Interval &I, const Tensor &lower, const Tensor &upper) {
    const Tensor crossed = lower.greater(upper);
    if (crossed.max().item() == 0)
        return I.elem_rank() == 1 ? Interval(lower, upper) : Interval::matrix(lower, upper);
    // every member crossed in some entry, or none
    Tensor per_member = crossed;
    for (int64_t a = 0; a < I.elem_rank(); ++a) per_member = per_member.any(-1);
    if (per_member.min().item() == 0)
        CORA_FAIL(Error::not_supported,
                  "mink_diff: the members of a batch must all be empty or none");
    return priv_empty(I);
}

} // namespace


// ===========================================  MAIN  =========================================== //

Interval Interval::mink_diff(const Interval &S) const {
    detail::Entry entry;
    CORA_CHECK(S.dim() == dim(), Error::dimension_mismatch, [&] {
        return "mink_diff: the intervals differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(S.dim());
    });
    return aux_box_or_empty(*this, inf_ - S.inf(), sup_ - S.sup());
}

Interval Interval::mink_diff(const Tensor &p) const {
    detail::Entry entry;
    CORA_CHECK(elem_rank_ == 1 && is_vector(p, dim()), Error::dimension_mismatch, [&] {
        return "mink_diff: the point must be a vector of the dimension " + std::to_string(dim()) +
               " of the interval, got the shape " + shape_text(p.shape());
    });
    return minus(p);
}

Interval Interval::mink_diff(const ContSet &S, const std::optional<std::string> &method) const {
    detail::Entry entry;
    // CORA raises for these as for the other operands
    CORA_CHECK(S.dim() == dim(), Error::dimension_mismatch, [&] {
        return "mink_diff: the sets differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(S.dim());
    });
    CORA_CHECK(!method || *method == "exact" || *method == "inner", Error::wrong_value, [&] {
        return "mink_diff: method must be 'exact' or 'inner', got '" + *method + "'";
    });
    // there is no exact algorithm for a set that is no box
    CORA_CHECK(!method || *method != "exact", Error::no_exact_alg,
               [&] { return "mink_diff: there is no exact algorithm for a " + S.class_name(); });
    // the support function of S along +-e_i bounds coordinate i of the difference
    const Tensor axes = inf_.eye_like(dim());
    std::vector<Tensor> up, down;
    for (int64_t i = 0; i < dim(); ++i) {
        const Tensor e = axes.slice(-2, i, i + 1).squeeze(-2);
        // a direction is a vector (n), its support (...) one coordinate
        up.push_back(S.support_function(e).expand_dims(-1));
        down.push_back(S.support_function(-e).expand_dims(-1));
    }
    return aux_box_or_empty(*this, inf_ + Tensor::concat(down, -1), sup_ - Tensor::concat(up, -1));
}

Set Interval::mink_diff_impl(const ContSet &S2, const std::optional<std::string> &method) const {
    // a box takes the exact rule of boxes, whatever the method; any other set the support function
    if (S2.class_id() == ClassId::interval) return mink_diff(static_cast<const Interval &>(S2));
    return mink_diff(S2, method);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
