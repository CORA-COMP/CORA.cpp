// reduce_over_domain - one halfspace for all rows over a box, as CORA's reduceOverDomain
//
// The rows are normalized to unit length; the interval A holds the least and the largest entry of
// every column, the interval b the least and the largest offset. The halfspace has the center of A
// as its normal and the offset b - (A - center)' dom: its supremum ("outer": a halfspace that
// contains the set over dom) or its infimum ("inner": one that the set contains over dom).
//
// Syntax:   Pres = P.reduce_over_domain(dom);   Pres = P.reduce_over_domain(dom, "inner");
// Inputs:   dom - a box (n, 1)
//           type - "outer" (default) or "inner"
// Outputs:  Pres - a polytope of one row
// See also: Polytope::normalize_constraints, Polytope::reduce
//
// CORA: contSet/@polytope/reduceOverDomain.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Polytope Polytope::reduce_over_domain(const Interval &dom, const std::string &type) const {
    detail::Entry entry;
    CORA_CHECK(type == "outer" || type == "inner", Error::wrong_value, [&] {
        return "reduce_over_domain: the type must be outer or inner, got '" + type + "'";
    });
    CORA_CHECK(is_h_rep() && A_.ndim() == 2 && A_.shape()[0] > 0, Error::special_error, [] {
        return std::string("reduce_over_domain: it needs a single polytope with halfspaces");
    });
    CORA_CHECK(dom.dim() == dim() && dom.inf().shape().back() == 1, Error::dimension_mismatch, [&] {
        return "reduce_over_domain: the domain must be a box of dimension " +
               std::to_string(dim()) + ", got the shape " + shape_text(dom.inf().shape());
    });
    // the unit rows, the ranges of their entries and of their offsets
    const Polytope unit = normalize_constraints("A");
    const Interval A(unit.A().min(0, true).matrix_transpose(),
                     unit.A().max(0, true).matrix_transpose());
    const Interval b(unit.b().min(0, true), unit.b().max(0, true));
    const Tensor reduced = A.center();
    // b - (A - reduced)' dom, a box of one number
    const Interval offset = b.minus((A.minus(reduced)).times(dom).sum(1));
    return {reduced.matrix_transpose(), type == "outer" ? offset.sup() : offset.inf()};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
