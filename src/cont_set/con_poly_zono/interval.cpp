// interval - a box that holds a constrained polynomial zonotope, as CORA's interval
//
// c -+ sum |G| -+ sum |GI|, each sum over the generators from the last (CORALean's
// ConPolyZono.interval: the constraint dropped). CORA bounds the constrained zonotope of the set
// by linear programs, which is tighter. support_function is the zonotope's, an upper bound.
//
// Syntax:   I = cPZ.interval();   s = cPZ.support_function(d);
// Inputs:   d - directions (..., n)
// Outputs:  I - the box
//           s - upper bounds of the support function
// See also: con_poly_zono.h
//
// CORA: contSet/@conPolyZono/interval.m
// CORA: contSet/@conPolyZono/supportFunc_.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_poly_zono/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval ConPolyZono::interval_impl() const {
    // the reach of each block, then the center minus and plus them, in that order
    const Tensor r_dep = generators().abs().sum(0), r_indep = independent_generators().abs().sum(0);
    return {c() - r_dep - r_indep, c() + r_dep + r_indep};
}

Tensor ConPolyZono::support_function_impl(const Tensor &d) const {
    return zonotope().support_function(d);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
