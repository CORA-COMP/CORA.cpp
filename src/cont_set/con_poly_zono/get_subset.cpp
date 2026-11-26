// get_subset - the set at a factor restricted to a sub-box, as CORA's getSubset
//
// The factor id becomes cen + rad a, cen and rad the center and the half-width of val: every
// generator and every monomial of the constraint is expanded binomially in a (CORALean's
// getSubset), one column per power, nothing merged (compact merges them).
//
// Syntax:   S = cPZ.get_subset(id, val);
// Inputs:   id - an identifier of the set
//           val - a box (1) in [-1, 1]
// Outputs:  S - the subset
// See also: con_poly_zono.h
//
// CORA: contSet/@conPolyZono/getSubset.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_poly_zono/private/priv.h"
#include "cora/global/error.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

ConPolyZono ConPolyZono::get_subset(int64_t id, const Interval &val) const {
    detail::Entry entry;
    const int64_t k = priv_find_id(ids(), id);
    CORA_CHECK(k >= 0, Error::wrong_value, [&] {
        return "get_subset: " + std::to_string(id) + " is not an identifier of the set";
    });
    CORA_CHECK(!val.is_empty() && val.inf().numel() == 1, Error::wrong_value,
               [] { return std::string("get_subset: val must be one nonempty interval"); });
    // the center and the radius of the box
    const double lo = val.inf().item(), hi = val.sup().item();
    const double cen = (hi + lo) * 0.5, rad = (hi - lo) * 0.5;
    const ExpandedColumns x =
        priv_get_subset_expand(generators(), exponents(), std::size_t(k), cen, rad);
    return {PolyZonotope(c(), x.G, independent_generators(), x.E, ids(), detail::Verbatim{}),
            cons_.get_subset(k, cen, rad)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
