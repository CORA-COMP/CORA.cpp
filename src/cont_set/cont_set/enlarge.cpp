// enlarge - a set scaled about its center, as CORA's contSet.enlarge
//
// A class with its own formula (a box) overrides it.
//
// Syntax:   R = S.enlarge(factor);   R = S.enlarge(2.0);
// Inputs:   S - set of dimension n
//           factor - a number or a column (..., n, 1) of one factor per coordinate
// Outputs:  R - c + F (S - c) for the center c and F = diag(factor), in the class of S
// See also: minus, linear_map, Interval::enlarge
//
// CORA: contSet/@contSet/enlarge.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/cont_set.h"
#include "cora/cont_set/set.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Set ContSet::enlarge(const Tensor &factor) const {
    detail::Entry entry;
    CORA_CHECK(is_column(factor, dim()) || factor.numel() == 1, Error::dimension_mismatch, [&] {
        return "enlarge: the factor must be a number or a column of the set's dimension " +
               std::to_string(dim()) + ", got the shape " + shape_text(factor.shape());
    });
    return enlarge_impl(factor);
}

Set ContSet::enlarge(double factor) const {
    detail::Entry entry;
    return enlarge_impl(center_impl().full_like({1, 1}, factor));
}

Set ContSet::enlarge_impl(const Tensor &factor) const {
    const Tensor c = center_impl();
    // S = S - c; S = S .* factor; S = S + c
    const Tensor scale = c.ones_like().multiply(factor).diag();
    return minus(c)->linear_map_impl(scale)->translate_impl(c);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
