// times - a set scaled coordinate by coordinate, as CORA's contSet.times
//
// Syntax:   R = S.times(factor);   R = S.times(2.0);
// Inputs:   S - a set of dimension n
//           factor - a number or a column (..., n, 1), one factor per
//           coordinate (CORA: a numeric vector; a matrix is Error::not_supported)
// Outputs:  R - diag(factor) * S, in the class of S; a class with its own product (a box)
//           overrides it
// See also: enlarge, linear_map, uminus
//
// CORA: contSet/@contSet/times.m
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

Set ContSet::times(const Tensor &factor) const {
    detail::Entry entry;
    CORA_CHECK(factor.ndim() >= 2 && factor.size(-1) == 1, Error::not_supported, [&] {
        return "times: the factor has to be a column vector, got the shape " +
               shape_text(factor.shape());
    });
    CORA_CHECK(is_column(factor, dim()) || factor.numel() == 1, Error::dimension_mismatch, [&] {
        return "times: the factor must be a number or a column of the set's dimension " +
               std::to_string(dim()) + ", got the shape " + shape_text(factor.shape());
    });
    return times_impl(factor);
}

Set ContSet::times(double factor) const {
    detail::Entry entry;
    return times_impl(center_impl().full_like({1, 1}, factor));
}

Set ContSet::times_impl(const Tensor &factor) const {
    return linear_map_impl(factor.ones_like({dim_impl(), 1}).multiply(factor).diag());
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
