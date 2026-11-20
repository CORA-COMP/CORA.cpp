// times - a set scaled coordinate by coordinate, as CORA's contSet.times
//
// CORA takes a numeric vector; CORA.cpp reads a factor (..., n) as a batch of vectors.
//
// A class with its own product (a box) overrides it.
//
// Syntax:   R = S.times(factor);   R = S.times(2.0);
// Inputs:   S - a set of dimension n
//           factor - a number or a vector (..., n), one factor per coordinate
// Outputs:  R - diag(factor) * S, in the class of S
// See also: enlarge, linear_map, uminus
//
// CORA: contSet/@contSet/times.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/cont_set.h"
#include "cora/cont_set/set.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Set ContSet::times(const Tensor &factor) const {
    detail::Entry entry;
    CORA_CHECK(is_vector(factor, dim()) || factor.numel() == 1, Error::dimension_mismatch, [&] {
        return "times: the factor must be a number or a vector of the set's dimension " +
               std::to_string(dim()) + ", got the shape " + shape_text(factor.shape());
    });
    // what the generic entry owns is checked: the class checks the rest
    entry.release();
    return times_impl(factor);
}

Set ContSet::times(double factor) const {
    detail::Entry entry;
    entry.release();
    return times_impl(center_impl().full_like({1}, factor));
}

Set ContSet::times_impl(const Tensor &factor) const {
    return linear_map_impl(factor.ones_like({dim_impl()}).multiply(factor).expand_dims(-1).diag());
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
