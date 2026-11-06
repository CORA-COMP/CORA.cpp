// minus - a set shifted by a vector, as CORA's contSet.minus
//
// Syntax:   R = S.minus(p);   R = ContSet::minus(p, S);
// Inputs:   S - set of dimension n
//           p - a column (..., n, 1)
// Outputs:  R - S + (-p) and -S + p; the Minkowski difference of two sets is minkowski_difference
// See also: minkowski_difference, Interval::minus
//
// CORA: contSet/@contSet/minus.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/cont_set/private/priv_noops.h"
#include "cora/cont_set/cont_set.h"
#include "cora/cont_set/set.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Set ContSet::minus(const Tensor &p) const {
    detail::Entry entry;
    CORA_CHECK(is_column(p, dim()), Error::dimension_mismatch, [&] {
        return "minus: the vector must be a column of the set's dimension " +
               std::to_string(dim()) + ", got the shape " + shape_text(p.shape());
    });
    return translate_impl(-p);
}

Set ContSet::minus(const Tensor &p, const ContSet &S) {
    detail::Entry entry;
    CORA_CHECK(is_column(p, S.dim()), Error::dimension_mismatch, [&] {
        return "minus: the vector must be a column of the set's dimension " +
               std::to_string(S.dim()) + ", got the shape " + shape_text(p.shape());
    });
    // -S + p: the matrix -I maps, the vector translates
    return S.linear_map_impl(-p.eye_like(S.dim()))->translate_impl(p);
}

Set ContSet::translate_impl(const Tensor &) const { priv_noops("minus", *this); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
