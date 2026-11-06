// linear_map - the image of a set under a matrix, as CORA's contSet.mtimes with a matrix
//
// Syntax:   R = S.linear_map(M);
// Inputs:   S - set of dimension n
//           M - matrix (..., m, n)
// Outputs:  R - M S, a set of dimension m in the class of S (a box: the hull of the image)
// See also: Interval::linear_map, Zonotope::linear_map
//
// CORA: contSet/@contSet/mtimes.m
// CORA authors: Tobias Ladner
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

Set ContSet::linear_map(const Tensor &M) const {
    detail::Entry entry;
    CORA_CHECK(M.ndim() >= 2 && M.size(-1) == dim(), Error::dimension_mismatch, [&] {
        return "linear_map: the matrix needs " + std::to_string(dim()) +
               " columns, the dimension of the set, got the shape " + shape_text(M.shape());
    });
    return linear_map_impl(M);
}

Set ContSet::linear_map_impl(const Tensor &) const { priv_noops("linear_map", *this); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
