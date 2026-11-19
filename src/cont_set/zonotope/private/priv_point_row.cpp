// priv_point_row - a point given to a zonotope operation, as a row of the zonotope's layout
//
// Syntax:   Tensor p = priv_point_row(v, Z);
// Inputs:   v - a point: a column (..., n, 1) or a row (..., n); a matrix for a matrix zonotope
//           Z - the zonotope
// Outputs:  p - the point in the shape of the center of Z (batch..., elem...), to broadcast
// See also: Zonotope::operator+, Zonotope::translate_impl
//
// CORA.cpp only: a column point of the column-layout code meets the batch-first center (#163).
// Authors:       Claude Opus 5.5
// Written:       18-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/zonotope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

// points are columns (..., n, 1), as every set takes them
Tensor priv_point_row(const Tensor &v, const Zonotope &Z) {
    if (Z.elem_rank() != 1 || v.ndim() < 2 || v.size(-1) != 1 || v.size(-2) != Z.dim()) return v;
    // a one-dimensional set takes (..., 1, 1) as a column only beyond the rank of its center
    if (Z.dim() == 1 && v.ndim() <= Z.center().ndim()) return v;
    return v.squeeze(-1);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
