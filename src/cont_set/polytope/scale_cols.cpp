// scale_cols - the polytope in coordinates scaled one by one
//
// Column j of every row is multiplied by d_j, the offsets stay: {y | diag(d) y in P}. A polytope
// in vertex form is converted by constraints first (not a batch). The empty-set flag is kept.
//
// Syntax:   Pres = P.scale_cols(d);
// Inputs:   P - polytope of dimension n
//           d - the scales (..., n), of the batch shape of P or broadcast to it
// Outputs:  Pres - {y | diag(d) y in P}, in halfspace form
// See also: rescale_cols, select_cols
//
// CORA.cpp only: the scaling step of CORALean's HPolytope.rescaleCols
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Polytope Polytope::scale_cols(const Tensor &d) const {
    detail::Entry entry;
    CORA_CHECK(d.defined() && is_vector(d, dim()), Error::dimension_mismatch, [&] {
        return "scale_cols: the scales must be a vector (..., " + std::to_string(dim()) +
               "), got the shape " + (d.defined() ? shape_text(d.shape()) : std::string("()"));
    });
    const Constraints C = constraints();
    // the scales as a row, one per column
    const Tensor row = d.expand_dims(-2);
    return Polytope(Parts{C.A.multiply(row), C.b, C.Ae.multiply(row), C.be, Tensor(), empty_});
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
