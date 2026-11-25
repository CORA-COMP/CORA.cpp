// rescale_cols - the polytope in the coordinates of a box, as CORALean's rescaleCols
//
// The coordinates x = mid + diag(rad) y: the rows A diag(rad) (scale_cols) and the offsets
// b - A mid, the equalities likewise; a constrained zonotope rescales its factors so (CORA's
// rescale). A polytope in vertex form is converted by constraints first (not a batch). The
// empty-set flag is kept. CORALean's float definition also bounds the rounding in a slack; the
// nearest values are these.
//
// Syntax:   Pres = P.rescale_cols(mid, rad);
// Inputs:   P - polytope of dimension n
//           mid, rad - the center and the radius of the box (..., n)
// Outputs:  Pres - {y | mid + diag(rad) y in P}, in halfspace form
// See also: scale_cols, ConZonotope::rescale
//
// CORA.cpp only: CORALean's HPolytope.rescaleCols, the factor rows of a constrained zonotope
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The offsets o - M mid of the rows M (..., rows, n) at the point mid (..., n).
Tensor aux_offsets(const Tensor &M, const Tensor &o, const Tensor &mid) {
    return o - M.matmul(mid.expand_dims(-1)).squeeze(-1);
}

} // namespace


// ===========================================  MAIN  =========================================== //

Polytope Polytope::rescale_cols(const Tensor &mid, const Tensor &rad) const {
    detail::Entry entry;
    const int64_t n = dim();
    CORA_CHECK(mid.defined() && rad.defined() && is_vector(mid, n) && is_vector(rad, n),
               Error::dimension_mismatch, [&] {
                   return "rescale_cols: the center and the radius must be vectors (..., " +
                          std::to_string(n) + "), got the shapes " +
                          (mid.defined() ? shape_text(mid.shape()) : std::string("()")) + " and " +
                          (rad.defined() ? shape_text(rad.shape()) : std::string("()"));
               });
    const Constraints C = constraints();
    const Polytope scaled = scale_cols(rad);
    return Polytope(Parts{scaled.A_, aux_offsets(C.A, C.b, mid), scaled.Ae_,
                          aux_offsets(C.Ae, C.be, mid), Tensor(), empty_});
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
