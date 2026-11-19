// priv_points_refusal - why a matrix is no list of points of a capsule's space
//
// Syntax:   const detail::Refusal refusal = priv_points_refusal(C, p);
// Inputs:   C - capsule
//           p - the points as rows, or anything else
// Outputs:  refusal - the identifier and text if p is no point (..., n); else nothing
// See also: Capsule::contains, Capsule::is_intersecting
//
// CORA.cpp only: MATLAB checks the dimensions with equalDimCheck
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       06-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/capsule/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

detail::Refusal priv_points_refusal(const Capsule &C, const Tensor &p) {
    if (!p.defined() || !is_vector(p, C.dim()))
        return std::pair{
            Error::dimension_mismatch,
            "the points must be (..., " + std::to_string(C.dim()) + "), one per row, got " +
                (p.defined() ? "the shape " + shape_text(p.shape()) : std::string("no tensor"))};
    return std::nullopt;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
