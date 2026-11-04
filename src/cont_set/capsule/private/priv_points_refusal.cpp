// priv_points_refusal - why a matrix is no list of points of a capsule's space
//
// Syntax:   const detail::Refusal refusal = priv_points_refusal(C, p);
// Inputs:   C - capsule;  p - the points as columns, or anything else
// Outputs:  refusal - the identifier and text for CORA_CHECK if p is no matrix (..., n, N) with the
//           dimension n of C; nothing otherwise
// See also: Capsule::contains, Capsule::is_intersecting
//
// CORA.cpp only: MATLAB checks the dimensions with equalDimCheck

#include "cont_set/capsule/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

detail::Refusal priv_points_refusal(const Capsule &C, const Tensor &p) {
    if (!p.defined() || p.ndim() < 2 || p.size(-2) != C.dim())
        return std::pair{
            Error::dimension_mismatch,
            "the points must be a matrix with " + std::to_string(C.dim()) + " rows, got " +
                (p.defined() ? "the shape " + shape_text(p.shape()) : std::string("no tensor"))};
    return std::nullopt;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
