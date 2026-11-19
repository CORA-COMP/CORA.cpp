// semi_axes - the semi-axes of an ellipsoid, as CORA's ellipsoid.radius
//
// The square roots of the i largest eigenvalues of Q, descending, for the shape matrix Q of the
// ellipsoid; the empty ellipsoid has none.
//
// Syntax:   r = E.semi_axes();   r = E.semi_axes(i);
// Inputs:   i - how many axes, 1..dim (default 1)
// Outputs:  r - the lengths, (..., i); (0) for the empty ellipsoid
// See also: ball_radius, volume, norm
//
// CORA: contSet/@ellipsoid/radius.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       06-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/ellipsoid/private/priv.h"
#include "cora/cont_set/ellipsoid/ellipsoid.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Ellipsoid::semi_axes(int64_t i) const {
    detail::Entry entry;
    CORA_CHECK(i >= 1 && i <= dim(), Error::wrong_value, [&] {
        return "semi_axes: i must be in 1.." + std::to_string(dim()) + ", got " + std::to_string(i);
    });
    if (priv_is_empty(*this)) return q_.zeros_like({0});
    // the eigenvalues ascend: the last i, reversed (eigh gives them as a column (..., n, 1))
    const Tensor values = Q_.eigh().first;
    const int axis = static_cast<int>(values.ndim()) - 2;
    return values.slice(axis, dim() - i, dim()).flip(axis).maximum(0.0).sqrt().squeeze(-1);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
