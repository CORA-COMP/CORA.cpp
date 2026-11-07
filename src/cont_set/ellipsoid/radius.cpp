// radius - the semi-axes of an ellipsoid, as CORA's ellipsoid.radius
//
// The square roots of the i largest eigenvalues of Q, descending, for the shape matrix Q of the
// ellipsoid; the empty ellipsoid gives a matrix without columns (n, 0).
//
// Syntax:   r = E.radius();   r = E.radius(i);
// Inputs:   i - how many axes, 1..dim (default 1)
// Outputs:  r - the lengths, (..., i, 1)
// See also: volume, norm
//
// CORA: contSet/@ellipsoid/radius.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/ellipsoid/private/priv.h"
#include "cora/cont_set/ellipsoid/ellipsoid.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Ellipsoid::radius(int64_t i) const {
    detail::Entry entry;
    CORA_CHECK(i >= 1 && i <= dim(), Error::wrong_value, [&] {
        return "radius: i must be in 1.." + std::to_string(dim()) + ", got " + std::to_string(i);
    });
    if (priv_is_empty(*this)) return q_.zeros_like({dim(), 0});
    // the eigenvalues ascend: the last i, reversed
    const Tensor values = Q_.eigh().first;
    const int axis = static_cast<int>(values.ndim()) - 2;
    return values.slice(axis, dim() - i, dim()).flip(axis).maximum(0.0).sqrt();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
