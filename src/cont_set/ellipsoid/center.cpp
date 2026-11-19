// center - the center of an ellipsoid, as CORA's ellipsoid.center
//
// The generic center is a column; for the empty ellipsoid it is NaN, as for the empty box (CORA's
// has no entries, (n, 0)).
//
// Syntax:   Tensor c = E.center();
// Inputs:   -
// Outputs:  c - the stored center q as a column (..., n, 1), a point of the ellipsoid
// See also: interval, ContSet::center
//
// CORA: contSet/@ellipsoid/center.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/ellipsoid/private/priv.h"
#include "cora/cont_set/ellipsoid/ellipsoid.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

// the generic center is a column until the points of every set are rows (#163); empty: NaN
Tensor Ellipsoid::center_impl() const {
    const Tensor q = q_.expand_dims(-1);
    if (priv_is_empty(*this))
        return q.full_like(q.shape(), std::numeric_limits<double>::quiet_NaN());
    return q;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
