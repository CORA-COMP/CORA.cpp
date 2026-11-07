// project - the projection of an ellipsoid onto some dimensions, as CORA's ellipsoid.project
//
// P E for the rows P of the identity that are kept: Q becomes P Q P', q becomes P q. CORA returns
// the empty ellipsoid unchanged; here it has the new dimension.
//
// Syntax:   Eres = E.project(dims);
// Inputs:   dims - the 0-based dimensions, in the order of the result
// Outputs:  Eres - the projected ellipsoid
// See also: lift, linear_map
//
// CORA: contSet/@ellipsoid/project.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/ellipsoid/private/priv.h"
#include "cora/cont_set/ellipsoid/ellipsoid.h"
#include "global/format.h"
#include "global/select.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Set Ellipsoid::project_impl(const std::vector<int64_t> &dims) const {
    if (priv_is_empty(*this)) return Ellipsoid::empty(static_cast<int64_t>(dims.size()));
    const Tensor P = select_dims(q_, dims, dim());
    return Ellipsoid(priv_congruence(P, Q_), P.matmul(q_), tol_);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
