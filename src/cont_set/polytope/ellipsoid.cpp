// ellipsoid - an ellipsoid that encloses a bounded polytope, as CORA's polytope.ellipsoid
//
// "outer" (default) is the covariance ellipsoid of the vertices (Ellipsoid::enclose_points); the
// empty set is the empty ellipsoid. The minimum-volume ellipsoid ("outer:min-vol") and the largest
// inscribed one ("inner") are semidefinite programs, for which CORA.cpp has no solver
// (Error::not_supported). Not a batch.
//
// Syntax:   Ellipsoid E = P.ellipsoid();   Ellipsoid E = P.ellipsoid("outer");
// Inputs:   method - "outer" (default), "outer:min-vol" or "inner"
// Outputs:  E - an ellipsoid that contains the set
// See also: Polytope::zonotope, Ellipsoid::enclose_points
//
// CORA: contSet/@polytope/ellipsoid.m
// CORA authors: Victor Gassmann, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Ellipsoid Polytope::ellipsoid(const std::string &method) const {
    detail::Entry entry;
    CORA_CHECK(method == "outer" || method == "outer:min-vol" || method == "inner",
               Error::wrong_value, [&] {
                   return "ellipsoid: the method must be outer, outer:min-vol or inner, got '" +
                          method + "'";
               });
    const Tensor &like = is_h_rep() ? A_ : V_;
    // a single, bounded set; the empty set has the empty ellipsoid
    priv_require_enclosable(*this, "ellipsoid", "an ellipsoid");
    if (representsa("emptySet", machine_eps(like.dtype()))) return Ellipsoid::empty(dim());
    CORA_CHECK(method == "outer", Error::not_supported, [&] {
        return "ellipsoid: '" + method + "' is a semidefinite program, which has no solver here";
    });
    return Ellipsoid::enclose_points(
        priv_points_tensor(like, dim(), priv_polytope_vertices(*this)));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
