// ellipsoid - an ellipsoid of a capsule, as CORA's capsule.ellipsoid
//
// A capsule without a generator is the ball of the ellipsoid r^2 I around c; the empty capsule
// gives the empty ellipsoid. With a generator the capsule is no ellipsoid: "exact" is
// Error::no_exact_alg, and "outer" and "inner" need the SDP-based union of ellipsoids and are
// Error::not_supported.
//
// Syntax:   E = C.ellipsoid();   E = C.ellipsoid(method);
// Inputs:   method - "exact" (default), "outer" or "inner"
// Outputs:  E - the ellipsoid
// See also: zonotope, polytope
//
// CORA: contSet/@capsule/ellipsoid.m
// CORA authors: Victor Gassmann, Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/ellipsoid/ellipsoid.h"
#include "cont_set/capsule/private/priv.h"
#include "cora/cont_set/capsule/capsule.h"
#include "cora/global/within_tol.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Ellipsoid Capsule::ellipsoid(const std::string &method) const {
    detail::Entry entry;
    CORA_CHECK(method == "exact" || method == "outer" || method == "inner", Error::wrong_value,
               [&] {
                   return "ellipsoid: the method must be 'exact', 'outer' or 'inner', got '" +
                          method + "'";
               });
    CORA_CHECK(c_.ndim() == 2, Error::not_supported,
               [] { return std::string("ellipsoid: it supports a single capsule, not a batch"); });
    if (is_empty()) return Ellipsoid::empty(dim());
    // CORA ignores the method for a capsule without generator, which is a ball
    const bool ball = priv_all(g_.equal(g_.zeros_like()));
    if (!ball) {
        CORA_CHECK(method != "exact", Error::no_exact_alg, [] {
            return std::string(
                "ellipsoid: a capsule with a generator is no ellipsoid, use 'outer'");
        });
        CORA_FAIL(Error::not_supported, "ellipsoid: the method '" + method +
                                            "' needs the union of two ellipsoids (an SDP)");
    }
    return {r_.square().multiply(c_.eye_like(dim())), c_};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
