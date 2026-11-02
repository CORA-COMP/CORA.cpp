// priv_verify - the private functions behind linear_sys verify, as CORA's @linearSys/private
//
// Not for callers of the class: verify uses them.
//
// Syntax:   priv_verify_ra_support_function(sys, params, specs);   priv_verify_ra_zonotope(sys, ...);
// See also: linear_sys.h, verify.cpp

#pragma once

#include "cora/cont_dynamics/linear_sys/linear_sys.h"

#include <optional>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The input in canonical form x' = A x + u, u in u_trans + GU [-1, 1]^k: u_trans = B center(U)
/// and GU = B generators(U), without zero generators (GU (n, 0) if there are none).
struct CanonicalInput {
    Tensor u_trans, GU;
};
CanonicalInput priv_canonical_input(const LinearSys &sys, const Zonotope &U);

/// The algorithm "reachavoid:supportFunc": the support function of the affine solution along
/// the specification directions, with an adaptive step size. The inputs are checked by verify.
VerifyResult priv_verify_ra_support_function(const LinearSys &sys, const Zonotope &R0,
                                             const Zonotope &U, double t_final,
                                             const std::vector<Specification> &specs);

/// The algorithm "reachavoid:zonotope": adaptive zonotope reachable sets with a bound on their
/// error to the exact sets, refined until the outer and inner approximations decide.
VerifyResult priv_verify_ra_zonotope(const LinearSys &sys, const Zonotope &R0, const Zonotope &U,
                                     double t_final, const std::vector<Specification> &specs);

/// The inverse of a square matrix, empty if it is (numerically) singular.
std::optional<Tensor> priv_inverse(const Tensor &A);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
