// priv_verify - the private functions behind linearSys verify, as CORA's @linearSys/private
//
// Not for callers of the class: verify uses them.
//
// Syntax:   priv_verifyRA_supportFunc(sys, params, specs);   priv_verifyRA_zonotope(sys, ...);
// See also: linearSys.h, verify.cpp

#pragma once

#include "contDynamics/linearSys/linearSys.h"

#include <optional>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The algorithm "reachavoid:supportFunc": the support function of the affine solution along
/// the specification directions, with an adaptive step size.
VerifyResult priv_verifyRA_supportFunc(const LinearSys &sys, const VerifyParams &params,
                                       const std::vector<Specification> &specs);

/// The algorithm "reachavoid:zonotope": adaptive zonotope reachable sets with a bound on their
/// error to the exact sets, refined until the outer and inner approximations decide.
VerifyResult priv_verifyRA_zonotope(const LinearSys &sys, const VerifyParams &params,
                                    const std::vector<Specification> &specs);

/// The inverse of a square matrix, empty if it is (numerically) singular.
std::optional<Tensor> priv_inverse(const Tensor &A);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
