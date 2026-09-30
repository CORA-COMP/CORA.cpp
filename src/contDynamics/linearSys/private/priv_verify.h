// priv_verify - the private functions behind linearSys verify, as CORA's @linearSys/private
//
// Not for callers of the class: verify uses them.
//
// Syntax:   priv_verifyRA_supportFunc(sys, params, specs);   priv_verifyRA_zonotope(...);
// See also: linearSys.h, verify.cpp

#pragma once

#include "contDynamics/linearSys/linearSys.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The algorithm "reachavoid:supportFunc": the support function of the affine solution along
/// the specification directions, with an adaptive step size.
VerifyResult priv_verifyRA_supportFunc(const LinearSys &sys, const VerifyParams &params,
                                       const std::vector<Specification> &specs);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
