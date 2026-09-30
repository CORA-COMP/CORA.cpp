// verify - verifies a linear system against reach-avoid specifications, as CORA's linearSys.verify
//
// Syntax:   res = sys.verify(params, alg, specs);
// Inputs:   params - initial set R0, input set U and time horizon tFinal
//           alg - VerifyAlg::SupportFunc or VerifyAlg::Zonotope
//           specs - safe sets and unsafe sets (one halfspace) over the outputs y = C x
// Outputs:  res - verified, the iterations of the adaptive loop, the last step size and step
//           count, and a falsifying initial state when a specification is hit
// See also: reach, private/priv_verifyRA_supportFunc

#include "contDynamics/linearSys/private/priv_verify.h"

#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

VerifyResult LinearSys::verify(const VerifyParams &params, VerifyAlg alg,
                               const std::vector<Specification> &specs) const {
    if (alg == VerifyAlg::SupportFunc) return priv_verifyRA_supportFunc(*this, params, specs);
    if (alg == VerifyAlg::Zonotope) return priv_verifyRA_zonotope(*this, params, specs);
    throw std::invalid_argument(
        "LinearSys::verify: unknown verifyAlg; use VerifyAlg::SupportFunc or VerifyAlg::Zonotope");
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
