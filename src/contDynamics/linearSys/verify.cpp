// verify - placeholder until the verification algorithms land
//
// Syntax:   VerifyResult r = sys.verify(params, alg, specs);
// Inputs:   params, alg, specs - see linearSys.h
// Outputs:  throws std::logic_error
// See also: reach

#include "contDynamics/linearSys/linearSys.h"

#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

VerifyResult LinearSys::verify(const VerifyParams &, VerifyAlg,
                               const std::vector<Specification> &) const {
    throw std::logic_error("verify: not implemented");
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
