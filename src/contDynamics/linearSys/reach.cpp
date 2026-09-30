// reach - the reachable sets of x' = A x, as CORA's linearSys.reach
//
// Syntax:   R = sys.reach(X0, timeStep, tFinal, taylorTerms, linAlg);
// Inputs:   X0 - initial set (zonotope);  timeStep, tFinal - step size and time horizon
//           taylorTerms - order of the Taylor series behind the curvature enlargement
//           linAlg - Algorithm::Standard or Algorithm::WrappingFree
// Outputs:  R - timeInt[k] encloses the states over [k*timeStep, (k+1)*timeStep];
//           timePoint[k] is the set at k*timeStep
// See also: simulate, correctionMatrixState, private/priv_reach_standard

#include "contDynamics/linearSys/private/priv.h"

#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Reach LinearSys::reach(const Zonotope &X0, double timeStep, double tFinal, int taylorTerms,
                       Algorithm linAlg) const {
    const int steps = priv_numSteps(tFinal, timeStep);
    const Tensor eAdt = (A_ * timeStep).expm();
    const Interval F = correctionMatrixState(timeStep, taylorTerms);

    if (linAlg == Algorithm::Standard) return priv_reach_standard(X0, eAdt, F, steps);
    if (linAlg == Algorithm::WrappingFree) return priv_reach_wrappingfree(X0, eAdt, F, steps);
    throw std::invalid_argument(
        "LinearSys::reach: unknown linAlg; use Algorithm::Standard or Algorithm::WrappingFree");
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
