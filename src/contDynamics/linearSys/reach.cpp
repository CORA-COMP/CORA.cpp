// reach - the reachable sets of x' = A x (+ B u), as CORA's linearSys.reach
//
// Syntax:   R = sys.reach(X0, timeStep, tFinal, taylorTerms, linAlg);
//           R = sys.reach(X0, U, timeStep, tFinal, taylorTerms, linAlg, zonotopeOrder);
// Inputs:   X0 - initial set (zonotope);  timeStep, tFinal - step size and time horizon
//           U - set of the input u(t) in R^m, any time-varying u(t) in U (needs B)
//           taylorTerms - order of the Taylor series behind the curvature enlargement
//           linAlg - Algorithm::Standard or Algorithm::WrappingFree
//           zonotopeOrder - order the zonotopes of the standard algorithm are reduced to
// Outputs:  R - timeInt[k] encloses the states over [k*timeStep, (k+1)*timeStep];
//           timePoint[k] is the set at k*timeStep
// See also: outputSet, simulate, correctionMatrixState, correctionMatrixInput,
//           private/priv_reach_standard

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

Reach LinearSys::reach(const Zonotope &X0, const Zonotope &U, double timeStep, double tFinal,
                       int taylorTerms, Algorithm linAlg, int zonotopeOrder) const {
    if (!B_)
        throw std::invalid_argument("LinearSys::reach: a set U needs B; use LinearSys(A, B)");
    if (U.dim() != B_->shape().back())
        throw std::invalid_argument("LinearSys::reach: U needs the dimension of the columns of B");
    if (A_.shape().size() != 2 || X0.c.shape().size() != 2)
        throw std::invalid_argument("LinearSys::reach: with an input set U, one system and one X0");
    if (linAlg != Algorithm::Standard && linAlg != Algorithm::WrappingFree)
        throw std::invalid_argument(
            "LinearSys::reach: unknown linAlg; use Algorithm::Standard or Algorithm::WrappingFree");

    // The step's propagation, curvature and input solution are the same for every step.
    const int steps = priv_numSteps(tFinal, timeStep);
    const Tensor eAdt = (A_ * timeStep).expm();
    const Interval F = correctionMatrixState(timeStep, taylorTerms);
    const InputSolution in = priv_inputSolution(*this, U, timeStep, taylorTerms);

    if (linAlg == Algorithm::Standard)
        return priv_reach_standard(X0, eAdt, F, in, steps, zonotopeOrder);
    return priv_reach_wrappingfree(X0, eAdt, F, in, steps);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
