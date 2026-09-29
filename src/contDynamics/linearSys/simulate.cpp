// simulate - trajectories of x' = A x, as CORA's linearSys.simulate
//
// Syntax:   x = sys.simulate(x0, timeStep, tFinal);
// Inputs:   x0 - start points, columns (..., n, N);  timeStep, tFinal - step size and horizon
// Outputs:  x - x[k] holds the N points at time k*timeStep: e^{A k timeStep} x0 (exact)
// See also: simulateRandom, reach

#include "contDynamics/linearSys/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::ct {


// ===========================================  MAIN  =========================================== //

std::vector<Tensor> LinearSys::simulate(const Tensor &x0, double timeStep, double tFinal) const {
    const int steps = priv_numSteps(tFinal, timeStep);
    const Tensor eAdt = (A_ * timeStep).expm();

    // The identity map broadcasts x0 to a batch of systems.
    Tensor x = A_.eyeLike().matmul(x0);
    std::vector<Tensor> trajectory{x};
    for (int k = 0; k < steps; ++k) {
        x = eAdt.matmul(x);
        trajectory.push_back(x);
    }
    return trajectory;
}

} // namespace cora::ct

// ---------------------------------------  END OF CODE  ---------------------------------------- //
