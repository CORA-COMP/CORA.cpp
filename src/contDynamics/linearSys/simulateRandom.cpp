// simulateRandom - simulations from random start points, as CORA's simulateRandom
//
// Syntax:   x = sys.simulateRandom(X0, N, timeStep, tFinal, rng);
//           x = sys.simulateRandom(X0, U, N, timeStep, tFinal, rng);
// Inputs:   X0 - initial set (any set);  U - input set (any set);  N - number of trajectories
//           rng - seedable
// Outputs:  x - as simulate, from N random points of X0 (and N random constant inputs of U)
// See also: simulate, ContSet::randPoint

#include "contDynamics/linearSys/linearSys.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::vector<Tensor> LinearSys::simulateRandom(const ContSet &X0, int64_t N, double timeStep,
                                              double tFinal, Rng &rng) const {
    return simulate(X0.randPoint(N, rng), timeStep, tFinal);
}

std::vector<Tensor> LinearSys::simulateRandom(const ContSet &X0, const ContSet &U, int64_t N,
                                              double timeStep, double tFinal, Rng &rng) const {
    const Tensor x0 = X0.randPoint(N, rng);
    return simulate(x0, U.randPoint(N, rng), timeStep, tFinal);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
