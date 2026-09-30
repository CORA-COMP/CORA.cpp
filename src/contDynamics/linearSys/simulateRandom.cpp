// simulateRandom - simulations from random start points, as CORA's simulateRandom
//
// Syntax:   x = sys.simulateRandom(X0, N, timeStep, tFinal, rng);
// Inputs:   X0 - initial set (any set);  N - number of trajectories;  rng - seedable
// Outputs:  x - as simulate, from N random points of X0
// See also: simulate, ContSet::randPoint

#include "contDynamics/linearSys/linearSys.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::vector<Tensor> LinearSys::simulateRandom(const ContSet &X0, int64_t N, double timeStep,
                                              double tFinal, Rng &rng) const {
    return simulate(X0.randPoint(N, rng), timeStep, tFinal);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
