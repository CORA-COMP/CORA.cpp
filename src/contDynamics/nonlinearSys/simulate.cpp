// simulate - trajectories of x' = f(x), as CORA's nonlinearSys.simulate
//
// Syntax:   x = sys.simulate(x0, timeStep, tFinal);
// Inputs:   x0 - start points, columns (n, N);  timeStep, tFinal - step size and horizon
// Outputs:  x - x[k] holds the N points at time k*timeStep, from the classical Runge-Kutta method
//           with steps of at most 1e-3
// See also: simulateRandom, reach

#include "contDynamics/nonlinearSys/nonlinearSys.h"
#include "contDynamics/linearSys/private/priv.h"

#include <cmath>
#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::ct {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// One step of size h of the classical Runge-Kutta method, for all columns of x at once.
Tensor aux_rungeKutta(const NonlinearSys &sys, const Tensor &x, double h) {
    const Tensor k1 = sys.dynamics(x);
    const Tensor k2 = sys.dynamics(x + k1 * (h / 2));
    const Tensor k3 = sys.dynamics(x + k2 * (h / 2));
    const Tensor k4 = sys.dynamics(x + k3 * h);
    return x + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (h / 6);
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::vector<Tensor> NonlinearSys::simulate(const Tensor &x0, double timeStep, double tFinal) const {
    const std::vector<int64_t> shape = x0.shape();
    if (shape.size() != 2 || shape[0] != n_)
        throw std::invalid_argument("cora: the start points of a nonlinear system are columns (" +
                                    std::to_string(n_) + ", N); batches are not supported");
    if (timeStep <= 0) throw std::invalid_argument("cora: the time step must be positive");
    const int steps = priv_numSteps(tFinal, timeStep);
    const int substeps = static_cast<int>(std::ceil(timeStep / 1e-3 - 1e-9));
    const double h = timeStep / substeps;

    std::vector<Tensor> trajectory{x0};
    Tensor x = x0;
    for (int k = 0; k < steps; ++k) {
        for (int s = 0; s < substeps; ++s) x = aux_rungeKutta(*this, x, h);
        trajectory.push_back(x);
    }
    return trajectory;
}

} // namespace cora::ct

// ---------------------------------------  END OF CODE  ---------------------------------------- //
