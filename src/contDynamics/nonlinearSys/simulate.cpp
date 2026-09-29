// simulate - trajectories of x' = f(x), as CORA's nonlinearSys.simulate
//
// Syntax:   x = sys.simulate(x0, timeStep, tFinal);
// Inputs:   x0 - start points, columns (n, N);  timeStep, tFinal - step size and horizon
// Outputs:  x - x[k] holds the N points at time k*timeStep, from the classical Runge-Kutta method
//           with steps of at most 1e-3
// See also: simulateRandom, reach

#include "contDynamics/nonlinearSys/nonlinearSys.h"
#include "contDynamics/linearSys/private/priv.h"

#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::ct {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using State = std::vector<double>;

/// x + s * k, elementwise.
State aux_shifted(const State &x, const State &k, double s) {
    State y(x);
    for (std::size_t i = 0; i < y.size(); ++i) y[i] += s * k[i];
    return y;
}

/// One step of size h of the classical Runge-Kutta method.
State aux_rungeKutta(const NonlinearSys &sys, const State &x, double h) {
    const State k1 = sys.values(x);
    const State k2 = sys.values(aux_shifted(x, k1, h / 2));
    const State k3 = sys.values(aux_shifted(x, k2, h / 2));
    const State k4 = sys.values(aux_shifted(x, k3, h));
    State y(x);
    for (std::size_t i = 0; i < y.size(); ++i)
        y[i] += h / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
    return y;
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::vector<Tensor> NonlinearSys::simulate(const Tensor &x0, double timeStep, double tFinal) const {
    const std::vector<int64_t> shape = x0.shape();
    if (shape.size() != 2 || shape[0] != n_)
        throw std::invalid_argument("cora: the start points of a nonlinear system are columns (" +
                                    std::to_string(n_) + ", N); batches are not supported");
    if (timeStep <= 0) throw std::invalid_argument("cora: the time step must be positive");
    const int64_t N = shape[1];
    const int steps = priv_numSteps(tFinal, timeStep);
    const int substeps = static_cast<int>(std::ceil(timeStep / 1e-3 - 1e-9));
    const double h = timeStep / substeps;

    // The points as N columns of n numbers, advanced one by one.
    const std::vector<double> data = x0.data();
    std::vector<State> points(N, State(n_));
    for (int64_t j = 0; j < N; ++j)
        for (int64_t i = 0; i < n_; ++i) points[j][i] = data[i * N + j];

    const auto snapshot = [&] {
        std::vector<double> out(n_ * N);
        for (int64_t j = 0; j < N; ++j)
            for (int64_t i = 0; i < n_; ++i) out[i * N + j] = points[j][i];
        return Tensor::like(x0, out, {n_, N});
    };

    std::vector<Tensor> trajectory{x0};
    for (int k = 0; k < steps; ++k) {
        for (State &p : points)
            for (int s = 0; s < substeps; ++s) p = aux_rungeKutta(*this, p, h);
        trajectory.push_back(snapshot());
    }
    return trajectory;
}

} // namespace cora::ct

// ---------------------------------------  END OF CODE  ---------------------------------------- //
