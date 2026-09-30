// simulate - trajectories of x' = A x (+ B u), as CORA's linearSys.simulate
//
// Syntax:   x = sys.simulate(x0, timeStep, tFinal);   x = sys.simulate(x0, u, timeStep, tFinal);
// Inputs:   x0 - start points, columns (..., n, N);  timeStep, tFinal - step size and horizon
//           u - a constant input per trajectory, columns (m, N), or one (m, 1) for all
// Outputs:  x - x[k] holds the N points at time k*timeStep: e^{A k timeStep} x0 (exact); with u
//               the states of x' = A x + B u, from e^{M k timeStep} [x0; u] with M = [A B; 0 0]
// See also: simulateRandom, reach

#include "contDynamics/linearSys/private/priv.h"

#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


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

std::vector<Tensor> LinearSys::simulate(const Tensor &x0, const Tensor &u, double timeStep,
                                        double tFinal) const {
    if (!B_)
        throw std::invalid_argument("LinearSys::simulate: inputs u need B; use LinearSys(A, B)");
    const int steps = priv_numSteps(tFinal, timeStep);
    const int64_t n = A_.shape().back(), m = B_->shape().back(), N = x0.shape().back();
    if (u.shape().back() != N && u.shape().back() != 1)
        throw std::invalid_argument("LinearSys::simulate: u has one column per trajectory, or one");

    // The input is a state that stays constant, so its effect is exact in one matrix exponential.
    const Tensor top = Tensor::catLast({A_, *B_});
    const Tensor M = Tensor::catRows({top, Tensor::zeros({m, n + m}, A_.device())});
    const Tensor E = (M * timeStep).expm();
    const Tensor uN = u.shape().back() == N ? u : u.matmul(Tensor::ones({1, N}, A_.device()));
    std::vector<int64_t> stateRows(n);
    for (int64_t i = 0; i < n; ++i) stateRows[i] = i;

    Tensor z = Tensor::catRows({x0, uN});
    std::vector<Tensor> trajectory;
    for (int k = 0; k <= steps; ++k) {
        trajectory.push_back(z.transpose().selectCols(stateRows).transpose());
        z = E.matmul(z);
    }
    return trajectory;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
