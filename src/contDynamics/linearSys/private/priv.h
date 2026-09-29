// The private functions of linearSys, as CORA's @linearSys/private: not for callers of the class.

#pragma once

#include "contDynamics/linearSys/linearSys.h"

#include <cmath>

namespace cora::ct {

/// The number of steps of size timeStep that cover tFinal; a tolerance keeps 0.3 / 0.1 at 3.
inline int priv_numSteps(double tFinal, double timeStep) {
    return static_cast<int>(std::ceil(tFinal / timeStep - 1e-9));
}

/// The algorithm "standard". eAdt = e^{A timeStep}, F the correction matrix.
Reach priv_reach_standard(const Zonotope &X0, const Tensor &eAdt, const Interval &F, int steps);

/// The algorithm "wrapping-free".
Reach priv_reach_wrappingfree(const Zonotope &X0, const Tensor &eAdt, const Interval &F, int steps);

} // namespace cora::ct
