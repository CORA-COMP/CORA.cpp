// priv - the private functions of linearSys, as CORA's @linearSys/private
//
// Not for callers of the class: reach uses them.
//
// Syntax:   priv_reach_standard(X0, eAdt, F, steps);   priv_numSteps(tFinal, timeStep);
//           priv_inputSolution(sys, U, timeStep, taylorTerms);
// See also: linearSys.h, reach.cpp

#pragma once

#include "contDynamics/linearSys/linearSys.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The number of steps of size timeStep that cover tFinal; a tolerance keeps 0.3 / 0.1 at 3.
inline int priv_numSteps(double tFinal, double timeStep) {
    return static_cast<int>(std::ceil(tFinal / timeStep - 1e-9));
}

/// The weight (i^(-i/(i-1)) - i^(-1/(i-1))) dt^i / i! of the i-th Taylor term in a correction
/// matrix, given dtOverFac = dt^i / i!; it is negative.
inline double priv_extremalWeight(int i, double dtOverFac) {
    return (std::pow(i, -double(i) / (i - 1)) - std::pow(i, -1.0 / (i - 1))) * dtOverFac;
}

/// The error box [-W, W] of the series sum_{i<=taylorTerms} |A|^i dt^i / i! against e^{|A| dt},
/// as CORA's priv_expmRemainder; multiplied with dt it bounds the truncated series of the input.
Interval priv_expmRemainder(const Tensor &A, double timeStep, int taylorTerms);

/// What the input u(t) in U adds over one step (CORA's particularSolution_timeVarying and
/// _constant, and the curvature of the input). With U = c + U0 (U0 centered): the constant part
/// is B c and the time-varying part B U0.
struct InputSolution {
    Zonotope PU;     ///< time-varying solution at t = timeStep, centered at the origin
    Tensor Pu;       ///< constant-input solution at t = timeStep (n, 1)
    Zonotope Cinput; ///< curvature enlargement of the constant input
};

InputSolution priv_inputSolution(const LinearSys &sys, const Zonotope &U, double timeStep,
                                 int taylorTerms);

/// The zonotope with center 0 and the generators diag(radius): the box [-radius, radius].
inline Zonotope priv_box(const Tensor &radius) { return {radius.zerosLike(), radius.diag()}; }

/// The algorithm "standard" with inputs; reduces to zonotopeOrder after each step.
Reach priv_reach_standard(const Zonotope &X0, const Tensor &eAdt, const Interval &F,
                          const InputSolution &in, int steps, int zonotopeOrder);

/// The algorithm "wrapping-free" with inputs; the time-varying part is kept as a box.
Reach priv_reach_wrappingfree(const Zonotope &X0, const Tensor &eAdt, const Interval &F,
                              const InputSolution &in, int steps);

/// The algorithm "standard". eAdt = e^{A timeStep}, F the correction matrix.
Reach priv_reach_standard(const Zonotope &X0, const Tensor &eAdt, const Interval &F, int steps);

/// The algorithm "wrapping-free".
Reach priv_reach_wrappingfree(const Zonotope &X0, const Tensor &eAdt, const Interval &F, int steps);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
