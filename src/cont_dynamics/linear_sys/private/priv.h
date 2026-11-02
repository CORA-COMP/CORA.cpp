// priv - the private functions of linear_sys, as CORA's @linearSys/private
//
// Not for callers of the class: reach uses them.
//
// Syntax:   priv_reach_standard(X0, e_adt, F, steps);   priv_num_steps(t_final, time_step);
// See also: linear_sys.h, reach.cpp

#pragma once

#include "cora/cont_dynamics/linear_sys/linear_sys.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The number of steps of size time_step that cover t_final; a tolerance keeps 0.3 / 0.1 at 3.
inline int priv_num_steps(double t_final, double time_step) {
    return static_cast<int>(std::ceil(t_final / time_step - 1e-9));
}

/// The weight (i^(-i/(i-1)) - i^(-1/(i-1))) dt^i / i! of the i-th Taylor term in a correction
/// matrix, given dt_over_fac = dt^i / i!; it is negative.
inline double priv_extremal_weight(int i, double dt_over_fac) {
    return (std::pow(i, -double(i) / (i - 1)) - std::pow(i, -1.0 / (i - 1))) * dt_over_fac;
}

/// The zonotope with center 0 and the generators diag(radius): the box [-radius, radius].
inline Zonotope priv_box(const Tensor &radius) { return {radius.zeros_like(), radius.diag()}; }

/// The algorithm "standard" with inputs; reduces to zonotope_order after each step.
ReachSet priv_reach_standard(const Zonotope &X0, const Tensor &e_adt, const Interval &F,
                             const InputSolution &in, int steps, int zonotope_order);

/// The algorithm "wrapping-free" with inputs; the time-varying part is kept as a box.
ReachSet priv_reach_wrappingfree(const Zonotope &X0, const Tensor &e_adt, const Interval &F,
                                 const InputSolution &in, int steps);

/// The algorithm "standard". e_adt = e^{A time_step}, F the correction matrix.
ReachSet priv_reach_standard(const Zonotope &X0, const Tensor &e_adt, const Interval &F, int steps);

/// The algorithm "wrapping-free".
ReachSet priv_reach_wrappingfree(const Zonotope &X0, const Tensor &e_adt, const Interval &F,
                                 int steps);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
