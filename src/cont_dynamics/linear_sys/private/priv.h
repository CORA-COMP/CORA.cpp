// priv - the private functions of linear_sys, as CORA's @linearSys/private
//
// Not for callers of the class: reach uses them.
//
// Syntax:   priv_reach_standard(X0, e_adt, F, steps);   priv_num_steps(t_final, time_step);
// See also: linear_sys.h, reach.cpp
//
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   09-November-2026
// Reviewed:      ---

#pragma once

#include "cont_dynamics/linear_sys/private/priv_block.h"
#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/global/tensor/tolerance.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The number of steps of size time_step that cover t_final; a slack keeps 0.3 / 0.1 at 3.
inline int priv_num_steps(double t_final, double time_step) {
    return static_cast<int>(std::ceil(t_final / time_step - numerics::kStepCount));
}

/// The weight (i^(-i/(i-1)) - i^(-1/(i-1))) dt^i / i! of the i-th Taylor term in a correction
/// matrix, given dt_over_fac = dt^i / i!; it is negative.
inline double priv_extremal_weight(int i, double dt_over_fac) {
    return (std::pow(i, -double(i) / (i - 1)) - std::pow(i, -1.0 / (i - 1))) * dt_over_fac;
}

/// The zonotope with center 0 and the generators diag(radius): the box [-radius, radius].
inline Zonotope priv_box(const Tensor &radius) { return {radius.zeros_like(), radius.diag()}; }

/// Whether M is a square matrix equal to the identity (exact comparison).
bool priv_is_identity(const Tensor &M);

/// The algorithm "standard" with inputs; reduces to zonotope_order after each step.
ReachSet priv_reach_standard(const Zonotope &X0, const Tensor &e_adt, const Interval &F,
                             const InputSolution &in, int steps, int zonotope_order);

/// The steps of "standard" with inputs one at a time, for callers that stop early (a location
/// of a hybrid automaton stops once its sets leave the invariant); priv_reach_standard loops it.
class StandardSteps {
  public:
    StandardSteps(const Zonotope &X0, const Tensor &e_adt, const Interval &F,
                  const InputSolution &in, int zonotope_order);

    /// The set at the start of the next step (X0 before the first).
    const Zonotope &time_point() const noexcept { return time_point_; }

    /// Advances one step; returns the enclosure of the step, time_point() is then its end.
    Zonotope next();

  private:
    Tensor e_adt_;
    Interval F_;
    InputSolution in_;
    int zonotope_order_, k_ = 0;
    Zonotope Htp_, Hti_, PUnext_, PU_, time_point_;
};

/// The algorithm "wrapping-free" with inputs; the time-varying part is kept as a box.
ReachSet priv_reach_wrappingfree(const Zonotope &X0, const Tensor &e_adt, const Interval &F,
                                 const InputSolution &in, int steps);

/// The algorithm "standard". e_adt = e^{A time_step}, F the correction matrix.
ReachSet priv_reach_standard(const Zonotope &X0, const Tensor &e_adt, const Interval &F, int steps);

/// The algorithm "wrapping-free".
ReachSet priv_reach_wrappingfree(const Zonotope &X0, const Tensor &e_adt, const Interval &F,
                                 int steps);

/// The algorithms "from_start" and "decomposition": each step is the first one mapped forward by
/// e^{A dt}^(k-1), per block of the partition `blocks` (one block: from_start). The input set,
/// if any, is split by canonical_form into the constant input and the centered set.
ReachSet priv_reach_from_start(const LinearSys &sys, const Zonotope &X0,
                               const std::optional<Zonotope> &input_set, double time_step,
                               int taylor_terms, int zonotope_order, const Blocks &blocks,
                               int steps);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
