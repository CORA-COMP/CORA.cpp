// priv_float_step - the linear step of the nonlinear algorithms, as CORALean's reachFloatInput
//
// One step of x' = A (x - x*) + v, v in V, from a set R: the interval-matrix exponential [e^{A
// dt}], the correction matrix [F] and the input solutions P of CORALean's float definitions, in the
// order and with the rounding that definition fixes (the sums nested to the right), and
// CORALean's Taylor order: the first order from max(eta, 2) with ||dt A|| <= (eta + 2) / 2.
// The set of the window is H(tau) + P^R and the set at its end H(t_1) + P^R, H(t_1) reduced.
//
// Syntax:   auto S = priv_float_one_step(A, xs, V, R, time_step, taylor_terms, zonotope_order);
// Inputs:   A - state matrix (n, n)
//           xs - point of linearization (n, 1)
//           V - input set of the state space, its center the constant input
//           R - set at the start of the step
//           time_step - step size
//           taylor_terms - least Taylor order
//           zonotope_order - order of the reduction
// Outputs:  S - the window and the end set, none if no admissible order is found
// See also: priv_steps.h, LinearSys::reach
//
// CORA.cpp only: CORA's oneStep puts the input curvature in the window; the step follows
// CORALean's LinearSys.reachFloatInput and NonlinearSys.oneStepFloat.
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/zonotope/zonotope.h"

#include <memory>
#include <optional>
#include <utility>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The window and the set at the end of a step.
using FloatStep = std::pair<Zonotope, Zonotope>;

/// The first Taylor order from max(taylor_terms, 2) with ||time_step A|| <= (order + 2) / 2, the
/// norm bounded as CORALean's normUpperBound; none within 4096 orders.
std::optional<int> priv_float_order(const Tensor &A, double time_step, int taylor_terms);

/// The matrices of a step of one system and length (the powers of A, [e^{A dt}], [F]), computed
/// once for the steps of a linearization that differ in the input and the start set.
class FloatStepper {
  public:
    /// The matrices for A and time_step; admissible() is false if no Taylor order is found.
    FloatStepper(const Tensor &A, double time_step, int taylor_terms);
    ~FloatStepper();
    FloatStepper(FloatStepper &&) noexcept;
    bool admissible() const;
    /// One step of the linearization at xs from R under the input set V (see the header).
    FloatStep step(const Tensor &xs, const Zonotope &V, const Zonotope &R,
                   int zonotope_order) const;

  private:
    struct Matrices;
    std::unique_ptr<Matrices> m_;
};

/// One step of the linearization at xs from R under the input set V (see the header).
std::optional<FloatStep> priv_float_one_step(const Tensor &A, const Tensor &xs, const Zonotope &V,
                                             const Zonotope &R, double time_step, int taylor_terms,
                                             int zonotope_order);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
