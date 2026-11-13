// priv_closed_form_step - CORA's oneStep in closed form, as CORALean's oneStepClosedFormFloat
//
// One step of x' = A x + u, u in U + u0, from a set X: the sets at its end (Rtp) and over it (Rti)
// of CORA's oneStep, with the matrix exponential, the correction matrices [F] and [G] and the
// input solutions as interval matrices. [F], [G] and the input solutions carry the entrywise
// remainder W, the enclosure of e^{|A| dt} at the order eta_e + 8 less its series at the
// admissible order eta_e >= eta that every sum runs to (CORALean !917); [e^{A dt}] is the
// order-16 enclosure at A dt / 2^s squared s times (s from priv_closed_form_halvings), tight also
// for a stiff A. The order and the rounding are those of CORALean's float definition (a Taylor sum
// with the newest term added to the running sum, powers by left multiplication, sums nested to the
// right); the matrices are computed once per system and step size, the steps of one linearization
// differ in the input and the start set only. The stepper is not admissible when no order below eta
// + 4096 encloses e^{|A| dt}, or no s < 128 halves A dt to a norm of at most 1/2.
//
// Syntax:   const ClosedFormStepper stepper(A, time_step, taylor_terms);
//           auto S = stepper.step(X, U, u);
// Inputs:   A - state matrix (n, n)
//           time_step - step size
//           taylor_terms - order of the series eta (raised to the admissible order eta_e)
//           X - start set
//           U - time-varying input set
//           u - constant input (n, 1)
// Outputs:  S - time_point (Rtp) and time_interval (Rti)
// See also: priv_float_step.h (the step of reachFloatInput), LinearSys::one_step
//
// CORA.cpp only: CORA evaluates oneStep with expm and A^-1; this is its closed form in
// CORALean's float definition (LinearSys.oneStepClosedFormFloat).
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   13-November-2026
// Reviewed:      ---

#pragma once

#include "cora/cont_set/zonotope/zonotope.h"

#include <memory>
#include <optional>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The sets of a closed-form step: at its end and over it.
struct ClosedFormStep {
    Zonotope time_point, time_interval;
};

/// The first order from taylor_terms (within 4096) with ||dt |A|||, bounded as CORALean's
/// normUpperBound, at most (order + 2) / 2; none if there is none.
std::optional<int> priv_closed_form_order(const Tensor &A, double time_step, int taylor_terms);

/// The first count s below 128 with ||dt A / 2^s||, bounded as CORALean's normUpperBound, at most
/// 1/2 (LinearSys.halvingsFloat); none if there is none.
std::optional<int> priv_closed_form_halvings(const Tensor &A, double time_step);

/// The entrywise remainder and corrections of CORALean's float definition at the admissible order.
struct EntrywiseCorrections {
    int order;     ///< eta_e: the first order from taylor_terms that admits the enclosure
    Tensor W;      ///< |e^{|A| dt} - series| bound, entry by entry
    Interval F, G; ///< the state and the constant-input correction, widened by W and dt W
};

/// MatExp.expRemainderEntrywise, intervalCorrectionMatrixEntrywise and
/// intervalConstCorrectionMatrixEntrywise of the point matrix A; none without an admissible order.
std::optional<EntrywiseCorrections> priv_entrywise_corrections(const Tensor &A, double time_step,
                                                               int taylor_terms);

/// priv_entrywise_corrections where it is asked for (CORACPP_TAYLOR=entrywise), for an (n, n)
/// matrix without gradient whose matrices are all finite; none otherwise (CORA's expm path
/// computes them). The last results are kept.
std::optional<EntrywiseCorrections> priv_taylor_corrections(const Tensor &A, double time_step,
                                                            int taylor_terms);

/// The matrices of the closed-form step of one system and step size.
class ClosedFormStepper {
  public:
    /// The matrices for A, time_step and taylor_terms; admissible() is false without an order.
    ClosedFormStepper(const Tensor &A, double time_step, int taylor_terms);
    ~ClosedFormStepper();
    ClosedFormStepper(ClosedFormStepper &&) noexcept;
    bool admissible() const;
    /// One step from X under the input set U and the constant input u (n, 1).
    ClosedFormStep step(const Zonotope &X, const Zonotope &U, const Tensor &u) const;

  private:
    struct Matrices;
    std::unique_ptr<Matrices> m_;
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
