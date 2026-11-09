// post - the next step of the algorithm "standard" or "wrapping_free", as CORA's post
//
// Maps the homogeneous sets of the state by e^{A dt} and adds the constant input; the input
// solutions accumulate in par (summed and reduced, or as a box for the wrapping-free algorithm).
//
// Syntax:   ReachStep next = sys.post(step.state, alg::Standard{.time_step = 0.1});
// Inputs:   state - the state of init_reach or of the previous post
//           alg - the algorithm and options of init_reach
// Outputs:  next - time_interval and time_point of the step, and the state for the next
// See also: init_reach, reach
//
// CORA: contDynamics/@linearSys/post.m
// CORA authors: Mark Wetzlinger, Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_steps.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The box of the sum of the boxes of a, b and c, shifted by -shift: CORA's interval arithmetic.
Zonotope aux_box_sum(const Zonotope &a, const Zonotope &b, const Zonotope &c, const Tensor &shift) {
    return (a.interval() + b.interval() + c.interval() - shift).zonotope();
}

} // namespace


// ===========================================  MAIN  =========================================== //

ReachStep LinearSys::post(const ReachState &state, const alg::LinearReach &alg) const {
    detail::Entry entry;
    const StepAlgorithm o = priv_step_algorithm("post", alg);
    if (entry.check()) std::visit([](const auto &a) { a.validate(); }, alg);
    if (entry.check()) validate_step("post", state.homogeneous, o.time_step, o.taylor_terms);
    CORA_CHECK(A().ndim() == 2, Error::not_supported, [] {
        return std::string("post: it supports a single system, not a batch of matrices A");
    });
    TaylorLinSys taylor(A());
    const Tensor e_at = taylor.e_adt(o.time_step);
    const Zonotope &trans = state.trans;
    Zonotope hom = state.homogeneous.linear_map(e_at),
             hom_tp = state.homogeneous_time_point.linear_map(e_at);
    const Zonotope aux = state.aux.linear_map(e_at);
    Zonotope par = state.par;
    if (o.wrapping_free) {
        // only the center of the constant part is propagated, its width joins the box
        hom = hom + trans.center();
        hom_tp = hom_tp + trans.center();
        par = aux_box_sum(state.par, aux, trans, trans.center());
    } else {
        hom = hom.plus(trans).reduce(o.zonotope_order);
        hom_tp = hom_tp.plus(trans).reduce(o.zonotope_order);
        par = state.par.plus(aux).reduce(o.zonotope_order);
    }
    return {hom.plus(par), hom_tp.plus(par), {hom, hom_tp, aux, par, trans}};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
