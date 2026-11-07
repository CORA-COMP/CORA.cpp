// init_reach - the first step of the algorithm "standard" or "wrapping_free", as CORA's initReach
//
// From R_init it computes the solution of the first step with the input solution, and the state
// that post continues from: the homogeneous sets, the input solution of the step, the sum of the
// input solutions so far and the constant part of the input. CORA's originContained is false.
//
// Syntax:   ReachStep s = sys.init_reach(R_init, U, u_trans, alg::Standard{.time_step = 0.1});
// Inputs:   R_init - initial set (zonotope of the dimension of A)
//           U - input set before B (zonotope of the dimension of the columns of B), or none
//           u_trans - constant input (m, 1), or none (both need B)
//           alg - alg::Standard or alg::WrappingFree
// Outputs:  s - time_interval and time_point of the first step, and the state for post
// See also: post, one_step, reach
//
// CORA: contDynamics/@linearSys/initReach.m
// CORA authors: Mark Wetzlinger, Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_steps.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The input solution of the first step: sum_i A^i dt^(i+1)/(i+1)! V and the matrix of the sum.
struct Solution {
    std::optional<Zonotope> sum;
    Tensor integral;
};

Solution aux_input_sum(const LinearSys &sys, TaylorLinSys &taylor, const std::optional<Zonotope> &V,
                       double time_step, int taylor_terms) {
    Solution s{std::nullopt, sys.A().eye_like() * time_step};
    if (V) s.sum = *V * time_step;
    for (int i = 1; i <= taylor_terms; ++i) {
        const Tensor term = taylor.Apower(i) * taylor.dtoverfac(time_step, i + 1);
        if (V) s.sum = s.sum->plus(V->linear_map(term));
        s.integral = s.integral + term;
    }
    return s;
}

} // namespace


// ===========================================  MAIN  =========================================== //

ReachStep LinearSys::init_reach(const Zonotope &R_init, const std::optional<Zonotope> &U,
                                const std::optional<Tensor> &u_trans,
                                const alg::LinearReach &alg) const {
    detail::Entry entry;
    // the algorithm first, then its options and the arguments against the system
    const StepAlgorithm o = priv_step_algorithm("init_reach", alg);
    if (entry.check()) std::visit([](const auto &a) { a.validate(); }, alg);
    if (entry.check()) validate_step("init_reach", R_init, o.time_step, o.taylor_terms);
    CORA_CHECK(!(U || u_trans) || B_.has_value(), Error::wrong_value,
               [] { return std::string("init_reach: an input needs B; use LinearSys(A, B)"); });
    CORA_CHECK((!U || U->dim() == B_->shape().back()) &&
                   (!u_trans || u_trans->shape() == std::vector<int64_t>({B_->shape().back(), 1})),
               Error::dimension_mismatch, [&] {
                   return std::string("init_reach: U and u_trans must have the dimension of the "
                                      "columns of B");
               });
    CORA_CHECK(A_.ndim() == 2, Error::not_supported, [] {
        return std::string("init_reach: it supports a single system, not a batch of matrices A");
    });
    // the matrices of the step: the remainder and curvature of the series, and e^{A dt}
    TaylorLinSys taylor(A_);
    const double dt = o.time_step;
    const Interval E = expm_remainder(dt, o.taylor_terms);
    const Interval F = correction_matrix_state(dt, o.taylor_terms);
    const Tensor e_at = taylor.e_adt(dt);

    // the input solution: V = B U varies in time, vTrans = B u_trans is constant
    std::optional<Zonotope> V;
    if (U) V = U->linear_map(*B_);
    const bool has_V = V && !V->representsa("origin");
    const Solution s = aux_input_sum(*this, taylor, has_V ? V : std::nullopt, dt, o.taylor_terms);
    const Zonotope origin{A_.zeros_like({dim(), 1}), A_.zeros_like({dim(), 0})};
    const Zonotope RV =
        has_V ? s.sum->plus(V->linear_map(dt * E)) : origin; // the series and its remainder
    const Interval e_at_int{s.integral + E.inf() * dt, s.integral + E.sup() * dt};
    std::optional<Zonotope> trans_point;
    if (u_trans) trans_point = Zonotope(B_->matmul(*u_trans), A_.zeros_like({dim(), 0}));
    const Zonotope R_trans = trans_point ? trans_point->linear_map(e_at_int) : origin;

    // the first step: the end of the step, then the enclosure with the curvatures
    Zonotope Rhom_tp = R_init.linear_map(e_at).plus(R_trans);
    Zonotope Rhom = R_init.enclose(Rhom_tp).plus(R_init.linear_map(F));
    if (trans_point)
        Rhom = Rhom.plus(trans_point->linear_map(correction_matrix_input(dt, o.taylor_terms)));
    Rhom = Rhom.reduce(o.zonotope_order);
    Rhom_tp = Rhom_tp.reduce(o.zonotope_order);
    const Zonotope RV_reduced = has_V ? RV.reduce(o.zonotope_order) : RV;
    // the wrapping-free algorithm accumulates the input solutions as a box
    const Zonotope Rpar = o.wrapping_free ? RV_reduced.interval().zonotope() : RV_reduced;
    return {Rhom.plus(RV_reduced),
            Rhom_tp.plus(RV_reduced),
            {Rhom, Rhom_tp, RV_reduced, Rpar, R_trans}};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
