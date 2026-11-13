// priv_lin_loop - CORA's loop of linReach.m as CORALean's linReachLoopFloat, for any kind of set
//
// 1.1 times the error of the step before is the input [-g, g] of one linear step, the Hessian
// remainder over the zonotope of its time-interval set (joined with the expansion point x*) is the
// error, accepted when it is at most g entry by entry (floats, not their quotient), else it is the
// next guess; the sets are one step with the accepted remainder in the input. NaN repeats, fuel
// fails. The linear step is a callable from the input zonotope to (time-interval set, end set); the
// zonotope of a time-interval set is another.
//
// Syntax:   auto s = priv_lin_loop<Set>(sys, like, error, U, alg, make_step, to_zonotope);
// Inputs:   like - the center of the start set (the point of linearization), error - the bound of
//           the step before, U - the input set (uncentered, optional), make_step - the step of an
//           AdaptiveLinearization, to_zonotope - the zonotope of a time-interval set
// Outputs:  the sets over the step and at its end, and the error bound of this step
// See also: priv_lin_reach.cpp, priv_lin_reach_poly.cpp
//
// CORA: contDynamics/@contDynamics/linReach.m
// CORA authors: Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Opus 5.5, Claude Sonnet 5.5
// Written:       13-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cont_dynamics/nonlinear_sys/private/priv.h"
#include "cont_dynamics/nonlinear_sys/private/priv_adaptive.h"

#include <optional>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The sets over the step and at its end, and the error the next step starts from.
template <class Set>
struct LinLoopStep {
    Set time_interval, time_point;
    Tensor error;
};

template <class Set, class MakeStep, class ToZonotope>
LinLoopStep<Set> priv_lin_loop(const NonlinearSys &sys, const Tensor &like, const Tensor &error,
                               const std::optional<Zonotope> &U, const alg::Linearization &alg,
                               const MakeStep &make_step, const ToZonotope &to_zonotope) {
    constexpr double enlarge = 1.1; // 2476979795053773 * 2^-51
    constexpr int fuel = 10000;
    const int64_t n = sys.dim(), m = sys.input_dim();
    // without an input set the input is the origin
    const std::optional<Zonotope> Uz =
        m > 0 ? std::optional<Zonotope>(
                    U ? *U : Zonotope(like.zeros_like({m, 1}), like.zeros_like({m, 0})))
              : std::nullopt;
    // x*, A, B and f(x*, u*) + B (U - u*), and U joined with the expansion point
    const AdaptiveLinearization L = priv_linearize_adaptive(sys, Uz, like, alg.time_step);
    const Vec us = Uz ? Uz->center().data() : Vec{};
    const std::optional<Box> Ub = Uz ? std::optional<Box>(priv_zonotope_box(*Uz)) : std::nullopt;
    const std::optional<Box> Uh = Ub ? std::optional<Box>(priv_hull(*Ub, us)) : std::nullopt;
    const auto step = make_step(L);

    Vec abstrerr = error.data();
    for (int pass = 0; pass < fuel; ++pass) {
        // the applied error, and the time-interval set of the step with it in the input
        Box applied{Vec(abstrerr.size()), Vec(abstrerr.size())};
        for (std::size_t i = 0; i < abstrerr.size(); ++i) {
            applied.hi[i] = enlarge * abstrerr[i];
            applied.lo[i] = -applied.hi[i];
        }
        const Zonotope Rmax = to_zonotope(step(L.Vbase + priv_box_zonotope(like, applied)).first);
        // the Hessian remainder over Rmax joined with x*, its magnitude
        const Box Xh = priv_hull(priv_zonotope_box(Rmax), L.xs);
        const Zonotope Verror = priv_box_zonotope(
            like, priv_hessian_remainder(priv_hessian(sys, like, Xh, Uh), {}, Xh, Uh, L.xs, us, n));
        const Vec true_error = priv_norm_mag(priv_zonotope_box(Verror));
        priv_check_exploded(priv_column(like, true_error));
        bool holds = true;
        for (std::size_t i = 0; i < true_error.size(); ++i)
            holds = holds && true_error[i] <= applied.hi[i];
        if (holds) {
            // the sets with the accepted remainder in the input
            auto S = step(L.Vbase + Verror);
            return {std::move(S.first), std::move(S.second), priv_column(like, true_error)};
        }
        abstrerr = true_error;
    }
    CORA_FAIL(Error::not_converged, "cora: the error loop of the step did not converge");
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
