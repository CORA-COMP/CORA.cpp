// priv_reach_poly_adaptive - the reachable sets of a nonlinear system with alg "poly-adaptive"
//
// Steps until the end time, each of a size the algorithm chooses (priv_lin_reach_poly_adaptive),
// from the set at the end of the step before. A set whose independent part is too large against the
// dependent one is restructured before its step. After a step the set at its end is reduced
// adaptively at zeta_z and then to max_dep_gen_order * n dependent generators, the one over it at
// five times zeta_z. The analysis stops, with a warning, if the step size shrinks so that the rest
// of the time span would take more than 10^9 times the last ten steps (CORALean's
// reachPolyAdaptiveFloat, CORA's reach_adaptive). CORA restructures by reducePca; restructure
// stands in for it (docs/cora-differences.md).
//
// Syntax:   PolyReachSet R = priv_reach_poly_adaptive(sys, problem, alg);
// Inputs:   problem - the initial set (one polynomial zonotope), the input set, t_final
//           alg - the tuning constants
// Outputs:  R - time_interval[k] over [time[k], time[k+1]], time_point[k] at time[k]
// See also: priv_lin_reach_poly_adaptive, NonlinearSys::reach
//
// CORA: contDynamics/@nonlinearSys/reach_adaptive.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/nonlinear_sys/private/priv_poly_adaptive.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether the step sizes have converged to 0 (CORA's aux_checkForAbortion): the remaining time
/// is more than 10^9 times the last ten steps.
bool aux_abort(const Vec &steps, double t, double t_final) {
    const std::size_t last = std::min<std::size_t>(10, steps.size());
    double sum = 0;
    for (std::size_t k = 0; k < last; ++k) sum = sum + steps[k];
    return !steps.empty() && 1e9 * sum < t_final - t;
}

/// Whether the third derivatives of f are all identically zero (CORA's thirdOrderTensorempty).
bool aux_third_order_empty(const NonlinearSys &sys, const PolyZonotope &X0,
                           const std::optional<Zonotope> &U) {
    const Tensor &x = X0.center();
    const std::optional<Range> u =
        sys.input_dim() > 0 ? std::optional<Range>(Range(U->center())) : std::nullopt;
    const auto T = sys.third_order_enclosure(Range(x), u);
    return std::none_of(T.begin(), T.end(), [](const auto &e) { return e.has_value(); });
}

} // namespace


// ===========================================  MAIN  =========================================== //

PolyReachSet priv_reach_poly_adaptive(const NonlinearSys &sys, const PolyReachProblem &problem,
                                      const alg::PolynomializationAdaptive &alg) {
    const PolyZonotope &X0 = *problem.initial_set;
    const int64_t n = sys.dim(), m = sys.input_dim();
    std::optional<Zonotope> U = problem.input_set;
    if (m > 0 && !U) U = Zonotope(X0.center().zeros_like({m, 1}), X0.center().zeros_like({m, 0}));
    const bool third_order_empty = aux_third_order_empty(sys, X0, U);

    // the state before the first step (linReachPolyAdaptiveInit)
    PolyAdaptiveState st(X0, n, 0.0);
    PolyReachSet R;
    R.time_point.push_back(X0);
    R.time.push_back(0.0);

    // the steps, each reduced for the next
    while (numerics::kHorizonEnd < problem.t_final - st.t &&
           !aux_abort(st.steps, st.t, problem.t_final)) {
        if (priv_volume_ratio_exceeds(st.R, alg.max_poly_zono_ratio)) st.R = priv_restructure(st.R);
        st = priv_lin_reach_poly_adaptive(sys, U, alg, third_order_empty, problem.t_final, st);
        st.R = priv_reduce_only_dep(priv_reduce_adaptive_poly(st.R, alg.zeta_z),
                                    alg.max_dep_gen_order);
        st.Rti = priv_reduce_adaptive_poly(st.Rti, alg.zeta_z * 5);
        R.time_interval.push_back(st.Rti);
        R.time_point.push_back(st.R);
        R.time.push_back(st.t);
    }
    return R;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
