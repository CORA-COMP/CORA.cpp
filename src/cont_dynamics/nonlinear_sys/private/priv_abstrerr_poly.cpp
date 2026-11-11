// priv_abstrerr_poly - the dynamic linearization error of a step with alg "poly"
//
// Rall is the set of the states over the step relative to x*, Rdiff the set of their differences
// to the start of the step, Zdelta the set at the start (priv_precomp_stat_error). The second-order
// error of the points z = Zdelta + Zdiff is 1/2 (quadMap(Zdelta, Zdiff) + quadMap(Zdiff, Zdelta) +
// quadMap(Zdiff)) with the Hessians at x*, where Zdiff is Rdiff reduced to error_order and
// extended by the input set. The third-order remainder is the box of priv_third_order_box. Their
// sum is reduced to intermediate_order; the bound is the largest |x| of its box plus that of the
// static error.
//
// Syntax:   PolyAbstractionError e = priv_abstrerr_poly(sys, Rall, Rdiff, lin, U0, stat, alg);
// Inputs:   Rall, Rdiff - polynomial zonotopes relative to lin.x_star
//           U0 - the centered input set
//           stat - the static error
//           alg - error_order and intermediate_order
// Outputs:  e - the bound (n, 1) and the zonotope of the dynamic error
// See also: priv_poly_reach, priv_precomp_stat_error, priv_abstrerr_lin
//
// CORA: contDynamics/@contDynamics/private/priv_abstrerr_poly.m
// CORA authors: Matthias Althoff, Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/nonlinear_sys/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

PolyAbstractionError priv_abstrerr_poly(const NonlinearSys &sys, const PolyZonotope &Rall,
                                        const PolyZonotope &Rdiff, const Linearized &lin,
                                        const std::optional<Zonotope> &U0,
                                        const PolyStaticError &stat,
                                        const alg::Polynomialization &alg) {
    // the boxes of the states and of the inputs
    const Interval IHx = Rall.interval();
    std::optional<Interval> IHu;
    if (U0) IHu = U0->interval();
    // the differences as a zonotope with the inputs
    const Zonotope Zred = Rdiff.zonotope().reduce(alg.error_order);
    const Zonotope Zdiff = U0 ? Zred.cart_prod(*U0) : Zred;
    // the second order: the three products of the start and the difference, and the quadratic term
    const Zonotope second = (stat.Zdelta.quad_map(Zdiff, stat.H) +
                             Zdiff.quad_map(stat.Zdelta, stat.H) + Zdiff.quad_map(stat.H)) *
                            0.5;
    const std::optional<Interval> third = priv_third_order_box(sys, IHx, IHu, lin);
    const Zonotope V = (third ? second + third->zonotope() : second).reduce(alg.intermediate_order);
    // the largest absolute value of the dynamic and the static error together
    const Interval dyn = V.interval(), fixed = stat.error.interval();
    const Tensor lo = dyn.inf() + fixed.inf(), hi = dyn.sup() + fixed.sup();
    return {lo.abs().maximum(hi.abs()), V};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
