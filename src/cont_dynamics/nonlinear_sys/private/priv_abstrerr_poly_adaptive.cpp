// priv_abstrerr_poly_adaptive - the static and the dynamic error of a step with alg "poly-adaptive"
//
// The static error Vstat is the quadratic map of the start set, reduced and with the Hessian at
// the linearization point, kept as a polynomial zonotope over the factors of the start set Rdelta
// (priv_precompStatError_adaptive); the independent generators go through the map as dependent
// ones, as CORALean's absorbRounding does. Zdelta is the zonotope of Rdelta with the input. The
// dynamic error VerrorDyn of a region Rmax with the differences Rdiff to the start of the step is
// the three second-order products carrying a difference (Zdelta with Zdiff, Zdiff with Zdelta,
// Zdiff with itself) and the Hessian remainder over the box of Rmax joined with the linearization
// point, less the Hessian at the point (CORALean's abstrErrPolyAdaptiveFloat).
//
// Syntax:   PolyStatic S = priv_precomp_stat_error_adaptive(sys, xs, U, zeta_z, Rdelta);
//           Zonotope V = priv_abstr_err_poly_adaptive(sys, xs, Rmax, Zdelta, Rdiff, U, zeta_z);
// Inputs:   xs - the linearization point, U - the input set (none: no input), zeta_z - redFactor
// Outputs:  the static error with Zdelta, and the dynamic error as a zonotope
// See also: priv_abstrerr_adaptive.cpp, priv_lin_reach_poly_adaptive_pass.cpp
//
// CORA: contDynamics/@nonlinearSys/private/priv_precompStatError_adaptive.m
// CORA: contDynamics/@nonlinearSys/private/priv_abstractionError_adaptive.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/nonlinear_sys/private/priv_poly_adaptive.h"

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

PolyStatic priv_precomp_stat_error_adaptive(const NonlinearSys &sys, const Vec &xs,
                                            const std::optional<Zonotope> &U, double red_factor,
                                            const PolyZonotope &Rdelta) {
    const Tensor &like = Rdelta.center();
    const int64_t n = sys.dim(), m = sys.input_dim();
    const double rf = std::sqrt(red_factor);
    // the reduced start set as a polynomial zonotope, and as a zonotope
    const PolyZonotope Rred = priv_reduce_adaptive_poly(Rdelta, rf);
    const Zonotope Rdz = Rdelta.zonotope().reduce_adaptive(rf);
    // the Hessians at the expansion point (xs, the center of U)
    const Vec us = m > 0 ? U->center().data() : Vec{};
    const Hessian Hp = priv_hessian(sys, like, Box{xs, xs},
                                    m > 0 ? std::optional<Box>(Box{us, us}) : std::nullopt);
    const std::vector<Tensor> Q = priv_half_hessians(like, Hp, n, n + m);
    // 1/2 quadMap(cartProd(Rred, U), H), reduced
    const Zonotope Urec =
        m > 0 ? *U - U->center() : Zonotope(like.zeros_like(), like.zeros_like({n, 0}));
    const PolyZonotope Z = m > 0 ? Rred.cart_prod(Urec) : Rred;
    const PolyZonotope second = Z.quad_map(Q);
    return {priv_reduce_adaptive_poly(second, rf), m > 0 ? Rdz.cart_prod(Urec) : Rdz};
}

Zonotope priv_abstr_err_poly_adaptive(const NonlinearSys &sys, const Vec &xs, const Zonotope &Rmax,
                                      const Zonotope &Zdelta, const PolyZonotope &Rdiff,
                                      const std::optional<Zonotope> &U, double red_factor) {
    const Tensor &like = Rmax.center();
    const int64_t n = sys.dim(), m = sys.input_dim();
    const double rf = std::sqrt(red_factor);
    // the boxes of Rmax and of U, each joined with the expansion point
    const Vec us = m > 0 ? U->center().data() : Vec{};
    const Box X = priv_hull(priv_zonotope_box(Rmax), xs);
    const std::optional<Box> Ub =
        m > 0 ? std::optional<Box>(priv_hull(priv_zonotope_box(*U), us)) : std::nullopt;
    // the differences, reduced, with the input
    const Zonotope Rred = Rdiff.zonotope().reduce_adaptive(rf);
    const Zonotope Zdiff = m > 0 ? Rred.cart_prod(*U - U->center()) : Rred;
    // the three products with a difference
    const Hessian Hp = priv_hessian(sys, like, Box{xs, xs},
                                    m > 0 ? std::optional<Box>(Box{us, us}) : std::nullopt);
    const std::vector<Tensor> Q = priv_half_hessians(like, Hp, n, n + m);
    const Zonotope second =
        (Zdelta.quad_map(Zdiff, Q) + Zdiff.quad_map(Zdelta, Q)) + Zdiff.quad_map(Q);
    // the Hessian over the region less the one at the point
    const Hessian H = priv_hessian(sys, like, X, Ub);
    const Zonotope rest =
        priv_box_zonotope(like, priv_hessian_remainder(H, Hp.lo, X, Ub, xs, us, n));
    return (second + rest).reduce_adaptive(rf);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
