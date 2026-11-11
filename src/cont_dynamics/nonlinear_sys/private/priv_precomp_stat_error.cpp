// priv_precomp_stat_error - the static part of the error of a step with alg "poly"
//
// The Hessians H of the components at (x*, u*) map the set Z = (Rdelta reduced to error_order, 0)
// by z_i = 1/2 x' H_i x; as a polynomial zonotope the image keeps the dependency of its
// generators on the factors of the initial set. The result is reduced to intermediate_order.
// Zdelta is the zonotope of Rdelta, reduced to error_order, with the input 0. It is computed
// once per step, as the error of the step changes while the loop of priv_poly_reach runs.
//
// Syntax:   PolyStaticError s = priv_precomp_stat_error(sys, Rdelta, lin, has_inputs, alg);
// Inputs:   Rdelta - the set relative to lin.x_star
//           has_inputs - whether the system has inputs
// Outputs:  s - the Hessians, Zdelta and the static error
// See also: priv_abstrerr_poly, priv_poly_reach
//
// CORA: contDynamics/@contDynamics/private/priv_precompStatError.m
// CORA authors: Matthias Althoff, Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/nonlinear_sys/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

PolyStaticError priv_precomp_stat_error(const NonlinearSys &sys, const PolyZonotope &Rdelta,
                                        const Linearized &lin, bool has_inputs,
                                        const alg::Polynomialization &alg) {
    const int64_t n = sys.dim(), m = sys.input_dim();
    // the reduced sets, with the static input 0 as a point set of dimension m
    const PolyZonotope Rred = Rdelta.reduce(alg.error_order);
    const Zonotope Rdelta_z = Rdelta.zonotope().reduce(alg.error_order);
    const Zonotope Ustat(lin.x_star.zeros_like({m, 1}), lin.x_star.zeros_like({m, 0}));
    const PolyZonotope Z = has_inputs ? Rred.cart_prod(Ustat) : Rred;
    const Zonotope Zdelta = has_inputs ? Rdelta_z.cart_prod(Ustat) : Rdelta_z;
    // the Hessians at the expansion point, and 1/2 quadMap(Z, H)
    const std::vector<Tensor> H =
        sys.hessian(lin.x_star, has_inputs ? std::optional<Tensor>(lin.u_star) : std::nullopt);
    const PolyZonotope second = Z.quad_map(H).linear_map(lin.x_star.eye_like(n) * 0.5);
    return {H, Zdelta, second.reduce(alg.intermediate_order)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
