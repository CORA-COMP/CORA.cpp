// priv_abstrerr_lin - the linearization error of a step, as CORA's priv_abstrerr_lin
//
// R is the reachable set of the step relative to x*, z = (x, u) its states and the inputs, dz the
// box of z - z*. At tensorOrder 2 the error of component i is the Lagrange remainder
//     l_i = 1/2 |dz|' max|H_i(Z)| |dz|,
// with H_i the Hessian over the box Z of z, and the error set is the box [-l, l]. At tensorOrder 3
// the second-order term is kept as the set 1/2 quadMap(R x U, H(z*)) (R reduced to error_order),
// and the third-order remainder 1/6 sum_j dz_j (dz' T_ij(Z) dz) is added as a box; the sum is
// reduced to intermediate_order and l is the bound of its box.
//
// Syntax:   AbstractionError e = priv_abstrerr_lin(sys, R, lin, U0, alg);
// Inputs:   R - the set relative to lin.x_star, lin - the linearization, U0 - the centered input
//           set (none without inputs), alg - tensor_order, error_order and intermediate_order
// Outputs:  e - the bound l (n, 1) and the error set
// See also: priv_lin_reach, NonlinearSys::hessian_enclosure, third_order_enclosure
//
// CORA: contDynamics/@contDynamics/private/priv_abstrerr_lin.m
// CORA authors: Matthias Althoff, Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/nonlinear_sys/private/priv.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A NaN or infinite remainder fails as CORA's reachSetExplosion.
void aux_check_finite(const Tensor &l) {
    for (const double v : l.data())
        if (!std::isfinite(v))
            CORA_FAIL(Error::reach_set_explosion,
                      "cora: the Lagrange remainder exploded (the time_step may be too large)");
}

} // namespace


// ===========================================  MAIN  =========================================== //

AbstractionError priv_abstrerr_lin(const NonlinearSys &sys, const Zonotope &R,
                                   const Linearized &lin, const std::optional<Zonotope> &U0,
                                   const alg::Linearization &alg) {
    const int64_t n = sys.dim();
    // the boxes of x - x* and u - u*, and of x and u
    const Interval IHx = R.interval();
    const Range x_box(IHx.inf() + lin.x_star, IHx.sup() + lin.x_star);
    std::optional<Interval> IHu;
    std::optional<Range> u_box;
    if (U0) {
        IHu = U0->interval();
        u_box = Range(IHu->inf() + lin.u_star, IHu->sup() + lin.u_star);
    }
    if (alg.tensor_order == 2) {
        // the Hessian over the box, and the largest |z - z*| per coordinate
        std::vector<Tensor> dz = priv_entries(priv_abs_max(IHx.inf(), IHx.sup()));
        if (IHu)
            for (Tensor &e : priv_entries(priv_abs_max(IHu->inf(), IHu->sup())))
                dz.push_back(std::move(e));
        const Tensor l = priv_second_order(sys.hessian_enclosure(x_box, u_box), dz, n);
        aux_check_finite(l);
        return {l, Zonotope(l.zeros_like(), l.diag())};
    }
    // tensorOrder 3: the quadratic term as a set, the cubic remainder as a box
    const Zonotope R_red = R.reduce(alg.error_order);
    const Zonotope Z = U0 ? R_red.cart_prod(*U0) : R_red;
    const std::optional<Tensor> u_star = U0 ? std::optional<Tensor>(lin.u_star) : std::nullopt;
    const Zonotope second = Z.quad_map(sys.hessian(lin.x_star, u_star)) * 0.5;
    const std::optional<Interval> third =
        priv_third_order(sys.third_order_enclosure(x_box, u_box), priv_dz(IHx, IHu), n, false);
    const Zonotope V = (third ? second + third->zonotope() : second).reduce(alg.intermediate_order);
    const Interval box = V.interval();
    return {priv_abs_max(box.inf(), box.sup()), V};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
