// priv_lin_reach - one step of a discrete-time nonlinear system, as CORA's linReach
//
// The step linearizes f at p = (center of R, center of U), x(k+1) = f(p) + A (x - p_x) + B (u -
// p_u)
// + error, where A, B are the Jacobians at p and the error is the Lagrange remainder over the box
// of the step: tensor order 2 keeps the center of the Hessians as a quadratic map of the set and
// the radius as a box; tensor order 3 maps the Hessian at p and bounds the third derivatives.
// The result is reduced to zonotope_order.
//
// Syntax:   DiscreteStep s = priv_lin_reach(sys, R, U, alg);
// Inputs:   R - the states at step k, U - the input set (none: the input 0), alg - the options
// Outputs:  s - the states at step k+1, and the error set that was added
// See also: NonlinearSysDT::post, NonlinearSys::hessian_enclosure
// References:
//    [1] M. Althoff, B. Krogh. "Reachability analysis of nonlinear differential-algebraic systems",
//        IEEE Transactions on Automatic Control 59(2), 2014 (the Lagrange remainder over an
//        interval Hessian).
//
// CORA: contDynamics/@nonlinearSysDT/linReach.m
// CORA: contDynamics/@nonlinearSysDT/private/priv_linearize.m
// CORA: contDynamics/@nonlinearSysDT/private/priv_linError_mixed_noInt.m
// CORA: contDynamics/@nonlinearSysDT/private/priv_linError_thirdOrder.m
// CORA authors: Matthias Althoff, Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/nonlinear_sys_dt/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A box (n) as the column (n, 1) the ranges are written on.
Tensor aux_col(const Tensor &bound) { return bound.expand_dims(-1); }

/// max(|inf|, |sup|) of a box, as the entries (1, 1) of dz in CORA's quadratic forms.
std::vector<Tensor> aux_abs_entries(const Interval &Ix, const std::optional<Interval> &Iu) {
    std::vector<Tensor> dz = priv_column_entries(aux_col(Ix.inf().abs().maximum(Ix.sup().abs())));
    if (Iu)
        for (Tensor &e : priv_column_entries(aux_col(Iu->inf().abs().maximum(Iu->sup().abs()))))
            dz.push_back(std::move(e));
    return dz;
}

/// The matrices (d, d) of one component of H: the 0.5 (inf + sup) and 0.5 (sup - inf) of each
/// entry, 0 where the second derivative is identically 0 (CORA's center(H{i}) and rad(H{i})).
std::pair<Tensor, Tensor> aux_mid_rad(const std::vector<std::optional<Range>> &H, int64_t i,
                                      int64_t d, const Tensor &zero) {
    std::vector<Tensor> mid_rows, rad_rows;
    for (int64_t j = 0; j < d; ++j) {
        std::vector<Tensor> mid, rad;
        for (int64_t k = 0; k < d; ++k) {
            const auto &h = H[std::size_t((i * d + j) * d + k)];
            mid.push_back(h ? h->mid() : zero);
            rad.push_back(h ? h->rad() : zero);
        }
        mid_rows.push_back(Tensor::concat(mid, -1));
        rad_rows.push_back(Tensor::concat(rad, -1));
    }
    return {Tensor::concat(mid_rows, -2), Tensor::concat(rad_rows, -2)};
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The input and the error -------------------------------------------------------------------------

std::optional<Zonotope> priv_input_set(const NonlinearSysDT &sys, const Tensor &like,
                                       const std::optional<Zonotope> &U) {
    const int64_t m = sys.input_dim();
    if (m == 0) return std::nullopt;
    if (U) return U;
    return Zonotope(like.zeros_like({m}), like.zeros_like({0, m}));
}

Zonotope priv_lin_error(const NonlinearSys &f, const Zonotope &Rdelta,
                        const std::optional<Zonotope> &U0, const Tensor &p_x, const Tensor &p_u,
                        const alg::DiscreteLinearization &alg) {
    const int64_t n = f.dim();
    // the boxes of the step relative to p, and of the step itself
    const Interval IHx = Rdelta.interval();
    std::optional<Interval> IHu;
    if (U0) IHu = U0->interval();
    // the set the Hessians map: the states reduced to error_order, with the inputs
    const Zonotope Rred = Rdelta.reduce(alg.error_order);
    const Zonotope Z = U0 ? Rred.cart_prod(*U0) : Rred;
    Zonotope error = Zonotope(p_x.zeros_like({n}), p_x.zeros_like({0, n}));
    if (alg.tensor_order == 3) {
        // the second-order term by the Hessian at p, the third-order remainder as a box
        const Zonotope second =
            Z.quad_map(
                f.hessian(p_x, f.input_dim() > 0 ? std::optional<Tensor>(p_u) : std::nullopt)) *
            0.5;
        const std::optional<Interval> third = priv_third_order_box(f, IHx, IHu, p_x, p_u);
        error = third ? second + *third : second;
    } else {
        // the Hessians over the box of the step, evaluated apart as center and radius
        const Tensor x_col = aux_col(p_x), u_col = aux_col(p_u);
        const Range x_box(aux_col(IHx.inf()) + x_col, aux_col(IHx.sup()) + x_col);
        std::optional<Range> u_box;
        if (IHu) u_box = Range(aux_col(IHu->inf()) + u_col, aux_col(IHu->sup()) + u_col);
        const std::vector<std::optional<Range>> H = f.hessian_enclosure(x_box, u_box);
        const std::vector<Tensor> dz = aux_abs_entries(IHx, IHu);
        const auto d = int64_t(dz.size());
        const Tensor dz_col = Tensor::concat(dz, -2);
        std::vector<Tensor> H_mid, radius;
        for (int64_t i = 0; i < n; ++i) {
            auto [mid, rad] = aux_mid_rad(H, i, d, dz[0].zeros_like());
            H_mid.push_back(std::move(mid));
            // 0.5 dz' H_rad dz, the row dz' H_rad first
            radius.push_back(dz_col.matrix_transpose().matmul(rad).matmul(dz_col) * 0.5);
        }
        const Tensor e = Tensor::concat(radius, -2).squeeze(-1);
        error = Z.quad_map(H_mid) * 0.5 + Interval(-e, e).zonotope();
    }
    return error.reduce(alg.zonotope_order);
}

// The step ----------------------------------------------------------------------------------------

DiscreteStep priv_lin_reach(const NonlinearSysDT &sys, const Zonotope &R,
                            const std::optional<Zonotope> &U,
                            const alg::DiscreteLinearization &alg) {
    const NonlinearSys &f = sys.state_equation();
    const int64_t n = sys.dim(), m = sys.input_dim();
    // the linearization point: the center of the states and of the inputs
    const Tensor p_x = R.center();
    const std::optional<Zonotope> Uset = priv_input_set(sys, p_x, U);
    const Tensor p_u = Uset ? Uset->center() : p_x.zeros_like({0});
    const std::optional<Tensor> u_point = m > 0 ? std::optional<Tensor>(p_u) : std::nullopt;
    const Tensor f0 = f.dynamics(p_x, u_point);
    const Jacobians J = f.jacobian(p_x, u_point);
    // the input of the affine system: B (U - p_u) around 0, and the constant f0
    std::optional<Zonotope> U0;
    Zonotope Udelta = Zonotope(f0.zeros_like({n}), f0.zeros_like({0, n}));
    if (Uset) {
        U0 = *Uset - p_u;
        Udelta = U0->linear_map(J.B);
    }
    const Zonotope Rdelta = R - p_x;
    const Zonotope Rtp = Rdelta.linear_map(J.A) + (Udelta + f0);
    // the error of the step, added and reduced
    const Zonotope error = priv_lin_error(f, Rdelta, U0, p_x, p_u, alg);
    return {(Rtp + error).reduce(alg.zonotope_order), error};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
