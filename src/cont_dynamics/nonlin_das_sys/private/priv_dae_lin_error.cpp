// priv_dae_lin_error - the linearization error of a step of a DAE, as priv_linError_mixed_noInt
//
// R is the set of the step relative to x*; the algebraic variables relative to y* are the set
// R_y = -F^-1 (g0 + D R + E U + V_y), which keeps what depends on R (R_y_cor, the factors of R)
// apart from the assumed algebraic error and the inputs (R_y_add, new factors). The error of the
// second order is the Lagrange remainder 1/2 dz' H_i dz of every row of f and g, with H_i the
// Hessian over the box of z = (x, y, u) joined with the expansion point, its midpoint on the set
// (x, y, u) (quadMap) and its radius on the box |dz|. Solving the linearized g for y turns the
// error of g, l_g, into -C F^-1 l_g in the flow [Althoff, Krogh, TAC 2014, Cor. 1]: CORA adds it (+
// C F^-1) and CORALean's linErrorCora counterexample shows it misses the error; the sign is
// corrected here. With a decomposition the remainder of each subsystem is evaluated on its own
// equations and variables (CORA's priv_linError_mixed_noInt_comp, [Althoff, TPS 2014, Sec. V-B]),
// the parts are summed over the states and constraints they own, and -C F^-1 is that of the whole
// system, not each subsystem's own block (CORALean's compLocalErrorCora counterexample).
//
// Syntax:   DaeError e = priv_dae_lin_error(sys, lin, R, V_y, U0, decomposition, alg);
// Inputs:   sys, lin - the system and its linearization, R - the set relative to lin.x_star,
//           V_y - the assumed error of g, U0 - the centered input set, decomposition - or none
// Outputs:  e - the error set and its bounds, the algebraic variables over the step
// See also: priv_dae_lin_reach, NonlinDASys::hessian_enclosure
//
// CORA: contDynamics/@nonlinDASys/private/priv_linError_mixed_noInt.m
// CORA: contDynamics/@nonlinDASys/private/priv_linError_mixed_noInt_comp.m
// CORA authors: Matthias Althoff, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/nonlin_das_sys/private/priv.h"
#include "cont_dynamics/nonlinear_sys/private/priv.h"

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The remainder of one system ---------------------------------------------------------------------

/// The second-order remainders of f and g, each as the midpoint set and the radius box: half of
/// quadMap(R_xyu, H_mid) and of |dz|' H_rad |dz| per row (CORA's error_x_mid, error_x_rad, ...).
struct Remainder {
    Zonotope x_mid, y_mid;
    Tensor x_rad, y_rad;
};

/// The box [-radius, radius] as a zonotope (CORA's zonotope(interval(-r, r))).
Zonotope aux_box(const Tensor &radius) { return Zonotope(radius.zeros_like(), radius.diag()); }

/// The bound max(|inf|, |sup|) of the box of a set, joined with the origin by the caller.
Tensor aux_abs_max(const Interval &I) { return I.inf().abs().maximum(I.sup().abs()); }

/// The bound sup |interval(Z)| of a set: the largest absolute value per coordinate.
Tensor aux_bound(const Zonotope &Z) { return aux_abs_max(Z.interval()); }

/// The matrices of the midpoints and the values 1/2 dz' H_rad dz of the rows (first, first + count)
/// of the second derivatives over a box, d variables; the zero entries stay zero.
void aux_rows(const std::vector<std::optional<Range>> &H, int64_t count, int64_t d,
              const std::vector<double> &dz, const Tensor &like, std::vector<Tensor> &mid,
              std::vector<double> &quad) {
    for (int64_t i = 0; i < count; ++i) {
        std::vector<double> m(std::size_t(d * d), 0.0);
        double q = 0;
        for (int64_t j = 0; j < d; ++j)
            for (int64_t k = 0; k < d; ++k) {
                const auto &h = H[std::size_t((i * d + j) * d + k)];
                if (!h) continue;
                const double lo = h->lo.data()[0], hi = h->hi.data()[0];
                m[std::size_t(j * d + k)] = 0.5 * (lo + hi);
                q += dz[std::size_t(j)] * (0.5 * (hi - lo)) * dz[std::size_t(k)];
            }
        mid.push_back(Tensor::like(like, m, {d, d}));
        quad.push_back(0.5 * q);
    }
}

/// The remainder of the system `sys` (the whole one or a subsystem) over the sets Rx of its
/// states, Ry_cor + Ry_add of its algebraic variables and Uz of its inputs, all relative to the
/// point (p_x, p_y, p_u): CORA's body of priv_linError_mixed_noInt for one system.
Remainder aux_remainder(const NonlinDASys &sys, const Tensor &p_x, const Tensor &p_y,
                        const Tensor &p_u, const Zonotope &Rx, const Zonotope &Ry_cor,
                        const Zonotope &Ry_add, const std::optional<Zonotope> &Uz,
                        int error_order) {
    const int64_t n = sys.dim(), p = sys.constraint_dim(), m = sys.input_dim(), d = n + p + m;
    // the boxes of the three sets joined with the point; the Hessian is read over their shifts
    const Interval dx = priv_with_origin(Rx.interval());
    const Interval dy = priv_with_origin((Ry_cor + Ry_add).interval());
    std::optional<Interval> du;
    if (m > 0) du = priv_with_origin(Uz->interval());
    const DaeHessians H = sys.hessian_enclosure(
        Range(dx.inf() + p_x, dx.sup() + p_x), Range(dy.inf() + p_y, dy.sup() + p_y),
        du ? Range(du->inf() + p_u, du->sup() + p_u) : Range(p_u));
    // the set of (x, y, u) of CORA: the center of the added y is one more generator of y
    std::vector<Tensor> y_gens{Ry_cor.generators()};
    const std::vector<double> add_c = Ry_add.center().data();
    if (std::any_of(add_c.begin(), add_c.end(), [](double v) { return v != 0.0; }))
        y_gens.push_back(Ry_add.center());
    y_gens.push_back(Ry_add.generators());
    const Tensor y_part = Tensor::concat(y_gens, -1);
    const Tensor no_x =
        Rx.center().zeros_like({n, y_part.shape()[1] - Ry_cor.generators().shape()[1]});
    const Tensor c = Tensor::concat({Rx.center(), Ry_cor.center()}, -2);
    const Tensor G = Tensor::concat({Tensor::concat({Rx.generators(), no_x}, -1), y_part}, -2);
    Zonotope Z(c, G);
    if (m > 0) Z = Z.cart_prod(*Uz);
    const Zonotope R_xyu = Z.reduce(error_order);
    // the largest |dz| per coordinate over the boxes
    std::vector<double> dz = aux_abs_max(dx).data();
    for (const double v : aux_abs_max(dy).data()) dz.push_back(v);
    if (du)
        for (const double v : aux_abs_max(*du).data()) dz.push_back(v);
    // the midpoint matrices and the radius terms of the rows of f, then of g
    std::vector<Tensor> Hf_mid, Hg_mid;
    std::vector<double> x_rad, y_rad;
    aux_rows(H.f, n, d, dz, p_x, Hf_mid, x_rad);
    aux_rows(H.g, p, d, dz, p_x, Hg_mid, y_rad);
    // one map for the rows of f and of g, so that both keep the generators of one layout
    std::vector<Tensor> Q = Hf_mid;
    Q.insert(Q.end(), Hg_mid.begin(), Hg_mid.end());
    const Zonotope E = R_xyu.quad_map(Q) * 0.5;
    const Tensor &Ec = E.center(), &EG = E.generators();
    return {Zonotope(Ec.slice(-2, 0, n), EG.slice(-2, 0, n)),
            Zonotope(Ec.slice(-2, n, n + p), EG.slice(-2, n, n + p)),
            Tensor::like(p_x, x_rad, {n, 1}), Tensor::like(p_x, y_rad, {p, 1})};
}

/// The zonotope with the center and generators of a remainder: mid + box(rad).
Zonotope aux_error(const Zonotope &mid, const Tensor &rad) { return mid + aux_box(rad); }

/// The set at the generators of `a` and `b` of one layout, combined as a - M b (the midpoints of
/// the error of f and the error of g stay correlated), plus the radii: a_rad - M b_rad.
Zonotope aux_combine(const Remainder &r, const Tensor &CF_inv) {
    const Zonotope mid(r.x_mid.center() - CF_inv.matmul(r.y_mid.center()),
                       r.x_mid.generators() - CF_inv.matmul(r.y_mid.generators()));
    return mid + aux_box(r.x_rad) + aux_box(r.y_rad).linear_map(-CF_inv);
}

// The error of the whole system and of the parts --------------------------------------------------

/// The sets of the error of the flow, of f alone and of g alone.
struct Parts {
    Zonotope set, x, y;
};

/// The error of the whole system: its Hessian, the midpoints of f and g kept on common factors.
Parts aux_whole(const NonlinDASys &sys, const DaeLinearized &lin, const Zonotope &R,
                const Zonotope &R_y_cor, const Zonotope &R_y_add, const std::optional<Zonotope> &U0,
                const alg::DaeLinearization &alg) {
    const Remainder r = aux_remainder(sys, lin.x_star, lin.y_star, lin.u_star, R, R_y_cor, R_y_add,
                                      U0, alg.error_order);
    return {aux_combine(r, lin.CF_inv).reduce(alg.zonotope_order),
            aux_error(r.x_mid, r.x_rad).compact("zeros", 0.0),
            aux_error(r.y_mid, r.y_rad).compact("zeros", 0.0).reduce(alg.zonotope_order)};
}

/// The error of the subsystems: each on its own equations, summed over the states and constraints
/// it owns, and combined with the C F^-1 of the whole system.
Parts aux_composed(const DaeLinearized &lin, const Zonotope &R, const Zonotope &R_y_cor,
                   const Zonotope &R_y_add, const std::optional<Zonotope> &U0,
                   const DaeDecomposition &decomposition, const alg::DaeLinearization &alg) {
    const int64_t n = lin.x_star.shape()[0], p = lin.y_star.shape()[0];
    const Zonotope R_y = R_y_cor + R_y_add;
    Zonotope total_x(lin.x_star.zeros_like(), lin.x_star.zeros_like({n, 0}));
    Zonotope total_y(lin.y_star.zeros_like(), lin.y_star.zeros_like({p, 0}));
    for (const DaeSubsystem &s : decomposition.subsystems) {
        // the subsystem's point and the inputs it reads: its neighbours' y and the voltages
        const Tensor p_x = s.X.matmul(lin.x_star), p_y = s.Y.matmul(lin.y_star);
        const Tensor p_u =
            s.Uv.matmul(decomposition.v_gen) + s.Uy.matmul(lin.y_star) + s.Uu.matmul(lin.u_star);
        Zonotope Usub = R_y.linear_map(s.Uy);
        if (U0) Usub = Usub + U0->linear_map(s.Uu);
        const Remainder r =
            aux_remainder(s.sys, p_x, p_y, p_u, R.linear_map(s.X), R_y_cor.linear_map(s.Y),
                          R_y_add.linear_map(s.Y), Usub, alg.error_order);
        // the parts are independent sums: without the generators that are zero in their rows
        const Zonotope part_x =
            aux_error(r.x_mid, r.x_rad).compact("zeros", 0.0).reduce(alg.zonotope_order);
        const Zonotope part_y =
            aux_error(r.y_mid, r.y_rad).compact("zeros", 0.0).reduce(alg.zonotope_order);
        total_x = total_x + part_x.linear_map(s.X.matrix_transpose());
        total_y = total_y + part_y.linear_map(s.Y.matrix_transpose());
    }
    const Zonotope error_y = total_y.reduce(alg.zonotope_order);
    return {(total_x + error_y.linear_map(lin.CF_inv * -1.0)).reduce(alg.zonotope_order), total_x,
            error_y};
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The linearization error of a step ---------------------------------------------------------------

DaeError priv_dae_lin_error(const NonlinDASys &sys, const DaeLinearized &lin, const Zonotope &R,
                            const Zonotope &V_y, const std::optional<Zonotope> &U0,
                            const std::optional<DaeDecomposition> &decomposition,
                            const alg::DaeLinearization &alg) {
    // the algebraic set: the part that depends on R, and the rest
    const Tensor minus_F_inv = lin.F_inv * -1.0;
    const Zonotope R_y_cor = (R.linear_map(lin.D) + lin.g0).linear_map(minus_F_inv);
    Zonotope uncorrelated = V_y;
    if (U0) uncorrelated = uncorrelated + (*U0).linear_map(lin.E);
    const Zonotope R_y_add = uncorrelated.linear_map(minus_F_inv);
    // the error of the flow, of f and of g
    const Parts error = decomposition
                            ? aux_composed(lin, R, R_y_cor, R_y_add, U0, *decomposition, alg)
                            : aux_whole(sys, lin, R, R_y_cor, R_y_add, U0, alg);
    // the algebraic variables over the step, and the bounds
    Zonotope u_part = R.linear_map(lin.D) + lin.g0;
    if (U0) u_part = u_part + (*U0).linear_map(lin.E);
    const Zonotope algebraic =
        ((u_part + error.y).linear_map(minus_F_inv) + lin.y_star).reduce(alg.zonotope_order);
    return {error.set, aux_bound(error.set), aux_bound(error.x), aux_bound(error.y), algebraic};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
