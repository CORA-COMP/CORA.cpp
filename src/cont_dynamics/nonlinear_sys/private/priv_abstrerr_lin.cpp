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

// The boxes ---------------------------------------------------------------------------------------

/// The entries of a column (n, 1) as n tensors of shape (1, 1).
std::vector<Tensor> aux_entries(const Tensor &column) {
    const Tensor row = column.matrix_transpose();
    std::vector<Tensor> out;
    out.reserve(std::size_t(column.shape()[0]));
    for (int64_t i = 0; i < column.shape()[0]; ++i) out.push_back(row.slice(-1, i, i + 1));
    return out;
}

/// The column of the entries.
Tensor aux_column(const std::vector<Tensor> &entries) { return Tensor::concat(entries, -2); }

/// max(|inf|, |sup|) of a box: CORA's max(abs(infimum(I)), abs(supremum(I))).
Tensor aux_abs_max(const Tensor &lo, const Tensor &hi) { return lo.abs().maximum(hi.abs()); }

/// The bounds of z - z* as ranges (1, 1): the box of R, then that of U0.
std::vector<Range> aux_dz(const Interval &IHx, const std::optional<Interval> &IHu) {
    std::vector<Range> dz;
    for (const Interval *I : {&IHx, IHu ? &*IHu : nullptr}) {
        if (!I) continue;
        const std::vector<Tensor> lo = aux_entries(I->inf()), hi = aux_entries(I->sup());
        for (std::size_t j = 0; j < lo.size(); ++j) dz.emplace_back(lo[j], hi[j]);
    }
    return dz;
}

/// A NaN or infinite remainder fails as CORA's reachSetExplosion.
void aux_check_finite(const Tensor &l) {
    for (const double v : l.data())
        if (!std::isfinite(v))
            CORA_FAIL(Error::reach_set_explosion,
                      "cora: the Lagrange remainder exploded (the time_step may be too large)");
}

// The orders --------------------------------------------------------------------------------------
/// tensorOrder 2: 1/2 dz' max|H_i| dz per component over the nonzero entries of the Hessian.
Tensor aux_second_order(const std::vector<std::optional<Range>> &H, const std::vector<Tensor> &dz,
                        int64_t n) {
    const auto d = int64_t(dz.size());
    std::vector<Tensor> l;
    l.reserve(std::size_t(n));
    for (int64_t i = 0; i < n; ++i) {
        // (dz' H) dz: the row dz' H first, as MATLAB multiplies from the left
        Tensor sum = dz[0].zeros_like();
        for (int64_t k = 0; k < d; ++k) {
            Tensor row = dz[0].zeros_like();
            for (int64_t j = 0; j < d; ++j)
                if (const auto &h = H[std::size_t((i * d + j) * d + k)])
                    row = row + dz[std::size_t(j)].multiply(aux_abs_max(h->lo, h->hi));
            sum = sum + row.multiply(dz[std::size_t(k)]);
        }
        l.push_back(sum * 0.5);
    }
    return aux_column(l);
}

/// tensorOrder 3: the box 1/6 sum_j dz_j (dz' T_ij dz) per component, over the j with an entry
/// of T_ij that is not identically 0 (CORA's ind{i}); none if there is no such entry at all.
std::optional<Interval> aux_third_order(const std::vector<std::optional<Range>> &T,
                                        const std::vector<Range> &dz, int64_t n) {
    const auto d = int64_t(dz.size());
    const Range zero(dz[0].lo.zeros_like());
    std::vector<Tensor> lo, hi;
    bool any = false;
    for (int64_t i = 0; i < n; ++i) {
        Range sum = zero;
        for (int64_t j = 0; j < d; ++j) {
            const std::size_t base = std::size_t((i * d + j) * d * d);
            bool nonzero = false;
            for (int64_t kl = 0; kl < d * d && !nonzero; ++kl) nonzero = T[base + kl].has_value();
            if (!nonzero) continue;
            any = true;
            // (dz' T_ij) dz, the row first
            Range quad = zero;
            for (int64_t l = 0; l < d; ++l) {
                Range r = zero;
                for (int64_t k = 0; k < d; ++k)
                    if (const auto &t = T[base + std::size_t(k * d + l)]) r = r + dz[k] * *t;
                quad = quad + r * dz[std::size_t(l)];
            }
            sum = sum + quad * dz[std::size_t(j)];
        }
        sum = sum * (1.0 / 6);
        lo.push_back(sum.lo);
        hi.push_back(sum.hi);
    }
    if (!any) return std::nullopt;
    return Interval(aux_column(lo), aux_column(hi));
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
        std::vector<Tensor> dz = aux_entries(aux_abs_max(IHx.inf(), IHx.sup()));
        if (IHu)
            for (Tensor &e : aux_entries(aux_abs_max(IHu->inf(), IHu->sup())))
                dz.push_back(std::move(e));
        const Tensor l = aux_second_order(sys.hessian_enclosure(x_box, u_box), dz, n);
        aux_check_finite(l);
        return {l, Zonotope(l.zeros_like(), l.diag())};
    }
    // tensorOrder 3: the quadratic term as a set, the cubic remainder as a box
    const Zonotope R_red = R.reduce(alg.error_order);
    const Zonotope Z = U0 ? R_red.cart_prod(*U0) : R_red;
    const std::optional<Tensor> u_star = U0 ? std::optional<Tensor>(lin.u_star) : std::nullopt;
    const Zonotope second = Z.quad_map(sys.hessian(lin.x_star, u_star)) * 0.5;
    const std::optional<Interval> third =
        aux_third_order(sys.third_order_enclosure(x_box, u_box), aux_dz(IHx, IHu), n);
    const Zonotope V = (third ? second + third->zonotope() : second).reduce(alg.intermediate_order);
    const Interval box = V.interval();
    return {aux_abs_max(box.inf(), box.sup()), V};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
