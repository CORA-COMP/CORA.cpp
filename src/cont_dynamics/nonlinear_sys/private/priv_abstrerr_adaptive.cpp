// priv_abstrerr_adaptive - the linearization error of the adaptive algorithm (VerrorDyn)
//
// For the set Rmax of a pass (absolute) and the input set U, expanded at (x*, u*): at the tensor
// order 2 the box [-l, l] with l_i = 1/2 dz' |H_i| dz, the Hessian taken over the box of Rmax and
// U joined with the expansion point and dz the largest deviation from it; at the order 3 the
// quadratic form with the Hessian at the point on the set reduced adaptively (a zonotope, no box)
// plus the box of what the Hessian over the box differs from the one at the point by, reduced
// adaptively. CORALean's abstrErrAdaptiveFloat, which repairs CORA's order 3 (the third derivative
// read as dz(j) where dz(ind{i}(j)) is meant) and its Hessian region.
//
// Syntax:   Zonotope V = priv_abstr_err_adaptive(sys, kappa, A, B, xs, Rmax, U, red_factor);
// Inputs:   see the declarations in priv_adaptive.h
// Outputs:  V - the zonotope of the linearization error
// See also: priv_lin_reach_adaptive_pass.cpp
//
// CORA: contDynamics/@nonlinearSys/private/priv_abstractionError_adaptive.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/nonlinear_sys/private/priv_adaptive.h"

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

// The boxes ---------------------------------------------------------------------------------------

/// The box of b joined with the point p.
Box priv_hull(const Box &b, const Vec &p) {
    Box out = b;
    for (std::size_t i = 0; i < p.size(); ++i) {
        out.lo[i] = std::min(b.lo[i], p[i]);
        out.hi[i] = std::max(b.hi[i], p[i]);
    }
    return out;
}

/// The Hessian over the box as bounds (lo, hi) of the entries (i, a, b), 0 where it is.
Hessian priv_hessian(const NonlinearSys &sys, const Tensor &like, const Box &x,
                     const std::optional<Box> &u) {
    const auto range = [&](const Box &b) {
        return Range(priv_column(like, b.lo), priv_column(like, b.hi));
    };
    const auto H =
        sys.hessian_enclosure(range(x), u ? std::optional<Range>(range(*u)) : std::nullopt);
    Hessian out{Vec(H.size(), 0.0), Vec(H.size(), 0.0)};
    for (std::size_t k = 0; k < H.size(); ++k)
        if (H[k]) {
            out.lo[k] = H[k]->lo.item();
            out.hi[k] = H[k]->hi.item();
        }
    return out;
}

/// The box [-l, l] with l_i = 1/2 dz' max(|H_i - Hp|) dz, dz the largest deviation of X and U
/// from the expansion point (hessRemainder; Hp the Hessian to subtract, 0 for none).
Box priv_hessian_remainder(const Hessian &H, const Vec &Hp, const Box &X,
                           const std::optional<Box> &U, const Vec &xs, const Vec &us, int64_t n) {
    const int64_t m = int64_t(us.size()), d = n + m;
    Vec dz(static_cast<std::size_t>(d), 0.0);
    for (int64_t a = 0; a < d; ++a) {
        const Box &B = a < n ? X : *U;
        const auto k = std::size_t(a < n ? a : a - n);
        const double s = a < n ? xs[k] : us[k];
        dz[std::size_t(a)] = std::max(std::abs(B.lo[k] - s), std::abs(B.hi[k] - s));
    }
    Box L{Vec(static_cast<std::size_t>(n), 0.0), Vec(static_cast<std::size_t>(n), 0.0)};
    for (int64_t i = 0; i < n; ++i) {
        // dz' |H_i| dz as dz . (|H_i| dz), each sum from the last term to the first
        Vec outer(static_cast<std::size_t>(d));
        for (int64_t a = 0; a < d; ++a) {
            Vec inner(static_cast<std::size_t>(d));
            for (int64_t b = 0; b < d; ++b) {
                const auto k = std::size_t((i * d + a) * d + b);
                const double shift = Hp.empty() ? 0.0 : Hp[k];
                const double mag = std::max(std::abs(H.lo[k] - shift), std::abs(H.hi[k] - shift));
                inner[std::size_t(b)] = mag * dz[std::size_t(b)];
            }
            outer[std::size_t(a)] = dz[std::size_t(a)] * priv_sum_up(inner);
        }
        const double l = 0.5 * priv_sum_up(outer);
        L.lo[std::size_t(i)] = -l;
        L.hi[std::size_t(i)] = l;
    }
    return L;
}


Tensor priv_column(const Tensor &like, const Vec &v) {
    return Tensor::like(like, v, {int64_t(v.size()), 1});
}

// the center minus and plus the sum of the absolute generators of each coordinate
Box priv_zonotope_box(const Zonotope &Z) {
    const int64_t n = Z.dim(), h = Z.generators().shape().back();
    const Vec c = Z.center().data(), a = Z.generators().abs().data();
    Box b{Vec(static_cast<std::size_t>(n)), Vec(static_cast<std::size_t>(n))};
    for (int64_t i = 0; i < n; ++i) {
        Vec row(static_cast<std::size_t>(h));
        for (int64_t j = 0; j < h; ++j) row[std::size_t(j)] = a[std::size_t(i * h + j)];
        const double radius = priv_sum_up(row);
        b.lo[std::size_t(i)] = c[std::size_t(i)] - radius;
        b.hi[std::size_t(i)] = c[std::size_t(i)] + radius;
    }
    return b;
}

// the midpoint, and a diagonal generator that reaches both ends
Zonotope priv_box_zonotope(const Tensor &like, const Box &b) {
    Vec center(b.lo.size()), radius(b.lo.size());
    for (std::size_t i = 0; i < b.lo.size(); ++i) {
        center[i] = 0.5 * (b.lo[i] + b.hi[i]);
        radius[i] = std::max(b.hi[i] - center[i], center[i] - b.lo[i]);
    }
    return {priv_column(like, center), priv_column(like, radius).diag()};
}

// f over the boxes by the natural interval extension of the system
Box priv_enclosure(const NonlinearSys &sys, const Tensor &like, const Box &x,
                   const std::optional<Box> &u) {
    const Range r = sys.enclosure(
        Range(priv_column(like, x.lo), priv_column(like, x.hi)),
        u ? std::optional<Range>(Range(priv_column(like, u->lo), priv_column(like, u->hi)))
          : std::nullopt);
    return {r.lo.data(), r.hi.data()};
}

// each row of the generators summed from the last entry to the first
Vec priv_row_sums(const Zonotope &Z) {
    const int64_t n = Z.dim(), h = Z.generators().shape().back();
    const Vec a = Z.generators().abs().data();
    Vec out(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        Vec row(static_cast<std::size_t>(h));
        for (int64_t j = 0; j < h; ++j) row[std::size_t(j)] = a[std::size_t(i * h + j)];
        out[std::size_t(i)] = priv_sum_up(row);
    }
    return out;
}

Vec priv_norm_mag(const Box &b) {
    Vec out(b.lo.size());
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = std::max(std::abs(b.lo[i]), std::abs(b.hi[i]));
    return out;
}

std::vector<Tensor> priv_half_hessians(const Tensor &like, const Hessian &Hp, int64_t n,
                                       int64_t d) {
    // one matrix per component, 1/2 of the Hessian entries (i, a, b) at the point
    std::vector<Tensor> Q;
    for (int64_t i = 0; i < n; ++i) {
        Vec q(static_cast<std::size_t>(d * d));
        for (int64_t k = 0; k < d * d; ++k)
            q[std::size_t(k)] = 0.5 * Hp.lo[std::size_t(i * d * d + k)];
        Q.push_back(Tensor::like(like, q, {d, d}));
    }
    return Q;
}

// The linearization and the error loop ------------------------------------------------------------

AdaptiveLinearization priv_linearize_adaptive(const NonlinearSys &sys,
                                              const std::optional<Zonotope> &U, const Tensor &like,
                                              double dt) {
    const int64_t n = sys.dim(), m = sys.input_dim();
    // x* = center + dt/2 f(center, u*), u* the center of U
    const Vec center = like.data(), us = m > 0 ? U->center().data() : Vec{};
    const auto point_u = [&](const Vec &u) {
        return m > 0 ? std::optional<Box>(Box{u, u}) : std::nullopt;
    };
    const Box fc = priv_enclosure(sys, like, Box{center, center}, point_u(us));
    Vec xs(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        xs[std::size_t(i)] = center[std::size_t(i)] +
                             (0.5 * dt) * (0.5 * (fc.lo[std::size_t(i)] + fc.hi[std::size_t(i)]));
    // the Jacobians there and the input of the linear system without the error
    Tensor xs_t = priv_column(like, xs);
    Jacobians J =
        sys.jacobian(xs_t, m > 0 ? std::optional<Tensor>(priv_column(like, us)) : std::nullopt);
    Zonotope Vbase = priv_box_zonotope(like, priv_enclosure(sys, like, Box{xs, xs}, point_u(us)));
    if (m > 0) Vbase = Vbase + (*U - U->center()).linear_map(J.B);
    return {xs, std::move(xs_t), std::move(J), std::move(Vbase)};
}

std::optional<Vec> priv_error_loop_adaptive(const Vec &error_adm, int fuel,
                                            const std::function<Vec(const Vec &)> &true_error_of) {
    Vec admissible = error_adm;
    double previous = 0;
    for (int pass = 0, counter = 1; pass < fuel; ++pass, ++counter) {
        const Vec true_error = true_error_of(admissible);
        // the largest ratio of the error to the admissible one
        double current = 0;
        bool within = true;
        for (std::size_t i = 0; i < admissible.size(); ++i) {
            current = std::max(current, true_error[i] <= 0 ? 0.0 : true_error[i] / admissible[i]);
            within = within && true_error[i] <= admissible[i];
        }
        if (within) return true_error;
        if (2 < counter && !(current <= previous)) return std::nullopt;
        previous = current;
        for (std::size_t i = 0; i < admissible.size(); ++i) admissible[i] = 1.1 * true_error[i];
    }
    return std::nullopt;
}

// VerrorDyn ---------------------------------------------------------------------------------------

Zonotope priv_abstr_err_adaptive(const NonlinearSys &sys, int kappa, const Tensor &A,
                                 const Tensor &B, const Vec &xs, const Zonotope &Rmax,
                                 const std::optional<Zonotope> &U, double red_factor) {
    (void)A;
    (void)B; // the slope terms [J(z*)] - A and [J_u] - B of hessRemainder are 0: A, B are J(z*)
    const Tensor &like = Rmax.center();
    const int64_t n = sys.dim(), m = sys.input_dim();
    // the box of Rmax and of U, each joined with the expansion point
    const Vec us = m > 0 ? U->center().data() : Vec{};
    const Box X = priv_hull(priv_zonotope_box(Rmax), xs);
    const std::optional<Box> Ub =
        m > 0 ? std::optional<Box>(priv_hull(priv_zonotope_box(*U), us)) : std::nullopt;
    const Hessian H = priv_hessian(sys, like, X, Ub);
    if (kappa != 3) return priv_box_zonotope(like, priv_hessian_remainder(H, {}, X, Ub, xs, us, n));

    // the Hessian at the point, the quadratic form on the reduced set, and the rest
    const Box point_x{xs, xs};
    const std::optional<Box> point_u = m > 0 ? std::optional<Box>(Box{us, us}) : std::nullopt;
    const Hessian Hp = priv_hessian(sys, like, point_x, point_u);
    const Zonotope Rred = (Rmax - priv_column(like, xs)).reduce_adaptive(std::sqrt(red_factor));
    const Zonotope Z = m > 0 ? Rred.cart_prod(*U - U->center()) : Rred;
    const std::vector<Tensor> Q = priv_half_hessians(like, Hp, n, n + m);
    const Zonotope second = Z.quad_map(Q);
    const Zonotope rest =
        priv_box_zonotope(like, priv_hessian_remainder(H, Hp.lo, X, Ub, xs, us, n));
    return (second + rest).reduce_adaptive(10 * red_factor);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
