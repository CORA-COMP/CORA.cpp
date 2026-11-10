// priv_order_step - one step of a nonlinear system at the tensor order 1 or 2, and without error
//
// The step linearizes f at x* (the center of the set moved half a step along the flow) and at the
// center of the input set, checks that the flow stays in a region around the set (regionFloat) and
// bounds the linearization error over that region: at the order 1 by the slopes of the Jacobian
// ([J_x] - A)(X - x*) + ([J_u] - B)(U - u*), at the order 2 by the Hessian, 1/2 dz' |H_i| dz. The
// affine system x' = A (x - x*) + f(x*) + v, v in B (U - u*) + L, is solved by one step of the
// linear system. The pass without error (priv_adapt_lin_pass) gives the sets whose size the error
// is measured against.
//
// Syntax:   StepSets s = priv_order_step(sys, 2, U, alg, dt, R);
// Inputs:   sys, U - the system and the input set (none: no inputs), alg, dt - the step size, R
// Outputs:  s - the sets over the step and at its end
// See also: priv_adapt_step.cpp, LinearSys::one_step
//
// CORA: contDynamics/@nonlinearSys/linReach_adaptive.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       10-November-2026
// Last update:   10-November-2026
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_steps.h"
#include "cont_dynamics/nonlinear_sys/private/priv_adaptive.h"

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The boxes ---------------------------------------------------------------------------------------

/// A box as the bounds of its coordinates.
struct Box {
    Vec lo, hi;
};

/// What the step functions share: the system, the algorithm and the input set (never empty for a
/// system with inputs).
struct Context {
    const NonlinearSys &sys;
    const alg::LinearizationAdaptive &alg;
    std::optional<Zonotope> U;
    Tensor like;
    int64_t n, m;
};

Context aux_context(const NonlinearSys &sys, const std::optional<Zonotope> &U,
                    const alg::LinearizationAdaptive &alg, const Zonotope &R) {
    // a system with inputs gets the input 0 if it is given none
    const int64_t m = sys.input_dim();
    std::optional<Zonotope> input;
    if (m > 0)
        input = U ? *U : Zonotope(R.center().zeros_like({m, 1}), R.center().zeros_like({m, 0}));
    return {sys, alg, input, R.center(), sys.dim(), m};
}

Box aux_box(const Interval &I) { return {I.inf().data(), I.sup().data()}; }

Box aux_point(const Vec &p) { return {p, p}; }

Tensor aux_column(const Context &c, const Vec &v) {
    return Tensor::like(c.like, v, {int64_t(v.size()), 1});
}

Range aux_range(const Context &c, const Box &b) {
    return {aux_column(c, b.lo), aux_column(c, b.hi)};
}

/// The box of a vector of ranges, the zonotope of a box, and f over boxes.
Box aux_box_of(const Range &r) { return {r.lo.data(), r.hi.data()}; }

Zonotope aux_zonotope(const Context &c, const Box &b) {
    // the center and the largest distance of the center to a bound, per coordinate
    Vec center(b.lo.size()), radius(b.lo.size());
    for (std::size_t i = 0; i < b.lo.size(); ++i) {
        center[i] = 0.5 * (b.lo[i] + b.hi[i]);
        radius[i] = std::max(b.hi[i] - center[i], center[i] - b.lo[i]);
    }
    return {aux_column(c, center), aux_column(c, radius).diag()};
}

Box aux_f(const Context &c, const Box &x, const std::optional<Box> &u) {
    return aux_box_of(c.sys.enclosure(aux_range(c, x),
                                      u ? std::optional<Range>(aux_range(c, *u)) : std::nullopt));
}

/// max over the coordinates of max(|inf|, |sup|): the norm of the box.
double aux_bound(const Box &b) {
    double bound = 0;
    for (std::size_t i = 0; i < b.lo.size(); ++i)
        bound = std::max(bound, std::max(std::abs(b.lo[i]), std::abs(b.hi[i])));
    return bound;
}

/// The box that holds b and the point p.
Box aux_hull(const Box &b, const Vec &p) {
    Box out = b;
    for (std::size_t i = 0; i < p.size(); ++i) {
        out.lo[i] = std::min(b.lo[i], p[i]);
        out.hi[i] = std::max(b.hi[i], p[i]);
    }
    return out;
}

// The region and the expansion point --------------------------------------------------------------

/// The region X + [0, dt] f(region) twice over, where the flow stays (regionFloat); none if the
/// bound of the field over the region does not keep it inside.
std::optional<Box> aux_region(const Context &c, double dt, const Box &X,
                              const std::optional<Box> &U) {
    // the region around X that the flow can leave in dt, by the bound of f on X
    const double M0 = aux_bound(aux_f(c, X, U));
    const double delta = ((M0 + M0) + 1) * dt;
    Box omega = X;
    for (std::size_t i = 0; i < X.lo.size(); ++i) {
        omega.lo[i] = X.lo[i] - delta;
        omega.hi[i] = X.hi[i] + delta;
    }
    const double M = aux_bound(aux_f(c, omega, U));
    if (!(M * dt < delta)) return std::nullopt;
    // the flow from X stays in the region if M dt < delta, then two Picard steps tighten it
    const auto picard = [&](const Box &om) {
        const Box F = aux_f(c, om, U);
        Box out = X;
        for (std::size_t i = 0; i < X.lo.size(); ++i) {
            out.lo[i] = X.lo[i] + std::min(0.0, dt * F.lo[i]);
            out.hi[i] = X.hi[i] + std::max(0.0, dt * F.hi[i]);
        }
        return out;
    };
    return picard(picard(omega));
}

/// The expansion point x* = center + dt/2 f(center, u*) (linPointFloat) and u* = center of U.
std::pair<Vec, Vec> aux_lin_point(const Context &c, double dt, const Zonotope &R) {
    // x* = center + dt/2 f(center, u*)
    const Vec center = R.center().data();
    const Vec us = c.m > 0 ? c.U->center().data() : Vec{};
    const Box f =
        aux_f(c, aux_point(center), c.m > 0 ? std::optional<Box>(aux_point(us)) : std::nullopt);
    Vec xs(center.size());
    for (std::size_t i = 0; i < xs.size(); ++i)
        xs[i] = center[i] + (0.5 * dt) * (0.5 * (f.lo[i] + f.hi[i]));
    return {xs, us};
}

/// f(x*, u*) as a zonotope plus B (U - u*): the input of the linearized system without error.
Zonotope aux_input(const Context &c, const Vec &xs, const Vec &us, const Tensor &B) {
    const Box f =
        aux_f(c, aux_point(xs), c.m > 0 ? std::optional<Box>(aux_point(us)) : std::nullopt);
    Zonotope V = aux_zonotope(c, f);
    if (c.m > 0) V = V + (*c.U - c.U->center()).linear_map(B);
    return V;
}

// The remainders ----------------------------------------------------------------------------------

/// The product of two intervals.
std::pair<double, double> aux_imul(double al, double ah, double bl, double bh) {
    const double p[4] = {al * bl, al * bh, ah * bl, ah * bh};
    return {*std::min_element(p, p + 4), *std::max_element(p, p + 4)};
}

/// ([J_x] - A)(X - x*) + ([J_u] - B)(U - u*) over X and U (slopeRemainder).
Box aux_slope_remainder(const Context &c, const Tensor &A, const Tensor &B, const Vec &xs,
                        const Vec &us, const Box &X, const std::optional<Box> &U) {
    // the slopes of the Jacobian over X and U against the matrices A and B
    const int64_t n = c.n, d = c.n + c.m;
    const auto J = c.sys.jacobian_enclosure(
        aux_range(c, X), U ? std::optional<Range>(aux_range(c, *U)) : std::nullopt);
    // row i of the result is the sum over the columns j of the interval products
    const Vec a = A.data(), b = B.data();
    Box L{Vec(std::size_t(n), 0.0), Vec(std::size_t(n), 0.0)};
    for (int64_t i = 0; i < n; ++i) {
        double lo = 0, hi = 0;
        for (int64_t j = 0; j < d; ++j) {
            // the slope minus the matrix entry, times the deviation from the expansion point
            const auto &entry = J[std::size_t(i * d + j)];
            const double centre =
                j < n ? a[std::size_t(i * n + j)] : b[std::size_t(i * c.m + j - n)];
            const double jl = entry ? entry->lo.item() : 0.0, jh = entry ? entry->hi.item() : 0.0;
            const double pt = j < n ? xs[std::size_t(j)] : us[std::size_t(j - n)];
            const Box &Z = j < n ? X : *U;
            const std::size_t k = std::size_t(j < n ? j : j - n);
            const auto prod = aux_imul(jl - centre, jh - centre, Z.lo[k] - pt, Z.hi[k] - pt);
            lo += prod.first;
            hi += prod.second;
        }
        L.lo[std::size_t(i)] = lo;
        L.hi[std::size_t(i)] = hi;
    }
    return L;
}

/// [-l, l] with l_i = 1/2 dz' |H_i| dz over X and U (hessRemainder; the slope terms of the Jacobian
/// at the point vanish there, A and B being its center).
Box aux_hessian_remainder(const Context &c, const Vec &xs, const Vec &us, const Box &X,
                          const std::optional<Box> &U) {
    const int64_t n = c.n, d = c.n + c.m;
    const auto H = c.sys.hessian_enclosure(
        aux_range(c, X), U ? std::optional<Range>(aux_range(c, *U)) : std::nullopt);
    // dz: the largest deviation of each coordinate from the expansion point
    Vec dz(static_cast<std::size_t>(d), 0.0);
    for (int64_t j = 0; j < d; ++j) {
        const Box &Z = j < n ? X : *U;
        const std::size_t k = std::size_t(j < n ? j : j - n);
        const double pt = j < n ? xs[std::size_t(j)] : us[std::size_t(j - n)];
        dz[std::size_t(j)] = std::max(std::abs(Z.lo[k] - pt), std::abs(Z.hi[k] - pt));
    }
    // l_i = 1/2 sum_jk dz_j |H_ijk| dz_k
    Box L{Vec(static_cast<std::size_t>(n), 0.0), Vec(static_cast<std::size_t>(n), 0.0)};
    for (int64_t i = 0; i < n; ++i) {
        double sum = 0;
        for (int64_t j = 0; j < d; ++j)
            for (int64_t k = 0; k < d; ++k) {
                const auto &entry = H[std::size_t((i * d + j) * d + k)];
                if (!entry) continue;
                const double mag = std::max(std::abs(entry->lo.item()), std::abs(entry->hi.item()));
                sum += dz[std::size_t(j)] * mag * dz[std::size_t(k)];
            }
        L.lo[std::size_t(i)] = -(0.5 * sum);
        L.hi[std::size_t(i)] = 0.5 * sum;
    }
    return L;
}

// The linear step ---------------------------------------------------------------------------------

/// One step of x' = A (x - x*) + v, v in V, from R (oneStepFloat): the linear system's step from
/// R - x*, shifted back; the Taylor order is raised until it is admissible.
StepSets aux_linear_step(const Context &c, const LinearSys &lin, TaylorLinSys &taylor,
                         const Tensor &xs, const Zonotope &V, const Zonotope &R, double dt) {
    // eta + 2 >= 2 ||A dt||, from the larger of eta and 2
    const double norm = lin.A().abs().sum(-1, true).max().item() * dt;
    int eta = std::max(c.alg.taylor_terms, 2);
    while (double(eta + 2) < 2 * norm && eta < 4096) ++eta;
    if (double(eta + 2) < 2 * norm)
        CORA_FAIL(Error::reach_set_explosion,
                  "reach: no Taylor order admits the step size " + std::to_string(dt));
    // the constant input is the center of V, the time-varying one V without it
    const Tensor &v = V.center();
    const Zonotope point(v, v.zeros_like({c.n, 0}));
    const BlockStep s =
        priv_one_step(lin, taylor, priv_one_block(c.n), R - xs, V - v, point, dt, eta);
    return {s.time_interval[0] + xs, s.time_point[0] + xs};
}

} // namespace


// ===========================================  MAIN  =========================================== //

/// What the passes of one step size from one set share: the expansion point, the linearized system
/// with its powers, the input without error and, once an order step asks, the region.
struct AdaptPass::Impl {
    Context c;
    double dt;
    Zonotope R;
    Vec xs, us;
    Tensor xs_t;
    Jacobians J;
    Zonotope base;
    LinearSys lin;
    TaylorLinSys taylor;
    std::optional<std::pair<Box, std::optional<Box>>> region;

    Impl(Context ctx, double step, const Zonotope &set, Vec x, Vec u, Tensor x_t, Jacobians jac,
         Zonotope input)
        : c(std::move(ctx)), dt(step), R(set), xs(std::move(x)), us(std::move(u)),
          xs_t(std::move(x_t)), J(std::move(jac)), base(std::move(input)), lin(J.A), taylor(J.A) {}
};

AdaptPass::AdaptPass(const NonlinearSys &sys, const std::optional<Zonotope> &U,
                     const alg::LinearizationAdaptive &alg, double dt, const Zonotope &R) {
    Context c = aux_context(sys, U, alg, R);
    // the expansion point and the linearized system, with the input without error
    const auto [xs, us] = aux_lin_point(c, dt, R);
    const Tensor xs_t = aux_column(c, xs);
    const Jacobians J =
        sys.jacobian(xs_t, c.m > 0 ? std::optional<Tensor>(aux_column(c, us)) : std::nullopt);
    Zonotope base = aux_input(c, xs, us, J.B);
    impl_ = std::make_shared<Impl>(std::move(c), dt, R, xs, us, xs_t, J, std::move(base));
}

LinPass AdaptPass::lin_pass() {
    Impl &m = *impl_;
    return {m.J.A, aux_linear_step(m.c, m.lin, m.taylor, m.xs_t, m.base, m.R, m.dt)};
}

StepSets AdaptPass::order_step(int kappa) {
    Impl &m = *impl_;
    // where the flow stays, around the set, the expansion point and the input set
    if (!m.region) {
        const std::optional<Box> Ub =
            m.c.m > 0 ? std::optional<Box>(aux_box(m.c.U->interval())) : std::nullopt;
        const std::optional<Box> omega = aux_region(m.c, m.dt, aux_box(m.R.interval()), Ub);
        if (!omega)
            CORA_FAIL(Error::reach_set_explosion,
                      "reach: the flow does not stay in the region for the step size " +
                          std::to_string(m.dt));
        m.region = std::make_pair(aux_hull(*omega, m.xs),
                                  Ub ? std::optional<Box>(aux_hull(*Ub, m.us)) : std::nullopt);
    }
    // the linearization error at the order kappa
    const Box L =
        kappa == 1
            ? aux_slope_remainder(m.c, m.J.A, m.J.B, m.xs, m.us, m.region->first, m.region->second)
            : aux_hessian_remainder(m.c, m.xs, m.us, m.region->first, m.region->second);
    const Zonotope V = m.base + aux_zonotope(m.c, L);
    return aux_linear_step(m.c, m.lin, m.taylor, m.xs_t, V, m.R, m.dt);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
