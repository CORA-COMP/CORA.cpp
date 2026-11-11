// intersect - the level set f(x) = 0 with a polynomial zonotope, as CORA's levelSet and_ ("approx")
//
// The method of Kochdumper et al. (HSCC 2020, Sec. 4.3) for an equation that is not solved for
// one state. The box of the set is contracted to the level set (tighten_domain); then f is
// expanded to second order at the center m of the box, with the Lagrange remainder rem of the
// third derivatives over the box, and solved for the state `var` of the largest partial
// derivative: x_var = m_var + (-f(m) - rem - sum_{i != var} f_i d_i - 1/2 d' H d) / f_var for the
// deviations d_i = r_i alpha_i, alpha in [-1, 1]. The result is a polynomial zonotope with the
// factors alpha_i (the image of this map is the set, together with the independent remainder).
// If the error 2 rad(rem + 1/2 d' H d) exceeds twice the radius of the box in `var`, or f_var is
// 0, the box itself is the result. A set that is a point is kept if f(point) <= 0.
//
// Not ported: the branch for an equation solved symbolically for one state (the solved forms of
// CORA's levelSet), the '<=' and '<' constraints, several constraints.
//
// Syntax:   std::optional<PolyZonotope> R = ls.intersect(pZ);
// Inputs:   pZ - a polynomial zonotope of the dimension of the level set
// Outputs:  R - a polynomial zonotope that holds the intersection, none if it is empty
// See also: tighten_domain, PolyZonotope
//
// CORA: contSet/@levelSet/and_.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/level_set/level_set.h"
#include "cora/cont_set/poly_zonotope/poly_zonotope.h"
#include "cora/global/within_tol.h"
#include "global/format.h"

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Intervals of numbers ----------------------------------------------------------------------------

struct Bounds {
    double lo = 0, hi = 0;
    double center() const { return 0.5 * (lo + hi); }
    double rad() const { return 0.5 * (hi - lo); }
};

Bounds aux_plus(const Bounds &a, const Bounds &b) { return {a.lo + b.lo, a.hi + b.hi}; }

Bounds aux_times(const Bounds &a, const Bounds &b) {
    const double p[4] = {a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi};
    return {*std::min_element(p, p + 4), *std::max_element(p, p + 4)};
}

/// The square of the interval: nonnegative.
Bounds aux_square(const Bounds &a) {
    const double lo = a.lo * a.lo, hi = a.hi * a.hi;
    if (a.lo >= 0) return {lo, hi};
    if (a.hi <= 0) return {hi, lo};
    return {0.0, std::max(lo, hi)};
}

/// CORA's aux_quadEval: x' Q x for x in the box d, with the cross terms summed first.
Bounds aux_quad_eval(const std::vector<std::vector<Bounds>> &Q, const std::vector<Bounds> &d) {
    Bounds res;
    for (std::size_t k = 0; k < d.size(); ++k) {
        Bounds cross;
        for (std::size_t l = k + 1; l < d.size(); ++l)
            cross = aux_plus(cross, aux_times(aux_plus(Q[k][l], Q[l][k]), d[l]));
        res = aux_plus(res, aux_plus(aux_times(Q[k][k], aux_square(d[k])), aux_times(cross, d[k])));
    }
    return res;
}

// The derivatives ---------------------------------------------------------------------------------

/// The derivatives of f at the center m and over the box: grad, Hessian, and the third derivative
/// matrices T[k] = d/dx_k of the Hessian.
struct Derivatives {
    double f = 0;
    std::vector<double> grad;
    std::vector<std::vector<double>> hess;
    std::vector<std::vector<std::vector<Bounds>>> third;
};

Derivatives aux_derivatives(const LevelSet &ls, const std::vector<double> &m,
                            const std::vector<double> &lo, const std::vector<double> &hi,
                            const Tensor &ref) {
    const auto n = std::size_t(ls.dim());
    // the expressions: f, f_i, f_ij and f_ijk
    std::vector<Expr> grad, hess, third;
    for (std::size_t i = 0; i < n; ++i) grad.push_back(ls.gradient(0, int64_t(i)));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) hess.push_back(grad[i].diff(int(j)));
    for (std::size_t k = 0; k < n; ++k)
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) third.push_back(hess[i * n + j].diff(int(k)));
    // the values at the center, one scalar per state
    std::vector<Tensor> at;
    std::vector<Range> box;
    for (std::size_t i = 0; i < n; ++i) {
        at.push_back(Tensor::like(ref, {m[i]}, {1, 1}));
        box.emplace_back(Tensor::like(ref, {lo[i]}, {1, 1}), Tensor::like(ref, {hi[i]}, {1, 1}));
    }
    Derivatives d;
    d.f = Expr::eval_tensor(ls.equations(), at).front().item();
    for (const Tensor &g : Expr::eval_tensor(grad, at)) d.grad.push_back(g.item());
    const std::vector<Tensor> h = Expr::eval_tensor(hess, at);
    d.hess.assign(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n * n; ++i) d.hess[i / n][i % n] = h[i].item();
    // the third derivatives over the box
    const std::vector<Range> t = Expr::enclose(third, box);
    d.third.assign(n, std::vector<std::vector<Bounds>>(n, std::vector<Bounds>(n)));
    for (std::size_t i = 0; i < n * n * n; ++i)
        d.third[i / (n * n)][(i / n) % n][i % n] = {t[i].lo.item(), t[i].hi.item()};
    return d;
}

// The polynomial zonotope -------------------------------------------------------------------------

/// The exponents of the quadratic terms: alpha_i^2, then alpha_i alpha_j for j > i, per i.
Exponents aux_quadratic_exponents(std::size_t n) {
    Exponents E;
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<int> e(n, 0);
        e[i] = 2;
        E.push_back(e);
        for (std::size_t j = i + 1; j < n; ++j) {
            std::vector<int> cross(n, 0);
            cross[i] = cross[j] = 1;
            E.push_back(cross);
        }
    }
    return E;
}

/// The point of the set if it has no extent: no independent generator, and no dependent one with
/// an exponent (CORA's aux_isavector).
bool aux_is_point(const PolyZonotope &S) {
    if (S.independent_generators().shape().back() > 0) return false;
    for (std::size_t k = 0; k < S.exponents().size(); ++k)
        if (std::any_of(S.exponents()[k].begin(), S.exponents()[k].end(),
                        [](int e) { return e != 0; }))
            return false;
    return true;
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::optional<PolyZonotope> LevelSet::intersect(const PolyZonotope &S) const {
    detail::Entry entry;
    CORA_CHECK(num_equations() == 1 && ops_[0] == CompOp::eq, Error::not_supported, [] {
        return std::string("LevelSet::intersect: one constraint f(x) == 0 is supported");
    });
    CORA_CHECK(S.dim() == dim(), Error::dimension_mismatch, [&] {
        return "LevelSet::intersect: the set has the dimension " + std::to_string(S.dim()) +
               ", the level set " + std::to_string(dim());
    });
    // a set with no extent is a point: kept if it fulfills f <= 0 (CORA's "< 0 or within tol")
    if (aux_is_point(S)) {
        const Tensor c = S.center() + S.generators().sum(-1, true);
        const double g = fun_han(c).item();
        if (g < 0 || within_tol(g, 0.0)) return PolyZonotope(Zonotope(c, c.zeros_like({dim(), 0})));
        return std::nullopt;
    }
    // the box of the set, contracted to the level set
    const std::optional<Interval> box = tighten_domain(S.interval());
    if (!box) return std::nullopt;
    const Interval &I = *box;
    const auto n = std::size_t(dim());
    const std::vector<double> lo = I.inf().data(), hi = I.sup().data();
    std::vector<double> m(n), r(n);
    for (std::size_t i = 0; i < n; ++i) {
        m[i] = 0.5 * (lo[i] + hi[i]);
        r[i] = 0.5 * (hi[i] - lo[i]);
    }
    const Derivatives D = aux_derivatives(*this, m, lo, hi, I.inf());
    // the state to solve for: the largest partial derivative (the first on a tie)
    std::size_t var = 0;
    for (std::size_t i = 1; i < n; ++i)
        if (std::abs(D.grad[i]) > std::abs(D.grad[var])) var = i;
    if (D.grad[var] == 0) return PolyZonotope(I); // a flat constraint cannot be solved
    // the Lagrange remainder: sum_i 1/6 d_i (d' T_i d) for the deviations d from the center
    std::vector<Bounds> dev(n);
    for (std::size_t i = 0; i < n; ++i) dev[i] = {lo[i] - m[i], hi[i] - m[i]};
    Bounds rem;
    for (std::size_t i = 0; i < n; ++i)
        rem = aux_plus(rem, aux_times(aux_times({1.0 / 6.0, 1.0 / 6.0}, dev[i]),
                                      aux_quad_eval(D.third[i], dev)));
    // the error of the expansion: 2 rad(rem + 1/2 d' H d), too large for the box to gain
    std::vector<std::vector<Bounds>> H(n, std::vector<Bounds>(n));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) H[i][j] = {D.hess[i][j], D.hess[i][j]};
    const Bounds quad = aux_quad_eval(H, dev);
    const Bounds total = aux_plus(rem, aux_times({0.5, 0.5}, quad));
    if (2 * total.rad() > 2 * r[var]) return PolyZonotope(I);
    // the dependent generators: quadratic and linear terms in the row var, the deviations elsewhere
    std::vector<double> g_var;
    Exponents E = aux_quadratic_exponents(n);
    for (std::size_t i = 0; i < n; ++i) {
        g_var.push_back(r[i] * r[i] * D.hess[i][i]);
        for (std::size_t j = i + 1; j < n; ++j)
            g_var.push_back(r[i] * r[j] * (D.hess[i][j] + D.hess[j][i]));
    }
    for (double &g : g_var) g = -(0.5 * g);
    std::vector<std::size_t> others;
    for (std::size_t i = 0; i < n; ++i)
        if (i != var) others.push_back(i);
    for (std::size_t i : others) {
        g_var.push_back(-(D.grad[i] * r[i]));
        std::vector<int> e(n, 0);
        e[i] = 1;
        E.push_back(e);
    }
    // the matrix: the row var holds the terms divided by the derivative, the others their box
    const std::size_t h = g_var.size() + others.size();
    std::vector<double> G(n * h, 0.0), c(n);
    for (std::size_t k = 0; k < g_var.size(); ++k) G[var * h + k] = g_var[k] / D.grad[var];
    for (std::size_t k = 0; k < others.size(); ++k) {
        G[others[k] * h + g_var.size() + k] = r[others[k]];
        std::vector<int> e(n, 0);
        e[others[k]] = 1;
        E.push_back(e);
    }
    c = m;
    c[var] = (-D.f - rem.center() + D.grad[var] * m[var]) / D.grad[var];
    std::vector<double> GI(n, 0.0);
    GI[var] = rem.rad() / D.grad[var];
    std::vector<int64_t> ids(n);
    for (std::size_t i = 0; i < n; ++i) ids[i] = int64_t(i) + 1;
    return PolyZonotope(Tensor::like(I.inf(), c, {int64_t(n), 1}),
                        Tensor::like(I.inf(), G, {int64_t(n), int64_t(h)}),
                        Tensor::like(I.inf(), GI, {int64_t(n), 1}), std::move(E), std::move(ids));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
