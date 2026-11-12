// priv_poly_window - the linear step from a polynomial zonotope of the poly-adaptive algorithm
//
// The system x' = A x + u with the input u in V, from the polynomial zonotope X0: the window over
// the step and the differences to its start, and the set at the end for a constant input with a
// polynomial zonotope over the factors of X0 (CORALean's polyWindowFloat and polyPointFloat). The
// matrix [e^{A dt}] and the correction F are interval matrices, so the factors of X0 stay. The
// input V is time-varying, its center held through hull(0, u). Its solutions are term by term
// sums of the series boxes: a summed series matrix bounds only a constant input, which only the
// integral of e^{As} at the end of the step does. As in CORA the pieces are reduced adaptively at
// zeta_z before they are added. The varying input is first boxed to n generators (a superset),
// since the sum multiplies its generators by the Taylor order and the independent part grows.
//
// Syntax:   auto series = PolySeries::make(A, dt, taylor_terms);
//           auto [Rti, Rdiff] = priv_poly_window(*series, V, X0, zeta_z);
//           PolyZonotope Rtp = priv_poly_point(*series, V, X0, Vs, zeta_z);
// Inputs:   series - the matrices of A at dt, V - input set, X0 - start set, Vs - constant input
// Outputs:  the window and the differences; the set at the end of the step
// See also: priv_lin_reach_poly_adaptive_pass.cpp, priv_poly_ops.cpp
//
// CORA: contDynamics/@linearSys/initReach_adaptive.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_float_scalars.h"
#include "cont_dynamics/nonlinear_sys/private/priv_poly_adaptive.h"

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using Mat = std::vector<double>;

/// a * b, a NaN corner (0 * Inf) taken as 0.
double aux_mul(double a, double b) {
    const double p = a * b;
    return std::isnan(p) ? 0.0 : p;
}

/// L R for (n, n) matrices, each sum over the inner index nested to the right, a(0) + (a(1) + ... +
/// 0).
Mat aux_matmul(const Mat &L, const Mat &R, std::size_t n) {
    Mat out(n * n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double sum = 0;
            for (std::size_t k = n; k-- > 0;) sum = aux_mul(L[i * n + k], R[k * n + j]) + sum;
            out[i * n + j] = sum;
        }
    return out;
}

Mat aux_identity(std::size_t n) {
    Mat I(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) I[i * n + i] = 1.0;
    return I;
}

/// c M, entry by entry.
Mat aux_scale(double c, const Mat &M) {
    Mat out(M.size());
    for (std::size_t e = 0; e < M.size(); ++e) out[e] = aux_mul(c, M[e]);
    return out;
}

/// The largest row sum of the magnitudes, each sum nested to the right, floored at 0
/// (CORALean's normUpperBound of a point matrix).
double aux_norm(const Mat &M, std::size_t n) {
    double nrm = 0;
    for (std::size_t i = 0; i < n; ++i) {
        double sum = 0;
        for (std::size_t j = n; j-- > 0;) sum = std::abs(M[i * n + j]) + sum;
        nrm = std::max(nrm, sum);
    }
    return nrm;
}

/// 2^-s dt, by s exact halvings.
double aux_halved(double dt, int s) {
    for (int i = 0; i < s; ++i) dt *= 0.5;
    return dt;
}

/// The first order from max(taylor_terms, 2) (within 4096) with ||dt A|| <= (order + 2) / 2
/// (CORALean's reachOrderFloat).
std::optional<int> aux_order(const Mat &A, std::size_t n, double dt, int taylor_terms) {
    const double nrm = aux_norm(aux_scale(dt, A), n);
    const int first = std::max(taylor_terms, 2);
    for (int eta = first; eta < first + 4096; ++eta)
        if (nrm <= double(eta + 2) * 0.5) return eta;
    return std::nullopt;
}

/// The first number s < 128 of halvings with ||dt A / 2^s|| <= 1/2 (CORALean's halvingsFloat).
std::optional<int> aux_halvings(const Mat &A, std::size_t n, double dt) {
    for (int s = 0; s < 128; ++s)
        if (aux_norm(aux_scale(aux_halved(dt, s), A), n) <= 0.5) return s;
    return std::nullopt;
}

/// The bounds of the interval product [Llo, Lhi] [Rlo, Rhi]: per entry the sums over the inner
/// index, nested to the right, of the smallest and the largest corner product (Interval.matMul).
std::pair<Mat, Mat> aux_interval_matmul(const Mat &Llo, const Mat &Lhi, const Mat &Rlo,
                                        const Mat &Rhi, std::size_t n) {
    Mat lo(n * n), hi(n * n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double sum_lo = 0, sum_hi = 0;
            for (std::size_t k = n; k-- > 0;) {
                const double a = Llo[i * n + k], b = Lhi[i * n + k];
                const double c = Rlo[k * n + j], d = Rhi[k * n + j];
                const double p[4] = {aux_mul(a, c), aux_mul(a, d), aux_mul(b, c), aux_mul(b, d)};
                sum_lo = std::min(std::min(p[0], p[1]), std::min(p[2], p[3])) + sum_lo;
                sum_hi = std::max(std::max(p[0], p[1]), std::max(p[2], p[3])) + sum_hi;
            }
            lo[i * n + j] = sum_lo;
            hi[i * n + j] = sum_hi;
        }
    return {std::move(lo), std::move(hi)};
}

/// The bounds of c M with c within [lo, hi]: the hull of the two products of each entry.
std::pair<Mat, Mat> aux_box_scale(double lo, double hi, const Mat &M) {
    Mat inf(M.size()), sup(M.size());
    for (std::size_t e = 0; e < M.size(); ++e) {
        const double a = aux_mul(lo, M[e]), b = aux_mul(hi, M[e]);
        inf[e] = std::min(a, b);
        sup[e] = std::max(a, b);
    }
    return {std::move(inf), std::move(sup)};
}

/// [e^{A dt}] by scaling and squaring (expSquaringFloat Δt Δt 16 s): the Taylor sum of order 16 of
/// A dt / 2^s, the powers by left multiplication and the newest term added to the running sum,
/// widened by the remainder bound in every entry, then squared s times.
std::pair<Mat, Mat> aux_exp_squaring(const Mat &A, std::size_t n, double dt, int halvings) {
    // the Taylor sum of A dt / 2^s
    const Mat M = aux_scale(aux_halved(dt, halvings), A);
    Mat P = aux_identity(n), S = aux_scale(1.0, P);
    for (int k = 0; k < 16; ++k) {
        P = aux_matmul(M, P, n);
        const Mat term = aux_scale(1.0 / priv_float_factorial(k + 1), P);
        for (std::size_t e = 0; e < S.size(); ++e) S[e] = term[e] + S[e];
    }
    const double R = priv_float_remainder_bound(aux_norm(M, n), 16);
    Mat inf(S), sup(S);
    for (std::size_t e = 0; e < S.size(); ++e) {
        inf[e] -= R;
        sup[e] += R;
    }
    for (int i = 0; i < halvings; ++i) {
        auto squared = aux_interval_matmul(inf, sup, inf, sup, n);
        inf = std::move(squared.first);
        sup = std::move(squared.second);
    }
    return {std::move(inf), std::move(sup)};
}

/// The zonotope of the host data: center c (n) and the generators G (n, h) row-major.
Zonotope aux_zonotope(const Tensor &like, const Vec &c, const Vec &G, std::size_t n,
                      std::size_t h) {
    const auto N = static_cast<int64_t>(n), H = static_cast<int64_t>(h);
    return {Tensor::like(like, c, {N, 1}),
            h == 0 ? like.zeros_like({N, 0}) : Tensor::like(like, G, {N, H})};
}

/// The time-varying input without its center, kept to a box of n generators (a superset): the
/// series applies it term by term, once per order, so every generator it has is multiplied.
Zonotope aux_cap(const Zonotope &Z) { return Z.reduce(1); }

} // namespace


// ===========================================  MAIN  =========================================== //

// The series --------------------------------------------------------------------------------------

std::optional<PolySeries> PolySeries::make(const Tensor &A, double dt, int taylor_terms,
                                           bool raise) {
    const auto n = static_cast<std::size_t>(A.size(-1));
    const Mat a = A.data();
    const std::optional<int> eta = raise ? aux_order(a, n, dt, taylor_terms) : taylor_terms;
    if (!eta) return std::nullopt;
    const std::optional<int> halvings = aux_halvings(a, n, dt);
    if (!halvings) return std::nullopt;
    const Shape shape{A.size(-1), A.size(-1)};
    const auto box = [&](const std::pair<Mat, Mat> &b) {
        return Interval(Tensor::like(A, b.first, shape), Tensor::like(A, b.second, shape));
    };
    Interval e_adt = box(aux_exp_squaring(a, n, dt, *halvings));
    // the powers of [A, A] by left multiplication
    std::vector<Mat> powers{aux_identity(n)};
    for (int i = 1; i <= *eta; ++i) powers.push_back(aux_matmul(a, powers.back(), n));
    // the box of every order of the input solution
    std::vector<Interval> terms;
    for (int i = 0; i <= *eta; ++i) {
        const auto [lo, hi] = priv_float_taylor_coeff(dt, i + 1);
        terms.push_back(box(aux_box_scale(lo, hi, powers[std::size_t(i)])));
    }
    // the remainder E(dt) and [-dt E, dt E]
    const double E = priv_float_remainder_bound(aux_norm(aux_scale(dt, a), n), *eta);
    const double w = dt * E;
    const Mat wide(n * n, w), narrow(n * n, -w);
    const Interval spread = box({narrow, wide});
    // [F], the orders 2 to eta each spanning the interval between it and 0, then the remainder
    Mat lo(n * n, 0.0), hi(n * n, 0.0);
    for (int i = *eta; i >= 2; --i) {
        const double c = priv_float_corr_coeff_low(i) * priv_float_taylor_coeff(dt, i).second;
        const auto [t_lo, t_hi] = aux_box_scale(c, 0.0, powers[std::size_t(i)]);
        for (std::size_t e = 0; e < n * n; ++e) {
            lo[e] = std::min(t_lo[e], 0.0) + lo[e];
            hi[e] = std::max(t_hi[e], 0.0) + hi[e];
        }
    }
    for (std::size_t e = 0; e < n * n; ++e) {
        lo[e] -= E;
        hi[e] += E;
    }
    const Interval F = box({lo, hi});
    // the integral of e^{As}: the terms added in order, then the remainder
    Mat s_lo(n * n), s_hi(n * n);
    for (int i = 0; i <= *eta; ++i) {
        const std::vector<double> t_lo = terms[std::size_t(i)].inf().data(),
                                  t_hi = terms[std::size_t(i)].sup().data();
        for (std::size_t e = 0; e < n * n; ++e) {
            s_lo[e] = i == 0 ? t_lo[e] : s_lo[e] + t_lo[e];
            s_hi[e] = i == 0 ? t_hi[e] : s_hi[e] + t_hi[e];
        }
    }
    for (std::size_t e = 0; e < n * n; ++e) {
        s_lo[e] += -w;
        s_hi[e] += w;
    }
    return PolySeries(*eta, std::move(e_adt), F, box({s_lo, s_hi}), spread, std::move(terms));
}

// The map by an interval matrix -------------------------------------------------------------------

Zonotope priv_interval_map(const Interval &M, const Zonotope &Z) {
    const Tensor &like = Z.center();
    const auto m = static_cast<std::size_t>(M.inf().size(-2)),
               n = static_cast<std::size_t>(M.inf().size(-1));
    const auto h = static_cast<std::size_t>(Z.generators().size(-1));
    const Vec lo = M.inf().data(), hi = M.sup().data(), c = Z.center().data(),
              G = Z.generators().data();
    // mid(M) and the residual radius of the box M - mid(M)
    Mat C(m * n), R(m * n);
    for (std::size_t e = 0; e < m * n; ++e) {
        C[e] = (lo[e] + hi[e]) * 0.5;
        R[e] = std::max(std::abs(lo[e] - C[e]), std::abs(hi[e] - C[e]));
    }
    // what a coordinate j of the nominal part reaches: |c_j| + the row sum of |G|
    Vec reach(n);
    for (std::size_t j = 0; j < n; ++j) {
        double sum = 0;
        for (std::size_t l = h; l-- > 0;) sum = std::abs(G[j * h + l]) + sum;
        reach[j] = std::abs(c[j]) + sum;
    }
    const auto dot = [&](std::size_t i, auto column) {
        double sum = 0;
        for (std::size_t k = n; k-- > 0;) sum = aux_mul(C[i * n + k], column(k)) + sum;
        return sum;
    };
    // the center and the generators mapped by mid(M), the box generators of the residual
    Vec c_out(m), G_out(m * (h + m), 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        c_out[i] = dot(i, [&](std::size_t k) { return c[k]; });
        for (std::size_t l = 0; l < h; ++l)
            G_out[i * (h + m) + l] = dot(i, [&](std::size_t k) { return G[k * h + l]; });
        double box = 0;
        for (std::size_t k = n; k-- > 0;) box = aux_mul(R[i * n + k], reach[k]) + box;
        G_out[i * (h + m) + h + i] = box;
    }
    return aux_zonotope(like, c_out, G_out, m, h + m);
}

// The input solution ------------------------------------------------------------------------------

Zonotope PolySeries::input_solution(const Zonotope &Z) const {
    Zonotope S = priv_interval_map(terms_[0], Z);
    for (std::size_t i = 1; i < terms_.size(); ++i) S = S.plus(priv_interval_map(terms_[i], Z));
    return S.plus(priv_interval_map(spread_, Z));
}

// The steps ---------------------------------------------------------------------------------------

std::pair<PolyZonotope, PolyZonotope> priv_poly_window(const PolySeries &series, const Zonotope &V,
                                                       const PolyZonotope &X0, double zeta_z) {
    const int64_t n = X0.dim();
    const Tensor zero = X0.center().zeros_like();
    // the segment [0, u] of the center, V without it (a point has n zero generators)
    const Tensor &u = V.center();
    const Zonotope point_zero(zero, zero.zeros_like({n, n})), point_u(u, zero.zeros_like({n, n}));
    const Zonotope seg = priv_lin_comb(point_zero, point_u);
    const Zonotope V0 = aux_cap(Zonotope(zero, V.generators()));
    const Zonotope PuR = series.input_solution(seg).reduce_adaptive(zeta_z);
    const Zonotope PU = series.input_solution(V0).reduce_adaptive(zeta_z);
    // [F] X0, the hull of X0 and [e^{A dt}] X0, the hull of {0} and ([e^{A dt}] - I) X0
    const PolyZonotope FX = priv_poly_interval_map(series.F(), X0);
    const PolyZonotope e_adt_x = priv_poly_interval_map(series.e_adt(), X0);
    const PolyZonotope hom =
        priv_reduce_adaptive_poly(priv_poly_plus(priv_poly_lin_comb(X0, e_adt_x), FX), zeta_z);
    // the (n, n) identity (eye_like of the column vector alone would not be)
    const Tensor eye = X0.center().eye_like(n);
    const Interval e_adt_minus_i(series.e_adt().inf() + eye * -1.0,
                                 series.e_adt().sup() + eye * -1.0);
    const PolyZonotope origin(zero, zero.zeros_like({n, 0}), zero.zeros_like({n, n}), {}, {},
                              detail::Verbatim{});
    const PolyZonotope dhom = priv_reduce_adaptive_poly(
        priv_poly_plus(priv_poly_lin_comb(origin, priv_poly_interval_map(e_adt_minus_i, X0)), FX),
        zeta_z);
    return {priv_poly_plus(priv_poly_plus(hom, PuR), PU), priv_poly_plus(dhom, PuR.plus(PU))};
}

PolyZonotope priv_poly_point(const PolySeries &series, const Zonotope &V, const PolyZonotope &X0,
                             const PolyZonotope &Vs, double zeta_z) {
    const int64_t n = X0.dim();
    const Tensor zero = X0.center().zeros_like();
    // the input: V with the time-varying (independent) part of Vs, its center ũ held
    const Zonotope Vt = V.plus(Zonotope(zero, Vs.independent_generators()));
    const Tensor &u = Vt.center();
    const Zonotope point_u(u, zero.zeros_like({n, n}));
    const Zonotope Pu = priv_interval_map(series.integral(), point_u);
    const Zonotope PU =
        series.input_solution(aux_cap(Zonotope(zero, Vt.generators()))).reduce_adaptive(zeta_z);
    // [e^{A dt}] X0 reduced, the integral of the dependent part of Vs, the dependency on X0 kept
    const PolyZonotope e_adt_red =
        priv_reduce_adaptive_poly(priv_poly_interval_map(series.e_adt(), X0), zeta_z);
    const PolyZonotope dep(Vs.center(), Vs.generators(), zero.zeros_like({n, 0}), Vs.exponents(),
                           Vs.ids(), detail::Verbatim{});
    const PolyZonotope Ps = priv_poly_interval_map(series.integral(), dep);
    return priv_poly_plus(priv_poly_plus(priv_poly_exact_plus(e_adt_red, Ps).compact_gens(), Pu),
                          PU);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
