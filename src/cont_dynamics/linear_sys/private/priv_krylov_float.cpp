// priv_krylov_float - the float Arnoldi iteration of CORALean and the bound of its error
//
// CORA's Arnoldi (arnoldi.m) and Jawecki's order test, repaired by CORALean for every matrix, at
// the dtype nearest (the oracle's rounding, which CORA.cpp computes with): v(1) = c / |c|, each
// pass w = A v(k) swept against the earlier vectors by modified Gram-Schmidt, h_{k+1,k} = |w|,
// v(k+1) = w / h_{k+1,k} or 0 at a happy breakdown (Krylov.arnoldiFloat). The error
//   ||e^{tA}c - nu V_k e^{tT_k} e_1|| <= e^{t mu} (|nu| (|w| Prod_{j<=k} |h_{j+1,j}| t / j
//       + (Sum_{j<k} eps_j) t) + rho)
// (Krylov.errorBoundFloat) needs no orthogonality and no nonexpansive A: mu = max(0, mu_A, mu_T)
// is Gershgorin's bound of the quadratic forms of A and T_k, eps_j the residual norm of column j,
// rho the distance of nu v(1) from c. The decay product runs by ratios, so no factorial overflows
// (CORA's k! is infinite from k = 171 on and its bound then reads 0). Every sum runs from its last
// term and no product is fused: the results are the oracle's bit for bit (test_lean_krylov).
//
// Syntax:   ArnoldiFloat s = priv_arnoldi_start(c);   priv_arnoldi_extend(s, A, passes, eps);
//           double b = priv_krylov_error_bound(s, A.quad_bound(), k, t, q);
// Inputs:   c - the start vector
//           A - the products with the matrix
//           passes - the number of Arnoldi passes
//           breakdown - the threshold of a happy breakdown
//           k - the order
//           t - the time of the bound
//           q - the squarings of the exponential (its range is t mu <= 2^(q-1))
// Outputs:  s - the Arnoldi data
//           b - the bound of the Krylov error over [0, t]
// See also: priv_krylov_subspace
//
// CORA: global/functions/helper/dynamics/contDynamics/linearSys/arnoldi.m
// CORA: contDynamics/@linearSys/private/priv_subspace_Krylov_individual_Jawecki.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_krylov.h"
#include "cont_dynamics/linear_sys/private/priv_krylov_rounding.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether every entry is zero.
bool aux_is_zero(const std::vector<double> &x) {
    return std::all_of(x.begin(), x.end(), [](double v) { return v == 0.0; });
}

/// The entry (i, j) of the Hessenberg matrix: zero below the subdiagonal.
double aux_h(const ArnoldiFloat &s, int64_t i, int64_t j) {
    return i <= j + 1 ? s.H[size_t(j)][size_t(i)] : 0.0;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The iteration -----------------------------------------------------------------------------------

double priv_krylov_norm_up(const std::vector<double> &x) {
    // sqrtUp of the squares of the magnitudes, summed from the last entry
    return std::sqrt(lean_sum(int64_t(x.size()), [&](int64_t i) {
        const double m = lean_absolute(x[size_t(i)]);
        return m * m;
    }));
}

ArnoldiFloat priv_arnoldi_start(const std::vector<double> &c) {
    // 1. v(1) <- c / |c|
    ArnoldiFloat s;
    s.c = c;
    s.nu = priv_krylov_norm_up(c);
    std::vector<double> v(c.size());
    for (size_t r = 0; r < c.size(); ++r) v[r] = c[r] / s.nu;
    s.V.push_back(std::move(v));
    return s;
}

void priv_arnoldi_extend(ArnoldiFloat &s, const HostMatVec &A, int64_t passes, double breakdown) {
    const auto n = size_t(A.n);
    std::vector<double> w(n), w0;
    while (s.passes() < passes) {
        const auto k = size_t(s.passes());
        std::vector<double> h(k + 2, 0.0), vnext(n, 0.0);
        double epsilon = 0.0;
        // a zero vector gives w = 0, h = 0, |w| = 0 and v(k+2) = 0 exactly: no product needed
        if (!aux_is_zero(s.V[k])) {
            // 3. w <- A v(k)
            A.apply(s.V[k].data(), w.data());
            w0 = w;
            // 4.-7. w swept against v(1), ..., v(k+1), each h from the w the earlier ones left
            for (size_t j = 0; j <= k; ++j) {
                const std::vector<double> &v = s.V[j];
                h[j] = lean_sum(int64_t(n), [&](int64_t r) { return w[size_t(r)] * v[size_t(r)]; });
                for (size_t r = 0; r < n; ++r) {
                    const double product = h[j] * v[r];
                    w[r] = w[r] - product;
                }
            }
            // 8. h_{k+1,k} = |w|, 9.-12. v(k+2) = w / h_{k+1,k} unless a happy breakdown
            h[k + 1] = priv_krylov_norm_up(w);
            if (!(h[k + 1] <= breakdown))
                for (size_t r = 0; r < n; ++r) vnext[r] = w[r] / h[k + 1];
            // the residual of the column, A v(k) - sum_i h_{i,k} v(i), each sum from its last term
            std::vector<double> column(n, 0.0);
            for (size_t i = k + 2; i-- > 0;) {
                const std::vector<double> &vi = i == k + 1 ? vnext : s.V[i];
                for (size_t r = 0; r < n; ++r) {
                    const double term = h[i] * vi[r];
                    column[r] = term + column[r];
                }
            }
            for (size_t r = 0; r < n; ++r) column[r] = lean_absolute(w0[r] - column[r]);
            epsilon = priv_krylov_norm_up(column);
        }
        s.H.push_back(std::move(h));
        s.V.push_back(std::move(vnext));
        s.residual.push_back(epsilon);
    }
}

// The bound ---------------------------------------------------------------------------------------

double priv_hessenberg_quad_bound(const ArnoldiFloat &s, int64_t k) {
    if (k == 0) return 0.0;
    double bound = 0.0;
    for (int64_t i = k; i-- > 0;) {
        const double other = lean_sum(k, [&](int64_t j) {
            return j == i ? 0.0 : lean_absolute(aux_h(s, i, j)) + lean_absolute(aux_h(s, j, i));
        });
        const double row = aux_h(s, i, i) + lean_half(other);
        bound = i == k - 1 ? row : lean_maximum(row, bound);
    }
    return bound;
}

KrylovBoundFn::KrylovBoundFn(const ArnoldiFloat &s, double mu_A, int64_t k) : s_(s), k_(k) {
    mu_ = lean_maximum(0.0, lean_maximum(mu_A, priv_hessenberg_quad_bound(s, k)));
    wn_ = priv_krylov_norm_up(s.V[size_t(k)]);
    // the residuals' share and the start vector's error
    share_ = lean_sum(k, [&](int64_t j) { return s.residual[size_t(j)]; });
    std::vector<double> d(s.c.size());
    for (size_t r = 0; r < d.size(); ++r) {
        const double product = s.nu * s.V[0][r];
        const double diff = s.c[r] - product;
        d[r] = lean_maximum(lean_absolute(diff), lean_absolute(diff));
    }
    rho_ = priv_krylov_norm_up(d);
}

double KrylovBoundFn::at(double t) const {
    // the decay product |w| Prod_{j<=k} |h_{j+1,j}| t / j, by ratios
    double decay = wn_;
    for (int64_t j = 0; j < k_; ++j) {
        const double scaled = decay * lean_absolute(aux_h(s_, j + 1, j));
        const double times = scaled * t;
        decay = times / lean_maximum(1.0, double(j + 1));
    }
    const double growth = std::exp(t * mu_);
    const double inner = lean_absolute(s_.nu) * (decay + share_ * t);
    return growth * (inner + rho_);
}

bool KrylovBoundFn::ok(double t, double tol, int q) const {
    if (k_ < 1) return false;
    const double range = double(int64_t(1) << std::max(q - 1, 0));
    return t * mu_ <= range && at(t) <= tol;
}

double priv_krylov_error_bound(const ArnoldiFloat &s, double mu_A, int64_t k, double t, int q) {
    (void)q; // the machine's exponential at nearest, whatever the squarings
    return KrylovBoundFn(s, mu_A, k).at(t);
}

bool priv_krylov_order_ok(const ArnoldiFloat &s, double mu_A, int64_t k, double t, double tol,
                          int q) {
    return KrylovBoundFn(s, mu_A, k).ok(t, tol, q);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
