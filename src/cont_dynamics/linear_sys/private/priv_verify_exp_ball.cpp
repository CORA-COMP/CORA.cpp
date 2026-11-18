// priv_verify_exp_ball - e^{A dt} as a matrix and a balanced radius, and the growth of e^{tA}
//
// Every rounded operation is the one of CORALean's definition at nearest, in its order: a sum over
// an index from the last index down (sumUp nests to the right), the Taylor sum from the term of
// order 0 up, each new term added in front (matPlus of the term and the sum). The products with A
// are host loops over its nonzeros when it is sparse, the squarings host loops while D is sparse
// and dense products of the backend after; both sum over the inner index from the last down, as
// the lean backend's products do.
//
// Syntax:   auto ball = priv_verify_exp_ball(A, s, dt);
// Inputs:   A - (n, n) system matrix
//           s - positive scales
//           dt - step size
// Outputs:  ball - the matrix D and the radius r, empty if no halving count or order is found
// See also: priv_verify_exp_ball.h
//
// CORA.cpp only: the float definitions of CORALean's MatExp.expBallFloat,
// step_scaledNormUpFloat and logNormUpFloat.
// Authors:       Claude Opus 5.5
// Written:       17-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_verify_exp_ball.h"
#include "cont_dynamics/linear_sys/private/priv_float_scalars.h"
#include "cora/global/threads.h"

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr int kHalvings = 1100; // the halving counts tried
constexpr int kFirstOrder = 6;  // the Taylor orders tried: 6 (so nu / (K + 2) <= 1/2) to 60
constexpr int kOrders = 60;
constexpr double kTheta = 4.0; // the balanced norm of A dt / 2^sigma at most
constexpr int kTolExp = -80;   // the remainder bound sought, 2^-80

// Sparse products ---------------------------------------------------------------------------------

/// A matrix by rows of nonzeros.
struct Rows {
    std::vector<int64_t> ptr, col;
    std::vector<double> val;
};

/// The nonzeros of the row-major (n, n) matrix m by rows.
Rows aux_rows(const std::vector<double> &m, int64_t n) {
    Rows r;
    r.ptr.assign(1, 0);
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < n; ++j)
            if (m[std::size_t(i) * n + j] != 0) {
                r.col.push_back(j);
                r.val.push_back(m[std::size_t(i) * n + j]);
            }
        r.ptr.push_back(static_cast<int64_t>(r.val.size()));
    }
    return r;
}

/// X P for X by rows of nonzeros and a dense P: each entry sums x_ik p_kj over k from the last
/// down, zero terms skipped (a sum from +0 is never -0, so the skips change no bit). Rows in
/// parallel from 2^24 flops: idle threads spin after a region, about a millisecond of CPU each.
std::vector<double> aux_rows_times(const Rows &X, const std::vector<double> &P, int64_t n) {
    constexpr long long kFlopsPerWork = 256; // threads_for's work units in flops of this product
    std::vector<double> out(std::size_t(n) * n, 0.0);
    [[maybe_unused]] const int threads =
        threads_for(n, static_cast<long long>(X.val.size()) * n / kFlopsPerWork);
#pragma omp parallel for num_threads(threads) schedule(static) if (threads > 1)
    for (int64_t i = 0; i < n; ++i) {
        double *row = &out[std::size_t(i) * n];
        for (int64_t q = X.ptr[i + 1] - 1; q >= X.ptr[i]; --q) {
            const double x = X.val[q];
            const double *p = &P[std::size_t(X.col[q]) * n];
            for (int64_t j = 0; j < n; ++j) row[j] = x * p[j] + row[j];
        }
    }
    return out;
}

/// Whether the rows of nonzeros are fewer than a tenth of the entries: then the host product.
bool aux_sparse(const Rows &r, int64_t n) {
    return double(r.val.size()) < 0.1 * double(n) * double(n);
}

/// X P, as host rows of nonzeros if X is sparse, else as the backend's product.
std::vector<double> aux_times(const Tensor &like, const std::vector<double> &X, const Rows &rows,
                              const std::vector<double> &P, int64_t n) {
    if (aux_sparse(rows, n)) return aux_rows_times(rows, P, n);
    return Tensor::like(like, X, {n, n}).matmul(Tensor::like(like, P, {n, n})).data();
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Norms -------------------------------------------------------------------------------------------

double priv_verify_scaled_norm_up(const std::vector<double> &m, const std::vector<double> &s,
                                  int64_t n) {
    // the largest row sum of |m_jk| s_k / s_j, each sum from the last index down
    double out = 0;
    for (int64_t j = 0; j < n; ++j) {
        double sum = 0;
        for (int64_t k = n - 1; k >= 0; --k)
            sum = (std::abs(m[std::size_t(j) * n + k]) * s[k]) / s[j] + sum;
        out = std::max(out, sum);
    }
    return out;
}

double priv_verify_log_norm_up(const std::vector<double> &a, const std::vector<double> &s,
                               int64_t n) {
    // the largest a_jj plus the row sum of |a_jk| s_k / s_j off the diagonal
    double out = 0;
    for (int64_t j = 0; j < n; ++j) {
        double sum = 0;
        for (int64_t k = n - 1; k >= 0; --k)
            if (k != j) sum = (std::abs(a[std::size_t(j) * n + k]) * s[k]) / s[j] + sum;
        const double row = a[std::size_t(j) * n + j] + sum;
        out = j == 0 ? row : std::max(out, row);
    }
    return out;
}

// The exponential ---------------------------------------------------------------------------------

std::optional<VerifyExpBall> priv_verify_exp_ball(const Tensor &A, const std::vector<double> &s,
                                                  double dt) {
    const int64_t n = A.shape()[0];
    const std::vector<double> a = A.data();
    // the halvings: nu / 2^sigma <= 4 for the balanced norm nu of A dt
    const double nu = priv_verify_scaled_norm_up(a, s, n) * dt;
    // halving is `half`, x * 0.5, one at a time (as rounded as CORALean's below the normal range)
    int sigma = 0;
    double h = dt, nu_h = nu;
    while (sigma < kHalvings && !(nu_h <= kTheta)) {
        ++sigma;
        h *= 0.5;
        nu_h *= 0.5;
    }
    if (sigma == kHalvings) return std::nullopt;
    // the order: the first from 6 whose remainder bound is at most 2^-80
    int order = kFirstOrder;
    while (order <= kOrders &&
           !(priv_float_remainder_bound(nu_h, order) <= std::ldexp(1.0, kTolExp)))
        ++order;
    if (order > kOrders) return std::nullopt;
    // the Taylor sum of A h: the powers X^i, X = A h, each term (1/i!) X^i added in front
    std::vector<double> X(a.size()), P(a.size(), 0.0);
    for (std::size_t e = 0; e < a.size(); ++e) X[e] = h * a[e];
    for (int64_t i = 0; i < n; ++i) P[std::size_t(i) * n + i] = 1.0;
    std::vector<double> S = P;
    const Rows Xrows = aux_rows(X, n);
    for (int i = 1; i <= order; ++i) {
        P = aux_times(A, X, Xrows, P, n);
        const double c = 1.0 / priv_float_factorial(i);
        for (std::size_t e = 0; e < S.size(); ++e) S[e] = c * P[e] + S[e];
    }
    // the radius of the sum: the remainder (its box is a point at nearest, of width 0)
    double r = priv_float_remainder_bound(nu_h, order);
    // the squarings: D D and the radius (2 ||D|| + r) r
    for (int j = 0; j < sigma; ++j) {
        const double p = priv_verify_scaled_norm_up(S, s, n);
        S = aux_times(A, S, aux_rows(S, n), S, n);
        r = ((p + p) + r) * r;
    }
    return VerifyExpBall{Tensor::like(A, S, {n, n}), r};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
