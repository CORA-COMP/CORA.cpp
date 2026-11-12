// priv_krylov_lyapunov - a checked Lyapunov weight of a matrix and the error of windows in it
//
// For a stable non-normal A the Gershgorin bound mu_A of the quadratic form can be large (551 on
// RND02) although e^{tA} decays. A weight P (A'P + PA = -I, solved by priv_lyapunov_solve and only
// checked here) gives ||e^{tA}|| <= sqrt(b / a') e^{rate t}, rate < 0, and the 2-norm error of
// restarted windows of local errors eps_j is carried in it, beta_{j+1} = e^{rate dt_j} beta_j +
// sqrt(b) eps_j, bound = beta_N / sqrt(a'). The check and the carry are CORALean's
// Krylov.lyapunovCertificateFloat and Krylov.restartedErrorBoundFloat at nearest, bit for bit:
// E = |P - aI - L'L| and W = |PA + (PA)' + I| are bounded by Gershgorin (quadBoundUp), a' = a -
// rho_E, rate = -(1 - rho_W) / (2 b), and P is accepted if exactly symmetric with a', the gap and
// b positive. Every sum runs from its last term and no product is fused.
//
// Syntax:   LyapunovCertificate c = priv_lyapunov_certificate(A, P, L, a);
//           RestartedBound r(c);   r.step(dt, eps);   double bound = r.bound();
// Inputs:   A - the sparse matrix
//           P - the weight, row-major n x n
//           L - a factor of P - aI (L'L = P - aI), row-major n x n
//           a - the shift
//           dt, eps - the length of a window and its local 2-norm error
// Outputs:  c - ok, lower (a'), upper (b) and rate
//           bound - the 2-norm error after the windows, rounded up
// See also: priv_lyapunov_solve, priv_krylov_windows
//
// CORA.cpp only: CORA carries the error of nothing; CORALean's Krylov.lyapunovCertificateFloat
// and Krylov.restartedErrorBoundFloat are the definitions
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_krylov.h"
#include "cont_dynamics/linear_sys/private/priv_krylov_rounding.h"

#include <algorithm>
#include <chrono>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Krylov.quadBoundUp of the dense row-major n x n matrix: the maximum, seeded at the last row, of
/// M_ii + half(sum over j != i from the last of |M_ij| + |M_ji|).
double aux_quad_bound(const std::vector<double> &M, int64_t n) {
    std::vector<double> rows(static_cast<size_t>(n));
#pragma omp parallel for schedule(static) if (n >= 256)
    for (int64_t i = 0; i < n; ++i) {
        const double other = lean_sum(n, [&](int64_t j) {
            return j == i
                       ? 0.0
                       : lean_absolute(M[size_t(i * n + j)]) + lean_absolute(M[size_t(j * n + i)]);
        });
        rows[size_t(i)] = M[size_t(i * n + i)] + lean_half(other);
    }
    double bound = 0.0;
    for (int64_t i = n; i-- > 0;)
        bound = i == n - 1 ? rows[size_t(i)] : lean_maximum(rows[size_t(i)], bound);
    return n == 0 ? 0.0 : bound;
}

/// The transpose of the row-major n x n matrix.
std::vector<double> aux_transpose(const std::vector<double> &M, int64_t n) {
    std::vector<double> T(M.size());
#pragma omp parallel for schedule(static) if (n >= 256)
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < n; ++j) T[size_t(j * n + i)] = M[size_t(i * n + j)];
    return T;
}

/// C_E = |P - aI - L'L| entrywise (matMulDown/Up and the roundings of the difference are the
/// machine's at nearest), formed in place of L'L.
std::vector<double> aux_error_bounds(const std::vector<double> &P, const std::vector<double> &L,
                                     double a, int64_t n) {
    const std::vector<double> Lt = aux_transpose(L, n);
    std::vector<double> C(size_t(n * n), 0.0);
#pragma omp parallel for schedule(static) if (n >= 64)
    for (int64_t i = 0; i < n; ++i) {
        double *row = &C[size_t(i * n)];
        // (L'L)_ij = sum over k from the last of L_ki L_kj
        for (int64_t k = n; k-- > 0;) {
            const double factor = Lt[size_t(i * n + k)];
            if (factor == 0.0) continue;
            const double *Lk = &L[size_t(k * n)];
            for (int64_t j = 0; j < n; ++j) {
                const double product = factor * Lk[j];
                row[j] = product + row[j];
            }
        }
        for (int64_t j = 0; j < n; ++j) {
            const double shifted = P[size_t(i * n + j)] - (i == j ? a : 0.0);
            const double e = shifted - row[j];
            row[j] = lean_maximum(lean_absolute(e), lean_absolute(e));
        }
    }
    return C;
}

/// C_W = |PA + (PA)' + I| entrywise, PA by the rows of the sparse A (a zero entry of A adds a
/// zero).
std::vector<double> aux_flow_bounds(const std::vector<double> &P, const SparseMatrix &A,
                                    int64_t n) {
    const std::vector<int64_t> &start = A.row_start(), &cols = A.columns();
    const std::vector<double> &vals = A.values();
    std::vector<double> PA(size_t(n * n), 0.0);
#pragma omp parallel for schedule(static) if (n >= 64)
    for (int64_t i = 0; i < n; ++i) {
        double *row = &PA[size_t(i * n)];
        // (PA)_ij = sum over k from the last of P_ik A_kj
        for (int64_t k = n; k-- > 0;) {
            const double factor = P[size_t(i * n + k)];
            if (factor == 0.0) continue;
            for (int64_t e = start[size_t(k)]; e < start[size_t(k) + 1]; ++e) {
                const double product = factor * vals[size_t(e)];
                row[cols[size_t(e)]] = product + row[cols[size_t(e)]];
            }
        }
    }
    // the pairs (i, j), (j, i) have the same sum bit for bit
#pragma omp parallel for schedule(static) if (n >= 256)
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = i; j < n; ++j) {
            const double sum = PA[size_t(i * n + j)] + PA[size_t(j * n + i)];
            const double w = sum + (i == j ? 1.0 : 0.0);
            PA[size_t(i * n + j)] = PA[size_t(j * n + i)] =
                lean_maximum(lean_absolute(w), lean_absolute(w));
        }
    }
    return PA;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The certificate ---------------------------------------------------------------------------------

LyapunovCertificate priv_lyapunov_certificate(const SparseMatrix &A, const std::vector<double> &P,
                                              const std::vector<double> &L, double a) {
    const int64_t n = A.rows();
    // 1.-2. E = |P - aI - L'L|, lower = a - rho_E, upper = quadBoundUp P
    const double rho_E = aux_quad_bound(aux_error_bounds(P, L, a, n), n);
    LyapunovCertificate c;
    c.lower = a - rho_E;
    c.upper = aux_quad_bound(P, n);
    // 3.-4. W = |PA + (PA)' + I|, gap = 1 - rho_W, rate = -(gap / (upper + upper))
    const double gap = 1.0 - aux_quad_bound(aux_flow_bounds(P, A, n), n);
    c.rate = -(gap / (c.upper + c.upper));
    // 5. P exactly symmetric, and lower, gap, upper positive
    bool symmetric = true;
    for (int64_t i = 0; i < n && symmetric; ++i)
        for (int64_t j = 0; j < n; ++j)
            if (!(P[size_t(i * n + j)] <= P[size_t(j * n + i)] &&
                  P[size_t(j * n + i)] <= P[size_t(i * n + j)])) {
                symmetric = false;
                break;
            }
    c.ok = symmetric && 0.0 < c.lower && 0.0 < gap && 0.0 < c.upper;
    return c;
}

// The weight of a matrix --------------------------------------------------------------------------

const LyapunovCertificate &HostMatVec::lyapunov() const {
    LyapunovSlot &slot = *lyapunov_slot;
    if (slot.formed) return slot.certificate;
    // the sign function solve if the check accepts it, else the Schur solve
    double solve = 0.0, check = 0.0;
    for (const LyapunovMethod method : {LyapunovMethod::sign, LyapunovMethod::schur}) {
        const auto t0 = std::chrono::steady_clock::now();
        const LyapunovWeight w = priv_lyapunov_solve(*M, method);
        const auto t1 = std::chrono::steady_clock::now();
        if (!w.P.empty()) slot.certificate = priv_lyapunov_certificate(*M, w.P, w.L, w.a);
        const auto t2 = std::chrono::steady_clock::now();
        solve += std::chrono::duration<double>(t1 - t0).count();
        check += std::chrono::duration<double>(t2 - t1).count();
        if (!w.P.empty() && slot.certificate.ok) {
            slot.method = method == LyapunovMethod::sign ? "sign" : "schur";
            break;
        }
    }
    slot.solve_seconds = solve;
    slot.check_seconds = check;
    slot.formed = true;
    return slot.certificate;
}

// The restarted error -----------------------------------------------------------------------------

RestartedBound::RestartedBound(const LyapunovCertificate &c)
    : c_(c), root_(std::sqrt(c.upper)), scale_(std::sqrt(1.0 / c.lower)) {}

void RestartedBound::step(double dt, double eps) {
    // beta <- e^{rate dt} beta + sqrt(upper) eps, each rounded
    const double decay = std::exp(c_.rate * dt);
    const double kept = decay * beta_;
    const double added = root_ * eps;
    beta_ = kept + added;
}

double RestartedBound::bound() const { return beta_ * scale_; }

double RestartedBound::window_bound(double eps) const {
    const double added = root_ * eps;
    return (beta_ + added) * scale_;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
