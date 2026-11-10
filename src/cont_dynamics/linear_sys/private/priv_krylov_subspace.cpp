// priv_krylov_subspace - the Arnoldi iteration, the order of a Krylov subspace by Jawecki's bound
//
// Arnoldi (CORA's arnoldi): v_1 = v / |v|, w = A v_j swept against v_1..v_j (modified Gram-Schmidt,
// h_ij = w'v_i), h_{j+1,j} = |w|, v_{j+1} = w / h_{j+1,j}; a happy breakdown (h_{j+1,j} <= eps)
// stops it at j. The order k tries k0 - step + step, + 2 step, ... until |v| tau gamma t^k / k!
// <= krylov_error (Err_a of Jawecki et al.), tau = h_{k+1,k}, gamma the product of h_{i+1,i}.
// CORA evaluates tau gamma with 34 digits (mp); here its logarithm, which neither overflows nor
// underflows. t^k / k! is CORA's double: k! is Inf from k = 171 on, so the bound is then 0.
// Arnoldi is extended pass by pass instead of rerun per order (the same passes) and stops at n
// columns, where CORA's would go on with a numerically zero residual.
//
// Syntax:   KrylovSubspace S = priv_krylov_subspace(A, v, order, error, step, t_final, like);
// Inputs:   A - the products with the matrix
//           v - the start vector (host)
//           order - the order of the previous subspace (in), the order found (out)
//           krylov_error - the bound to reach
//           step - how much the order grows per try
//           t_final - the time of the bound
//           like - backend, device and dtype of V and H
// Outputs:  S - V (n, k), H (k, k) and |v|
// See also: priv_reach_krylov
//
// CORA: global/functions/helper/dynamics/contDynamics/linearSys/arnoldi.m
// CORA: contDynamics/@linearSys/private/priv_subspace_Krylov_individual_Jawecki.m
// CORA authors: Matthias Althoff
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_krylov.h"
#include "cora/global/sparse_matrix.h"
#include "cora/global/tensor/tolerance.h"
#include "matrix_set/interval_matrix/private/priv_factorial.h"

#include <cmath>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Arnoldi -----------------------------------------------------------------------------------------

/// The Arnoldi vectors (columns on the host) and the entries h_ij of H by column.
struct Arnoldi {
    std::vector<std::vector<double>> V;
    std::vector<std::vector<double>> H; ///< column j holds h_1j .. h_{j+1,j}
    bool breakdown = false;
};

double aux_dot(const std::vector<double> &a, const std::vector<double> &b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

/// The passes of Arnoldi up to `passes` (fewer after a happy breakdown or at n columns).
void aux_extend(Arnoldi &state, const HostMatVec &A, int64_t passes) {
    const auto n = size_t(A.n);
    std::vector<double> w(n);
    while (!state.breakdown && int64_t(state.H.size()) < passes) {
        const size_t j = state.H.size();
        A.apply(state.V[j].data(), w.data());
        std::vector<double> h(j + 2);
        // the sweep against the earlier vectors, each with the w it has left
        for (size_t i = 0; i <= j; ++i) {
            h[i] = aux_dot(w, state.V[i]);
            for (size_t r = 0; r < n; ++r) w[r] -= h[i] * state.V[i][r];
        }
        h[j + 1] = std::sqrt(aux_dot(w, w));
        state.H.push_back(h);
        if (h[j + 1] <= machine_eps() || j + 1 == n) {
            state.breakdown = true;
            break;
        }
        for (double &x : w) x /= h[j + 1];
        state.V.push_back(w);
    }
}

// The error bound ---------------------------------------------------------------------------------

/// log(tau gamma t^k / k!) of the first k passes (-inf: the bound is 0, +inf: it is not finite).
double aux_log_bound(const Arnoldi &state, int64_t k, double t_final) {
    const double right = std::pow(t_final, double(k)) / priv_factorial(k);
    if (right == 0) return -std::numeric_limits<double>::infinity();
    if (!std::isfinite(right)) return std::numeric_limits<double>::infinity();
    double log_bound = std::log(state.H[size_t(k) - 1][size_t(k)]) + std::log(right);
    for (int64_t i = 1; i < k; ++i) log_bound += std::log(state.H[size_t(i) - 1][size_t(i)]);
    return std::isnan(log_bound) ? std::numeric_limits<double>::infinity() : log_bound;
}

/// Whether |v| exp(log_bound) > krylov_error.
bool aux_too_large(double v_norm, double log_bound, double krylov_error) {
    if (log_bound == -std::numeric_limits<double>::infinity()) return false;
    if (krylov_error == 0) return true;
    return std::log(v_norm) + log_bound > std::log(krylov_error);
}

} // namespace


// ===========================================  MAIN  =========================================== //

KrylovSubspace priv_krylov_subspace(const HostMatVec &A, const std::vector<double> &v,
                                    int64_t &order, double krylov_error, int krylov_step,
                                    double t_final, const Tensor &like) {
    const double v_norm = std::sqrt(aux_dot(v, v));
    Arnoldi state;
    state.V.push_back(v);
    for (double &x : state.V[0]) x /= v_norm;
    // the order of the first try, as CORA's: the last order found, then krylov_step more
    int64_t k = std::min(std::max<int64_t>(order - krylov_step, 1), A.n);
    double log_bound = std::numeric_limits<double>::infinity();
    while (aux_too_large(v_norm, log_bound, krylov_error) && k <= A.n) {
        k += krylov_step;
        aux_extend(state, A, k);
        if (state.breakdown) {
            k = int64_t(state.H.size());
            break;
        }
        log_bound = aux_log_bound(state, k, t_final);
    }
    order = k;
    // V (n, k) and H (k, k) from the first k vectors and columns
    const auto n = size_t(A.n);
    std::vector<double> V(n * size_t(k)), H(size_t(k * k), 0.0);
    for (size_t j = 0; j < size_t(k); ++j) {
        for (size_t r = 0; r < n; ++r) V[r * size_t(k) + j] = state.V[j][r];
        for (size_t i = 0; i < state.H[j].size() && i < size_t(k); ++i)
            H[i * size_t(k) + j] = state.H[j][i];
    }
    return {Tensor::like(like, V, {A.n, k}), Tensor::like(like, H, {k, k}), v_norm, k};
}

HostMatVec priv_state_matvec(const LinearSys &sys) {
    const int64_t n = sys.dim();
    if (const SparseMatrix *S = sys.A_sparse())
        return {n, [S](const double *x, double *y) { S->matvec(x, y); }};
    // a dense A through the kernel of a sparse one: the same sets on every compiler (a*b + c fuses)
    auto D = std::make_shared<const SparseMatrix>(SparseMatrix::from_dense(sys.A()));
    return {n, [D](const double *x, double *y) { D->matvec(x, y); }};
}

HostMatVec priv_integrator_matvec(const HostMatVec &A, const std::vector<double> &u) {
    const int64_t n = A.n;
    return {n + 1, [A, u, n](const double *x, double *y) {
                A.apply(x, y);
                for (int64_t r = 0; r < n; ++r) y[r] += u[size_t(r)] * x[n];
                y[n] = 0.0;
            }};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
