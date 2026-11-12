// priv_krylov_subspace - the order of a Krylov subspace by the error bound of CORALean
//
// CORA grows the order k of the subspace of v until Jawecki's bound |v| tau gamma t^k / k! at the
// final time is at most krylov_error, a test that is valid for a nonexpansive A only and reads 0
// from k = 171 on (k! overflows). Here the order grows by krylov_step until CORALean's
// Krylov.errorBoundFloat is at most krylov_error: the bound of the error of the stored basis for
// every matrix (priv_krylov_float.cpp; Krylov.subspaceFloat_error_le). Arnoldi is extended pass by
// pass instead of rerun per order (the same passes), and stops at n columns.
//
// Syntax:   KrylovSubspace S = priv_krylov_subspace(A, v, order, error, step, t_final, like);
// Inputs:   A - the products with the matrix
//           v - the start vector (host)
//           order - the order of the first try (in), the order found (out)
//           krylov_error - the tolerance of the bound
//           step - how much the order grows per try
//           t_final - the time of the bound
//           like - backend, device and dtype of V and H
// Outputs:  S - V (n, k), H (k, k), |v|, the order, the bound and whether it meets the tolerance
// See also: priv_reach_krylov, priv_krylov_float
//
// CORA: contDynamics/@linearSys/private/priv_subspace_Krylov_individual_Jawecki.m
// CORA authors: Matthias Althoff
// Authors:       Claude Opus 5.5, Claude Sonnet 5.5
// Written:       09-November-2026
// Last update:   12-November-2026
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_krylov.h"
#include "cora/global/error.h"
#include "cora/global/sparse_matrix.h"
#include "cora/global/tensor/tolerance.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

// The order search --------------------------------------------------------------------------------

KrylovSearch priv_krylov_search(const HostMatVec &A, const std::vector<double> &c, int64_t k0,
                                int64_t step, double tf, double tol, int64_t fuel, int q,
                                double breakdown) {
    KrylovSearch out;
    out.state = priv_arnoldi_start(c);
    const double mu_A = A.quad_bound();
    int64_t k = std::min(k0, A.n);
    for (int64_t tries = 0;; ++tries) {
        priv_arnoldi_extend(out.state, A, k, breakdown);
        out.order = k;
        out.ok = priv_krylov_order_ok(out.state, mu_A, k, tf, tol, q);
        out.bound = priv_krylov_error_bound(out.state, mu_A, k, tf, q);
        if (out.ok || A.n <= k || tries >= fuel) break;
        k = std::min(k + step, A.n);
    }
    return out;
}

KrylovSubspace priv_krylov_subspace(const HostMatVec &A, const std::vector<double> &v,
                                    int64_t &order, double krylov_error, int krylov_step,
                                    double t_final, const Tensor &like) {
    // the order of the first try, then krylov_step more per try until the bound meets the tolerance
    const KrylovSearch found =
        priv_krylov_search(A, v, std::max<int64_t>(order, 1), krylov_step, t_final, krylov_error,
                           A.n, kKrylovSquarings, machine_eps());
    const ArnoldiFloat &s = found.state;
    const int64_t k = found.order, n = A.n;
    order = k;
    // V (n, k) and H (k, k) from the first k vectors and columns
    std::vector<double> V(size_t(n * k)), H(size_t(k * k), 0.0);
    for (int64_t j = 0; j < k; ++j) {
        for (int64_t r = 0; r < n; ++r) V[size_t(r * k + j)] = s.V[size_t(j)][size_t(r)];
        for (int64_t i = 0; i <= std::min(j + 1, k - 1); ++i)
            H[size_t(i * k + j)] = s.H[size_t(j)][size_t(i)];
    }
    // the tensors like `like`, with the scale, the order and the bound
    return {Tensor::like(like, V, {n, k}),
            Tensor::like(like, H, {k, k}),
            s.nu,
            k,
            found.ok,
            found.bound};
}

void priv_krylov_require_ok(const KrylovSubspace &S, double krylov_error, const char *caller) {
    if (S.ok) return;
    CORA_FAIL(Error::not_converged,
              std::string(caller) + ": the Krylov subspace of order " + std::to_string(S.order) +
                  " has the error bound " + std::to_string(S.bound) + ", above krylov_error " +
                  std::to_string(krylov_error) + " (e^{t mu} of its growth rate mu is large)");
}

HostMatVec priv_state_matvec(const LinearSys &sys) {
    const int64_t n = sys.dim();
    // a dense A through the kernel of a sparse one: the same sets on every compiler (a*b + c fuses)
    if (const SparseMatrix *S = sys.A_sparse())
        return {n, std::make_shared<SparseMatrix>(*S), std::nullopt};
    auto dense = std::make_shared<SparseMatrix>(SparseMatrix::from_dense(sys.A()));
    return {n, dense, std::nullopt};
}

HostMatVec priv_integrator_matvec(const HostMatVec &A, const std::vector<double> &u) {
    return {A.n + 1, std::make_shared<SparseMatrix>(A.M->augment_column(u)), std::nullopt};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
