// priv_linear_algebra - the simultaneous diagonalization and the congruence of shape matrices
//
// Syntax:   auto [T, D] = priv_simdiag(M1, M2, tol);   Tensor Q = priv_congruence(M, Q);
// Inputs:   M1, M2, Q - symmetric matrices (n, n) (the congruence: (..., n, n));  M - (..., m, n)
// Outputs:  T with T M1 T' = I and D = T M2 T' diagonal;  the shape matrix M Q M' symmetrized
// See also: priv.h, Ellipsoid::is_bigger, Ellipsoid::is_bad_dir
//
// CORA: global/functions/helper/sets/contSet/ellipsoid/simdiag.m

#include "cont_set/ellipsoid/private/priv.h"
#include "cora/global/within_tol.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The smallest eigenvalue of the symmetric matrix M.
double aux_smallest_eigenvalue(const Tensor &M) { return M.eigh().first.min().item(); }

} // namespace


// ===========================================  MAIN  =========================================== //

std::pair<Tensor, Tensor> priv_simdiag(const Tensor &M1, const Tensor &M2, double tol) {
    const bool symmetric = within_tol(M1, M1.matrix_transpose(), tol).min().item() != 0.0 &&
                           within_tol(M2, M2.matrix_transpose(), tol).min().item() != 0.0;
    if (!symmetric || aux_smallest_eigenvalue(M1) < tol || aux_smallest_eigenvalue(M2) < -tol)
        CORA_FAIL(Error::special_error,
                  "simdiag: both matrices need to be symmetric, the first positive definite and "
                  "the second positive semidefinite");
    const std::array<Tensor, 3> usv = M1.svd();
    const Tensor S_inv_root = (1.0 / usv[1].sqrt()).diag();
    const Tensor U1_t = usv[0].matrix_transpose();
    const Tensor U2 = S_inv_root.matmul(U1_t).matmul(M2).matmul(usv[0]).matmul(S_inv_root).svd()[0];
    const Tensor T = U2.matrix_transpose().matmul(S_inv_root).matmul(U1_t);
    return {T, T.matmul(M2).matmul(T.matrix_transpose())};
}

Tensor priv_congruence(const Tensor &M, const Tensor &Q) {
    const Tensor image = M.matmul(Q).matmul(M.matrix_transpose());
    return (image + image.matrix_transpose()) * 0.5;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
