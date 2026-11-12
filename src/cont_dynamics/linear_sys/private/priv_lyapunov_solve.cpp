// priv_lyapunov_solve - the Lyapunov weight P of a stable matrix, A'P + PA = -I, with a factor
//
// Bartels-Stewart: with the real Schur form A = U T U' (T quasi upper triangular, blocks of 1 or 2)
// the equation reads T'Y + YT = -I for Y = U'PU, solved block column by block column (a block pair
// by a small Sylvester equation of at most 4 unknowns); P = U Y U' is symmetrized. Faster is
// Roberts' sign function iteration A_{k+1} = (cA_k + (cA_k)^{-1}) / 2, R_{k+1} = (cR_k +
// (cA_k)^{-T} cR_k (cA_k)^{-1}) / 2, R_0 = I, c the norm scaling: A_k'P + PA_k = -R_k, P = R_inf /
// 2 (matrix products Eigen runs on all threads, but less accurate for an ill conditioned A). The
// result is not trusted, priv_lyapunov_certificate checks it, so any method and arithmetic will do.
// A shift a just below the least eigenvalue of P comes from bisection on the success of the
// Cholesky factorization of P - aI (within 1 %).
//
// Syntax:   LyapunovWeight w = priv_lyapunov_solve(A, LyapunovMethod::schur);
// Inputs:   A - the sparse square matrix (stable: all eigenvalues in the open left half-plane)
//           method - the sign function iteration or the Schur form
// Outputs:  w - P and L (row-major, n x n; L'L = P - aI) and a; empty if there is none
// See also: priv_krylov_lyapunov
//
// CORA.cpp only: CORA has no Lyapunov weight; the solve is Bartels-Stewart (Comm. ACM 15, 1972)
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_krylov.h"
#include "cora/global/tensor/tolerance.h"
#include "cora/global/threads.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using Matrix = Eigen::MatrixXd;

// The Schur form ----------------------------------------------------------------------------------

/// The diagonal blocks of the quasi triangular T: the first index and the size (1 or 2) of each.
std::vector<std::pair<int64_t, int64_t>> aux_blocks(const Matrix &T) {
    std::vector<std::pair<int64_t, int64_t>> blocks;
    const int64_t n = T.rows();
    for (int64_t i = 0; i < n;) {
        const int64_t size = (i + 1 < n && T(i + 1, i) != 0.0) ? 2 : 1;
        blocks.emplace_back(i, size);
        i += size;
    }
    return blocks;
}

/// Y with T'Y + YT = -I, T quasi upper triangular.
Matrix aux_solve_schur(const Matrix &T) {
    const int64_t n = T.rows();
    const auto blocks = aux_blocks(T);
    Matrix Y = Matrix::Zero(n, n);
    for (const auto &[j0, bj] : blocks) {
        // the right-hand side of the column block: -I - Y(:, <j0) T(<j0, J)
        Matrix Z = -Matrix::Identity(n, n).middleCols(j0, bj);
        if (j0 > 0) Z.noalias() -= Y.leftCols(j0) * T.block(0, j0, j0, bj);
        const Matrix Tjj = T.block(j0, j0, bj, bj);
        for (const auto &[i0, bi] : blocks) {
            // minus T(<i0, I)' Y(<i0, J), then the small equation T_II' X + X T_JJ = Z_I
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 2, 2> Zi =
                Z.block(i0, 0, bi, bj);
            if (i0 > 0) Zi.noalias() -= T.block(0, i0, i0, bi).transpose() * Y.block(0, j0, i0, bj);
            const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 2, 2> Tii =
                T.block(i0, i0, bi, bi);
            if (bi == 1 && bj == 1) {
                Y(i0, j0) = Zi(0, 0) / (Tii(0, 0) + Tjj(0, 0));
                continue;
            }
            // vec(X) with (I_bj (x) T_II' + T_JJ' (x) I_bi) vec(X) = vec(Z_I), column-major
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 4, 4> K(bi * bj, bi * bj);
            K.setZero();
            for (int64_t c = 0; c < bj; ++c)
                for (int64_t r = 0; r < bi; ++r)
                    for (int64_t s = 0; s < bi; ++s) K(c * bi + r, c * bi + s) += Tii(s, r);
            for (int64_t c = 0; c < bj; ++c)
                for (int64_t d = 0; d < bj; ++d)
                    for (int64_t r = 0; r < bi; ++r) K(c * bi + r, d * bi + r) += Tjj(d, c);
            Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 4, 1> z(bi * bj);
            for (int64_t c = 0; c < bj; ++c)
                for (int64_t r = 0; r < bi; ++r) z(c * bi + r) = Zi(r, c);
            const Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 4, 1> x = K.partialPivLu().solve(z);
            for (int64_t c = 0; c < bj; ++c)
                for (int64_t r = 0; r < bi; ++r) Y(i0 + r, j0 + c) = x(c * bi + r);
        }
    }
    return Y;
}

/// Whether P - aI has a Cholesky factor; it replaces `work` by the factor (lower) on success.
bool aux_factor(const Matrix &P, double a, Matrix &work) {
    work = P;
    work.diagonal().array() -= a;
    Eigen::LLT<Eigen::Ref<Matrix>> llt(work);
    return llt.info() == Eigen::Success;
}

// The sign function -------------------------------------------------------------------------------

/// P with A'P + PA = -I by the sign function iteration (empty-sized if it does not converge).
Matrix aux_solve_sign(const Matrix &A) {
    const int64_t n = A.rows();
    Matrix Ak = A, R = Matrix::Identity(n, n);
    for (int step = 0; step < 100; ++step) {
        Matrix inverse = Ak.partialPivLu().inverse();
        // the scaling c = sqrt(|A^-1| / |A|) of the determinant-free kind; Abar = c A, Rbar = c R
        const double c = std::sqrt(inverse.norm() / Ak.norm());
        inverse /= c;
        R *= c;
        const Matrix next_R = 0.5 * (R + inverse.transpose() * R * inverse);
        const Matrix next_A = 0.5 * (c * Ak + inverse);
        const double change = (next_A - Ak).norm();
        Ak = next_A;
        R = next_R;
        if (!(change <= 100 * machine_eps() * Ak.norm())) continue;
        return 0.5 * R;
    }
    return Matrix();
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The weight --------------------------------------------------------------------------------------

LyapunovWeight priv_lyapunov_solve(const SparseMatrix &A, LyapunovMethod method) {
    configure_threads(); // Eigen's pool, as the cgroup's CPUs
    const int64_t n = A.rows();
    LyapunovWeight out;
    // A as a dense matrix
    Matrix D = Matrix::Zero(n, n);
    for (int64_t r = 0; r < n; ++r)
        for (int64_t e = A.row_start()[size_t(r)]; e < A.row_start()[size_t(r) + 1]; ++e)
            D(r, A.columns()[size_t(e)]) = A.values()[size_t(e)];
    Matrix P;
    if (method == LyapunovMethod::sign) {
        P = aux_solve_sign(D);
        if (P.rows() != n) return out;
    } else {
        // A = U T U', then the equation for T
        const Eigen::RealSchur<Matrix> schur(D);
        if (schur.info() != Eigen::Success) return out;
        D.resize(0, 0);
        const Matrix &U = schur.matrixU();
        P = U * aux_solve_schur(schur.matrixT()) * U.transpose();
    }
    // the symmetric part, the same bits above and below the diagonal
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = i; j < n; ++j) P(i, j) = P(j, i) = 0.5 * (P(i, j) + P(j, i));
    // the shift: the largest a within 1 % of the failure point with a Cholesky factor of P - aI
    double lo = 0.0, hi = P.diagonal().minCoeff();
    if (!(hi > 0.0)) return out;
    Matrix work, best;
    for (double mid = 0.5 * hi; hi - lo > 0.01 * hi; mid = 0.5 * (lo + hi)) {
        if (aux_factor(P, mid, work)) {
            lo = mid;
            best.swap(work);
        } else {
            hi = mid;
        }
    }
    if (lo == 0.0) return out;
    // L (upper) with L'L = P - aI: the transpose of the lower factor
    out.a = lo;
    out.P.resize(size_t(n * n));
    out.L.assign(size_t(n * n), 0.0);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < n; ++j) {
            out.P[size_t(i * n + j)] = P(i, j);
            if (j >= i) out.L[size_t(i * n + j)] = best(j, i);
        }
    return out;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
