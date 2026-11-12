// priv_lyapunov_solve - the Lyapunov weight P of a stable matrix, A'P + PA = -I, with a factor
//
// Roberts' sign function iteration A_{k+1} = (cA_k + (cA_k)^{-1}) / 2, R_{k+1} = (cR_k +
// (cA_k)^{-T} cR_k (cA_k)^{-1}) / 2, R_0 = I, c the norm scaling: A_k'P + PA_k = -R_k, P = R_inf /
// 2. It runs in single precision (twice the speed) and stops once the residual |PA + (PA)' + I|
// has a Gershgorin bound below 1/2, not at the last bit; then in double, last Bartels-Stewart:
// with A = U T U' (T quasi upper triangular, blocks of 1 or 2) the equation reads T'Y + YT = -I
// for Y = U'PU, solved block column by block column (a block pair by a small Sylvester equation
// of at most 4 unknowns); P = U Y U' is symmetrized. The result is not trusted, the certificate
// (priv_lyapunov_certificate) checks it, so any method and arithmetic will do. The inverse of the
// iteration is the LU factorization with both triangular factors inverted by blocks, so that all
// of the work is in matrix products. A shift a below the least eigenvalue of P comes from an
// inverse iteration with the Cholesky factor of P, then the Cholesky factorization of P - aI.
//
// Syntax:   LyapunovWeight w = priv_lyapunov_solve(A, LyapunovMethod::schur);
// Inputs:   A - the sparse square matrix (stable: all eigenvalues in the open left half-plane)
//           method - the sign function iteration (single or double precision) or the Schur form
// Outputs:  w - P and L (row-major, n x n; L'L = P - aI) and a; empty if there is none
// See also: priv_krylov_lyapunov
//
// CORA.cpp only: CORA has no Lyapunov weight; the solve is Bartels-Stewart (Comm. ACM 15, 1972)
//                and Roberts (Int. J. Control 32, 1980)
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   12-November-2026
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_krylov.h"
#include "cora/global/tensor/tolerance.h"
#include "cora/global/threads.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

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

template <class S>
using Dense = Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>;

/// Below this order the triangular inverse is Eigen's own solve.
constexpr int64_t kBaseOrder = 96;

/// The inverse of the lower triangular L in place, by blocks: with L = [L11 0; L21 L22] the inverse
/// is [L11^-1 0; -L22^-1 L21 L11^-1, L22^-1], and the products are dense (Eigen's triangular
/// products run on one thread and no faster).
template <class S>
void aux_invert_lower(Eigen::Ref<Dense<S>> L) {
    const int64_t n = L.rows();
    if (n <= kBaseOrder) {
        const Dense<S> inverse =
            L.template triangularView<Eigen::Lower>().solve(Dense<S>::Identity(n, n));
        L = inverse;
        return;
    }
    const int64_t h = n / 2;
    Eigen::Ref<Dense<S>> L11 = L.topLeftCorner(h, h), L22 = L.bottomRightCorner(n - h, n - h);
    Eigen::Ref<Dense<S>> L21 = L.bottomLeftCorner(n - h, h);
    aux_invert_lower<S>(L11);
    aux_invert_lower<S>(L22);
    Dense<S> product(n - h, h);
    product.noalias() = L21 * L11;
    L21.noalias() = -(L22 * product);
}

/// W = U L for the upper triangular U and the lower triangular L, by blocks: with the halves W11 =
/// U11 L11 + U12 L21, W12 = U12 L22, W21 = U22 L21, W22 = U22 L22, where the first and the last
/// product are of the same kind (half of the work of the dense product).
template <class S>
void aux_multiply_upper_lower(const Eigen::Ref<const Dense<S>> &U,
                              const Eigen::Ref<const Dense<S>> &L, Eigen::Ref<Dense<S>> W) {
    const int64_t n = U.rows();
    if (n <= kBaseOrder) {
        W.noalias() = U * L;
        return;
    }
    const int64_t h = n / 2, m = n - h;
    aux_multiply_upper_lower<S>(U.topLeftCorner(h, h), L.topLeftCorner(h, h),
                                W.topLeftCorner(h, h));
    W.topLeftCorner(h, h).noalias() += U.topRightCorner(h, m) * L.bottomLeftCorner(m, h);
    W.topRightCorner(h, m).noalias() = U.topRightCorner(h, m) * L.bottomRightCorner(m, m);
    W.bottomLeftCorner(m, h).noalias() = U.bottomRightCorner(m, m) * L.bottomLeftCorner(m, h);
    aux_multiply_upper_lower<S>(U.bottomRightCorner(m, m), L.bottomRightCorner(m, m),
                                W.bottomRightCorner(m, m));
}

/// Below this order the symmetric product is formed whole.
constexpr int64_t kSymmetricOrder = 512;

/// Z = G'Y for the symmetric result (Y = RG with a symmetric R) on the rows and columns first to
/// first + size - 1, from the blocks on and below the diagonal, the upper ones mirrored.
template <class S>
void aux_symmetric_product(const Dense<S> &G, const Dense<S> &Y, Dense<S> &Z, int64_t first,
                           int64_t size) {
    if (size <= kSymmetricOrder) {
        Z.block(first, first, size, size).noalias() =
            G.middleCols(first, size).transpose() * Y.middleCols(first, size);
        return;
    }
    const int64_t h = size / 2, m = size - h;
    aux_symmetric_product<S>(G, Y, Z, first, h);
    aux_symmetric_product<S>(G, Y, Z, first + h, m);
    Z.block(first + h, first, m, h).noalias() =
        G.middleCols(first + h, m).transpose() * Y.middleCols(first, h);
    Z.block(first, first + h, h, m) = Z.block(first + h, first, m, h).transpose();
}

/// The inverse of A by its LU factorization: A = P'LU gives A^-1 = U^-1 L^-1 P.
template <class S>
Dense<S> aux_inverse(const Dense<S> &A) {
    const int64_t n = A.rows();
    const Eigen::PartialPivLU<Dense<S>> lu(A);
    Dense<S> Li = lu.matrixLU().template triangularView<Eigen::UnitLower>();
    // U' is lower triangular, so the inverse of U is the transpose of the inverse of U'
    Dense<S> Ut = lu.matrixLU().transpose().template triangularView<Eigen::Lower>();
    aux_invert_lower<S>(Li);
    aux_invert_lower<S>(Ut);
    const Dense<S> Ui = Ut.transpose();
    Dense<S> product(n, n);
    aux_multiply_upper_lower<S>(Ui, Li, product);
    return product * lu.permutationP();
}

/// The Gershgorin bound 1 - max_i sum_j |W_ij| of W = PA + (PA)' + I for P = R / 2, A sparse: the
/// gap the certificate sees (the weight passes with a gap above 0, and a rate proportional to it).
template <class S>
double aux_gap(const Dense<S> &R, const SparseMatrix &A) {
    const int64_t n = R.rows();
    const std::vector<int64_t> &start = A.row_start(), &cols = A.columns();
    const std::vector<double> &vals = A.values();
    // (RA)_ij = sum over k of R_ik A_kj, row i from the column i of the symmetric R
    std::vector<S> RA(size_t(n * n), S(0));
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        S *row = &RA[size_t(i * n)];
        for (int64_t k = 0; k < n; ++k) {
            const S factor = R(k, i);
            for (int64_t e = start[size_t(k)]; e < start[size_t(k) + 1]; ++e)
                row[cols[size_t(e)]] += factor * S(vals[size_t(e)]);
        }
    }
    // the row sums of |W|, blockwise for the cache
    constexpr int64_t tile = 64;
    std::vector<double> sums(size_t(n), 0.0);
#pragma omp parallel for schedule(static)
    for (int64_t ib = 0; ib < n; ib += tile)
        for (int64_t jb = 0; jb < n; jb += tile)
            for (int64_t i = ib; i < std::min(ib + tile, n); ++i)
                for (int64_t j = jb; j < std::min(jb + tile, n); ++j) {
                    const double w = 0.5 * (double(RA[size_t(i * n + j)]) + RA[size_t(j * n + i)]) +
                                     (i == j ? 1.0 : 0.0);
                    sums[size_t(i)] += std::abs(w);
                }
    return 1.0 - *std::max_element(sums.begin(), sums.end());
}

/// The weight gap above which the iteration stops: the certificate wants a gap above 0, and the
/// rate of the decay it proves is proportional to the gap.
constexpr double kStopGap = 0.5;

/// P with A'P + PA = -I by the sign function iteration in the arithmetic S, as soon as its gap
/// (aux_gap) is at least kStopGap or the iteration converged; empty-sized if it does neither.
template <class S>
Matrix aux_solve_sign(const SparseMatrix &A, const Matrix &D) {
    const int64_t n = D.rows();
    Dense<S> Ak = D.cast<S>(), R = Dense<S>::Identity(n, n), Y(n, n), Z(n, n);
    // converged: a relative step below the cube root of the double epsilon, single reaches it too
    const double converged = std::cbrt(machine_eps());
    for (int step = 0; step < 100; ++step) {
        Dense<S> inverse = aux_inverse<S>(Ak);
        // the scaling c = sqrt(|A^-1| / |A|) of the determinant-free kind; Abar = c A, Rbar = c R
        const S c = std::sqrt(inverse.norm() / Ak.norm());
        inverse /= c;
        R *= c;
        Y.noalias() = R * inverse;
        aux_symmetric_product<S>(inverse, Y, Z, 0, n);
        R = S(0.5) * (R + Z);
        inverse = S(0.5) * (c * Ak + inverse);
        const double change = (inverse - Ak).norm() / inverse.norm();
        Ak = std::move(inverse);
        if (!std::isfinite(change)) return Matrix();
        if (aux_gap<S>(R, A) >= kStopGap || change <= converged)
            return 0.5 * R.template cast<double>();
    }
    return Matrix();
}

// The shift ---------------------------------------------------------------------------------------

/// An upper bound of the least eigenvalue of the symmetric positive definite P by inverse
/// iteration with its Cholesky factor (the Rayleigh quotients decrease to it), 0 if P has none.
double aux_least_eigenvalue(const Matrix &P) {
    const int64_t n = P.rows();
    Matrix factor = P;
    const Eigen::LLT<Eigen::Ref<Matrix>> llt(factor);
    if (llt.info() != Eigen::Success) return 0.0;
    // a start vector that is not an eigenvector of a structured matrix
    Eigen::VectorXd x(n);
    for (int64_t i = 0; i < n; ++i) x(i) = 1.0 + 0.5 * std::sin(double(i * i + 1));
    double rayleigh = (x.dot(P * x)) / x.squaredNorm();
    for (int step = 0; step < 30; ++step) {
        x = llt.solve(x);
        x /= x.norm();
        rayleigh = std::min(rayleigh, x.dot(P * x));
    }
    return rayleigh;
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
    if (method == LyapunovMethod::sign_single || method == LyapunovMethod::sign) {
        P = method == LyapunovMethod::sign ? aux_solve_sign<double>(A, D)
                                           : aux_solve_sign<float>(A, D);
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
    // the shift: a fraction of the least eigenvalue, lowered until P - aI has a Cholesky factor
    double a = 0.75 * aux_least_eigenvalue(P);
    if (!(a > 0.0)) return out;
    Matrix best;
    bool factored = false;
    for (int attempt = 0; attempt < 40 && !factored; ++attempt, a *= 0.75)
        factored = aux_factor(P, a, best);
    if (!factored) return out;
    a /= 0.75; // the shift that was factored
    // L (upper) with L'L = P - aI: the transpose of the lower factor
    out.a = a;
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
