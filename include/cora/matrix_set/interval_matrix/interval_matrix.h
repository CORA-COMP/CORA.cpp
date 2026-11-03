// interval_matrix - the class of interval matrices {M | inf <= M <= sup}, as CORA's intervalMatrix
//
// Holds one Interval with matrix bounds (..., rows, cols), CORA's `int` field, and no second
// representation: the methods of CORA's matrixSet that mean something else than the same name of
// an interval (abs, contains, mtimes, norm) are methods of this class, the rest is the Interval's.
// Immutable: every operation returns a new set. Leading dimensions are batches.
//
// Syntax:     IntervalMatrix A(C, D);   IntervalMatrix A(C);   IntervalMatrix A(I);
// Operators:  A * B, A * M, M * A, s * A, A * s, A * Z (matmul), as CORA's mtimes
// Operations: abs, center, contains, dependent_terms, exact_square, expm, expm_independent,
//             expm_mixed, exponential_remainder, infimum, infinity_norm, matrix_power, matmul,
//             norm (one file each); the constructor checks its input (interval_matrix.cpp)
// See also:   cora/cont_set/interval/interval.h, cora/cont_set/zonotope/zonotope.h
//
// CORA: matrixSet/@intervalMatrix/intervalMatrix.m

#pragma once

#include "cora/cont_set/empty_set/empty_set.h"
#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/zonotope/zonotope.h"

#include <string>
#include <utility>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

struct ExpmResult;
struct ExpmMixedResult;

class IntervalMatrix {
  public:
    /// The matrices C +- |D|, entrywise [C - |D|, C + |D|]; C and D are (..., rows, cols).
    IntervalMatrix(Tensor center, Tensor delta);

    /// The point matrix C.
    explicit IntervalMatrix(Tensor center);

    /// The interval matrix of the bounds of I (CORA writes `intMat.int = I`).
    explicit IntervalMatrix(Interval bounds);

    // Bounds and matrices of the set -------------------------------------------------------------

    /// The bounds, an Interval of matrix bounds.
    const Interval &interval() const noexcept { return int_; }

    /// The number of rows and of columns.
    int64_t rows() const { return int_.inf().size(-2); }
    int64_t cols() const { return int_.inf().size(-1); }

    /// The lower bound matrix.
    Tensor infimum() const;

    /// The center matrix (inf + sup) / 2.
    Tensor center() const;

    /// The largest absolute value of each entry over the matrices: max(|inf|, |sup|).
    Tensor abs() const;

    /// Whether the matrix M (..., rows, cols) is in the set, entry by entry up to tol (as
    /// within_tol). type "exact" and "approx" give the same answer for a matrix of numbers, as in
    /// CORA. CORA's default tol is 1e-12, here the repository tolerance (docs/architecture.md).
    bool contains(const Tensor &M, const std::string &type = "exact", Tol tol = std::nullopt) const;

    // Squares and exponentials -------------------------------------------------------------------

    /// The interval matrix of the entrywise range of M * M, M in the set: the exact square.
    IntervalMatrix exact_square() const;

    /// The dependent terms of the Taylor series of exp(M r): {r^2 [M]^2 (exact), I + [M] r +
    /// [M]^2 r^2 / 2 (the loop's H)}, each rebuilt from center and radius. A is square.
    std::pair<IntervalMatrix, IntervalMatrix> dependent_terms(double r) const;

    /// The remainder box [-W, W] of the Taylor series to order max_order: W = exp(|M|) - sum
    /// |M|^i / i!; Inf where exp(|M|) is not a number. A is square.
    IntervalMatrix exponential_remainder(int64_t max_order) const;

    /// The enclosure of exp(M) over the set: the series to order max_order (default 10) with the
    /// dependent terms of the second order and the remainder, at the step r = 1.
    ExpmResult expm(int64_t max_order = 10) const;

    /// The same with the step r != 0: the dependent terms of [A] / r at the step r.
    ExpmResult expm(double r, int64_t max_order) const;

    /// The series continued from initial_power = [A]^initial_order: its terms from that order on,
    /// with the remainder (the lower ones are the caller's); r is not used, as in CORA.
    ExpmResult expm(double r, int64_t max_order, int64_t initial_order,
                    const IntervalMatrix &initial_power) const;

    /// The enclosure of exp(M) with every power evaluated on its own, without the dependent
    /// terms (CORA's expmInd), from order 0 with [A]^0 = I.
    ExpmResult expm_independent(int64_t max_order) const;

    /// The same continued from initial_power = [A]^initial_order, as expm does.
    ExpmResult expm_independent(int64_t max_order, int64_t initial_order,
                                const IntervalMatrix &initial_power) const;

    /// The series in two parts: the sum to intermediate_order >= 2 with the dependent terms, and
    /// the rest of it up to max_order with the remainder.
    ExpmMixedResult expm_mixed(double r, int64_t intermediate_order, int64_t max_order) const;

    // Norms and powers ----------------------------------------------------------------------------

    /// The largest absolute row sum of the matrices of the set: norm(abs(A), Inf).
    Tensor infinity_norm() const;

    /// The norm of the matrix |center| + rad: type one, inf or fro. The 2-norm is refused
    /// (Error::wrong_value) as in CORA: it is NP-hard for interval matrices.
    Tensor norm(Tensor::Norm type) const;

    /// The power [A]^k of a square set: the identity for k = 0, else A (A (... A)) with k factors,
    /// each an interval matrix product (CORALean's matPow; CORA multiplies from the left). A 1 x 1
    /// matrix is a product too, not an elementwise power.
    IntervalMatrix matrix_power(int64_t exponent) const;

    // Products (CORA's mtimes) --------------------------------------------------------------------

    /// [A] * [B] by the sum of the hulls of the corner products; [A] * M and M * [A] for a matrix
    /// of numbers; [A] * s for a number s (negative swaps the bounds).
    IntervalMatrix matmul(const IntervalMatrix &B) const;
    IntervalMatrix matmul(const Tensor &M) const;
    IntervalMatrix matmul(double s) const;
    static IntervalMatrix matmul(const Tensor &M, const IntervalMatrix &A);

    /// [A] * Z: the zonotope that encloses the product (CORA's aux_mtimes_zonotope), and [A] * the
    /// empty set: the empty set of rows dimensions.
    Zonotope matmul(const Zonotope &Z) const;
    EmptySet matmul(const EmptySet &E) const;

  private:
    Interval int_;
};

/// The result of expm: the enclosure of the series, the powers of A that were multiplied
/// (powers[k] is of order first_order + k) and the remainder box.
struct ExpmResult {
    IntervalMatrix enclosure;
    std::vector<IntervalMatrix> powers;
    int64_t first_order;
    IntervalMatrix remainder;
};

/// The result of expm_mixed: the sum of the terms to the intermediate order, the rest of the
/// series, the powers of both parts (orders 1.. and intermediate_order + 1..) and the remainder.
struct ExpmMixedResult {
    IntervalMatrix enclosure, enclosure_rest;
    std::vector<IntervalMatrix> powers, powers2;
    IntervalMatrix remainder;
};

inline IntervalMatrix operator*(const IntervalMatrix &A, const IntervalMatrix &B) {
    return A.matmul(B);
}
inline IntervalMatrix operator*(const IntervalMatrix &A, const Tensor &M) { return A.matmul(M); }
inline IntervalMatrix operator*(const Tensor &M, const IntervalMatrix &A) {
    return IntervalMatrix::matmul(M, A);
}
inline IntervalMatrix operator*(double s, const IntervalMatrix &A) { return A.matmul(s); }
inline IntervalMatrix operator*(const IntervalMatrix &A, double s) { return A.matmul(s); }
inline Zonotope operator*(const IntervalMatrix &A, const Zonotope &Z) { return A.matmul(Z); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
