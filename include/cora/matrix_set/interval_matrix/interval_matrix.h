// interval_matrix - the class of interval matrices {M | inf <= M <= sup}, as CORA's intervalMatrix
//
// Holds one Interval with matrix bounds (..., rows, cols), CORA's `int` field, and no second
// representation: the methods of CORA's matrixSet that mean something else than the same name of
// an interval (abs, contains, mtimes, norm) are methods of this class, the rest is the Interval's.
// Immutable: every operation returns a new set. Leading dimensions are batches.
//
// Syntax:     IntervalMatrix A(C, D);   IntervalMatrix A(C);   IntervalMatrix A(I);
// Operators:  A * B, A * M, M * A, s * A, A * s, A * Z (matmul), as CORA's mtimes;
//             A + B, A + M, M + A, A + s, s + A (plus)
// Operations: abs, center, contains, dependent_terms, exact_square, expm, expm_independent,
//             expm_mixed, exponential_remainder, infimum, infinity_norm, matrix_power, matmul,
//             norm, plus, powers, rad, representsa, supremum, vertices, volume, dim, display,
//             generate_random, expm_absolute_bound, expm_independent_mixed, expm_norm_infinity,
//             expm_taylor_horner_scaling, expm_different_algorithms (one file each);
//             the constructor checks its input (interval_matrix.cpp)
// See also:   cora/cont_set/interval/interval.h, cora/cont_set/zonotope/zonotope.h
//
// CORA: matrixSet/@intervalMatrix/intervalMatrix.m
// CORA: matrixSet/@intervalMatrix/interval.m
// CORA authors: Matthias Althoff, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/empty_set/empty_set.h"
#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/zonotope/zonotope.h"

#include <array>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

struct ExpmResult;
struct ExpmMixedResult;

class Rng;

class IntervalMatrix {
  public:
    /// The matrices C +- |D|, entrywise [C - |D|, C + |D|]; C and D are (..., rows, cols).
    IntervalMatrix(const Tensor &center, const Tensor &delta);

    /// The point matrix C.
    explicit IntervalMatrix(const Tensor &center);

    /// The interval matrix of the bounds of I (CORA writes `intMat.int = I`).
    explicit IntervalMatrix(Interval bounds);

    // Bounds and matrices of the set -------------------------------------------------------------

    /// The bounds, an Interval of matrix bounds (CORA's interval(A)).
    const Interval &interval() const noexcept { return int_; }

    /// The number of rows and of columns.
    int64_t rows() const { return int_.inf().size(-2); }
    int64_t cols() const { return int_.inf().size(-1); }

    /// {rows, cols}, CORA's dim(A); dim(rc) is one of them, rc = 1 for the rows and 2 for the
    /// columns, as CORA's dim(A, rc).
    std::array<int64_t, 2> dim() const;
    int64_t dim(int rc) const;

    /// The set as text, as CORA displays it: what `std::cout << A` prints.
    std::string display() const;

    /// A random interval matrix, as CORA's generateRandom: dimension {rows, cols} (default the
    /// shape of center or max_radius, else each random in 1..10), center (default 10 N(0, 1) per
    /// entry), max_radius (radii >= 0: a matrix, or a 1 x 1 tensor for all; default 10 U[0, 1]).
    using Dimension = std::optional<std::array<int64_t, 2>>;
    using Matrix = std::optional<Tensor>;
    static IntervalMatrix generate_random(Rng &rng, Dimension dimension = std::nullopt,
                                          const Matrix &center = std::nullopt,
                                          const Matrix &max_radius = std::nullopt);

    /// The lower bound matrix.
    Tensor infimum() const;

    /// The upper bound matrix.
    Tensor supremum() const;

    /// The center matrix (inf + sup) / 2.
    Tensor center() const;

    /// The radius matrix (sup - inf) / 2.
    Tensor rad() const;

    /// The largest absolute value of each entry over the matrices: max(|inf|, |sup|).
    Tensor abs() const;

    /// Whether the matrix M (..., rows, cols) is in the set, entry by entry up to tol (as
    /// within_tol). type "exact" and "approx" give the same answer for a matrix of numbers, as in
    /// CORA. CORA's default tol is 1e-12, here the repository tolerance (docs/architecture.md).
    bool contains(const Tensor &M, const std::string &type = "exact", Tol tol = std::nullopt) const;

    /// Whether the set is also a set of another type (an Interval::representsa type), decided on
    /// the entries stacked into one vector, as CORA does; tol as there.
    bool representsa(const std::string &type, Tol tol = std::nullopt) const;

    /// The product of the entry widths, (..., 1, 1): the volume of the box of the entries.
    Tensor volume() const;

    /// The distinct matrices with every entry at an end of its interval, one (K, rows, cols)
    /// tensor per batch member (the batch flattened); at most 20 entries.
    std::vector<Tensor> vertices() const;

    // Sums ----------------------------------------------------------------------------------------

    /// The Minkowski sum with an interval matrix, a matrix of numbers or a number: the bounds add.
    IntervalMatrix plus(const IntervalMatrix &B) const;
    IntervalMatrix plus(const Tensor &M) const;
    IntervalMatrix plus(double s) const;

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

    /// The exponential by CORA's other algorithms, for the order max_order: style 0 the Taylor
    /// series, 1 Horner's scheme, 2 Horner on [A] / 2^p squared p times. Nothing where the norm
    /// is too large.
    std::optional<IntervalMatrix> expm_taylor_horner_scaling(int64_t max_order, int64_t style,
                                                             int64_t scaling_exponent = 1) const;

    /// The same, and style 4: the enclosure of expm at the step 1.
    std::optional<IntervalMatrix> expm_different_algorithms(int64_t max_order, int64_t style,
                                                            int64_t scaling_exponent = 1) const;

    /// The series in two parts with independent powers: the sum to intermediate_order >= 1, and
    /// the rest of it up to max_order with the remainder (CORA's expmIndMixed).
    ExpmMixedResult expm_independent_mixed(int64_t intermediate_order, int64_t max_order) const;

    /// CORA's closed-form bound of the infinity norm of the exponential's remainder at time t
    /// (expmNormInf), and its matrix bound (expmAbsoluteBound) for square A.
    Tensor expm_norm_infinity(double t) const;
    Tensor expm_absolute_bound(double t) const;

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

    /// The powers [A]^i from order initial_order (default 1, A) to max_order, each the previous
    /// times A from the right as CORA's powers; powers[k] is of order initial_order + k.
    std::vector<IntervalMatrix> powers(int64_t max_order) const;
    std::vector<IntervalMatrix> powers(int64_t max_order, int64_t initial_order,
                                       const IntervalMatrix &initial_power) const;

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
    /// The checks of the two expm algorithm entries (style 4 only if allow_expm).
    void check_series_input(int64_t max_order, int64_t style, int64_t scaling_exponent,
                            bool allow_expm) const;

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

// the text of a set, and CORA's plus and mtimes as operators
inline std::ostream &operator<<(std::ostream &out, const IntervalMatrix &A) {
    return out << A.display();
}
inline IntervalMatrix operator+(const IntervalMatrix &A, const IntervalMatrix &B) {
    return A.plus(B);
}
inline IntervalMatrix operator+(const IntervalMatrix &A, const Tensor &M) { return A.plus(M); }
inline IntervalMatrix operator+(const Tensor &M, const IntervalMatrix &A) { return A.plus(M); }
inline IntervalMatrix operator+(const IntervalMatrix &A, double s) { return A.plus(s); }
inline IntervalMatrix operator+(double s, const IntervalMatrix &A) { return A.plus(s); }
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
