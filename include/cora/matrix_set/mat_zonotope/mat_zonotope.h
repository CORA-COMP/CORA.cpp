// mat_zonotope - the class of matrix zonotopes, as CORA's matZonotope
//
// The matrices C + sum_i b_i G_i with |b|_inf <= 1: the center C is (..., rows, cols) and the
// generators G are (..., h, rows, cols), one matrix per generator (CORA stores them as (rows, cols,
// h)). CORA's matZonotope is a matrixSet, not a contSet: the methods that mean something else than
// a zonotope's (mtimes, norm, contains) are methods of this class. Immutable; leading dimensions
// are batches. The deprecated cell of generators of CORA's constructor is not offered.
//
// Syntax:     MatZonotope A(C, G);   MatZonotope A(C);   MatZonotope A;
// Operators:  A * B, A * Z, A * M, M * A, s * A, A * s (matmul, as CORA's mtimes), A + B, A + M
//             (plus)
// Operations: center, contains, dependent_terms, expm, expm_independent, expm_mixed,
//             expm_one_param, interval_matrix, matmul, matrix_power, norm, plus, powers,
//             representsa, transpose, zonotope (one file each); the constructors check their
//             input (mat_zonotope.cpp)
// Gradients:  every operation is differentiable except interval_matrix, norm, expm_independent,
//             expm_mixed and expm_one_param (piecewise: absolute values) and contains and
//             representsa (not differentiable: they answer a bool)
// See also:   cora/matrix_set/interval_matrix/interval_matrix.h, cora/cont_set/zonotope/zonotope.h
//
// CORA: matrixSet/@matZonotope/matZonotope.m

#pragma once

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/matrix_set/interval_matrix/interval_matrix.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

class MatZonotope {
  public:
    /// The empty 0 x 0 matrix zonotope.
    MatZonotope();

    /// The point matrix C (..., rows, cols), without generators.
    explicit MatZonotope(Tensor center);

    /// The matrices C + sum_i b_i G_i: C is (..., rows, cols), G is (..., h, rows, cols).
    MatZonotope(Tensor center, Tensor generators);

    // The parts of the set ------------------------------------------------------------------------

    /// The center matrix C.
    Tensor center() const;

    /// The generators (..., h, rows, cols).
    const Tensor &generators() const noexcept { return G_; }

    /// The number of rows and of columns, and of generators.
    int64_t rows() const { return C_.size(-2); }
    int64_t cols() const { return C_.size(-1); }
    int64_t num_generators() const { return G_.size(-3); }

    /// The set of one vector zonotope: the stacked center and generators, column by column
    /// (reshape(., [], 1) of CORA), as a zonotope of rows * cols dimensions.
    Zonotope zonotope() const;

    /// The box around the set as an interval matrix: C +- sum_i |G_i|.
    IntervalMatrix interval_matrix() const;

    /// Whether the matrices M are in the set: M is (rows, cols) or (p, rows, cols) for p matrices,
    /// all of which must be inside. type "exact" or "approx", as the zonotope's contains of the
    /// stacked points, relaxed by tol (CORA's default is 1e-12, here the repository tolerance,
    /// docs/architecture.md). Not a batch.
    bool contains(const Tensor &M, const std::string &type = "exact", Tol tol = std::nullopt) const;

    /// Whether the set is also a set of the CORA type ("origin", "point", ...): the zonotope's
    /// representsa of the stacked set, so the types and tol are its. Not a batch. tol default: the
    /// repository tolerance.
    bool representsa(const std::string &type, Tol tol = std::nullopt) const;

    // Sums and products (CORA's plus and mtimes) --------------------------------------------------

    /// The Minkowski sum with a matrix zonotope (the centers add, the generators are joined), or
    /// with a matrix of numbers (the center shifts).
    MatZonotope plus(const MatZonotope &B) const;
    MatZonotope plus(const Tensor &M) const;

    /// The product A B of two matrix zonotopes, an enclosure: the center C_A C_B and the products
    /// G_A(i) C_B, C_A G_B(j) and G_A(i) G_B(j) (j varying fastest) of the generators, in the order
    /// of CORALean; CORA lists the same products with A's index fastest.
    MatZonotope matmul(const MatZonotope &B) const;

    /// A M and M A for a matrix of numbers, and s A for a number: exact.
    MatZonotope matmul(const Tensor &M) const;
    MatZonotope matmul(double s) const;
    static MatZonotope matmul(const Tensor &M, const MatZonotope &A);

    /// A Z for a zonotope Z (columns): the zonotope that encloses the products, with the center
    /// C_A c, the generators C_A G, G_i c and G_i g_j. A 1 x 1 set as a scalar factor is refused
    /// (Error::not_supported).
    Zonotope matmul(const Zonotope &Z) const;

    /// The transpose of every matrix of the set: exact.
    MatZonotope transpose() const;

    /// The power A^k of a square set: the identity for k = 0, A for k = 1, else ((A A) A) ... with
    /// k factors, every product matmul.
    MatZonotope matrix_power(int64_t exponent) const;

    /// The powers A^i for i = initial_order (the first entry) to max_order, each the previous one
    /// times A from the right; the first is A (initial_order 1) or initial_power. A is square.
    std::vector<MatZonotope> powers(int64_t max_order) const;
    std::vector<MatZonotope> powers(int64_t max_order, int64_t initial_order,
                                    const MatZonotope &initial_power) const;

    /// An upper bound of the norm of every matrix of the set: the norm of its interval_matrix
    /// (1, inf or fro; the 2-norm is refused, as in CORA), (..., 1, 1).
    Tensor norm(Tensor::Norm type) const;

    // The matrix exponential ----------------------------------------------------------------------

    /// The terms of the Taylor series of exp(M r) that depend on one another (A square):
    /// (r^2 M^2 as a matrix zonotope, I + M r + M^2 r^2 / 2), each an enclosure over the set.
    std::pair<MatZonotope, MatZonotope> dependent_terms(double r) const;

    /// An enclosure of exp(M) over the set: expm_independent (CORA's expm calls expmInd).
    MatZonotope expm(int64_t max_order = 10) const;

    /// The series to max_order with every power evaluated on its own, plus the remainder box of
    /// the interval matrix of the set.
    MatZonotope expm_independent(int64_t max_order) const;

    struct ExpmMixedResult;
    struct ExpmOneParamResult;

    /// The series in two parts: the terms to intermediate_order >= 2 as a matrix zonotope (with the
    /// dependent terms of the step r > 0) and the rest up to max_order as interval matrices.
    ExpmMixedResult expm_mixed(double r, int64_t intermediate_order, int64_t max_order) const;

    /// The Taylor series of exp(M r) to max_order for the matrices C + b G_1 of a set with exactly
    /// one generator, each power's terms in b kept apart (CORALean's K(r), Althoff and Dolan, ACC
    /// 2012); the remainder box is returned beside it, not added. CORA sums the leading term of
    /// the l-th generator from the first coefficient of M^l, which is not the coefficient of b^l,
    /// so its set misses matrices (docs/cora-differences.md): the sum starts at the right one.
    /// The same with the constant input u(b) = c + b g, a zonotope with at most one generator in
    /// the same parameter (a column u is the zonotope (u, 0)): constant_input encloses the sum of
    /// r^k / k! M^(k-1) u over the set.
    ExpmOneParamResult expm_one_param(double r, int64_t max_order,
                                      const std::optional<Zonotope> &u_trans = std::nullopt) const;

  private:
    Tensor C_, G_;
};

/// The result of expm_mixed: the terms to the intermediate order, the rest of the series, the
/// powers of A of orders 1 to the intermediate order, the powers of the second part and the
/// remainder box.
struct MatZonotope::ExpmMixedResult {
    MatZonotope enclosure;
    IntervalMatrix enclosure_rest;
    std::vector<MatZonotope> powers;
    std::vector<IntervalMatrix> powers2;
    IntervalMatrix remainder;
};

/// The result of expm_one_param: the matrix zonotope of the Taylor terms, the remainder box, the
/// powers (powers[k] of order k + 1, a generator per power of the parameter) and, with an input,
/// the zonotope of the input terms.
struct MatZonotope::ExpmOneParamResult {
    MatZonotope enclosure;
    IntervalMatrix remainder;
    std::vector<MatZonotope> powers;
    std::optional<Zonotope> constant_input;
};

inline MatZonotope operator*(const MatZonotope &A, const MatZonotope &B) { return A.matmul(B); }
inline MatZonotope operator*(const MatZonotope &A, const Tensor &M) { return A.matmul(M); }
inline MatZonotope operator*(const Tensor &M, const MatZonotope &A) {
    return MatZonotope::matmul(M, A);
}
inline MatZonotope operator*(double s, const MatZonotope &A) { return A.matmul(s); }
inline MatZonotope operator*(const MatZonotope &A, double s) { return A.matmul(s); }
inline Zonotope operator*(const MatZonotope &A, const Zonotope &Z) { return A.matmul(Z); }
inline MatZonotope operator+(const MatZonotope &A, const MatZonotope &B) { return A.plus(B); }
inline MatZonotope operator+(const MatZonotope &A, const Tensor &M) { return A.plus(M); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
