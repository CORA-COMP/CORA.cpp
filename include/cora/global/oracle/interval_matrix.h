// interval_matrix - lean::IntervalMatrix, an interval matrix of exact values, and its oracle ops
//
// Syntax:     lean::IntervalMatrix A(lean::Interval(inf, sup));   A.exact_square();
// Values:     the bounds are exact; an operation returns what the oracle proves to enclose the real
//             result (outward rounded); a number or a matrix is a point of the dtype
// Operations: one method per `intervalMatrix.*` op of the oracle, named as CORA.cpp names CORA's
//             (mtimes is matmul, mpower is matrix_power, infNorm is infinity_norm)
// Options:    CORA's option names; `tol` is a plain number; `steps` tunes the square root of the
//             Frobenius norm (CORA has no such argument)
// See also:   lean/interval.h, lean/request.h
//
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   04-November-2026
// Reviewed:      ---

#pragma once

#include "cora/global/oracle/interval.h"

#include <utility>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

/// The result of vertices: the corner matrices and a box E that encloses the rounding errors.
struct MatrixVertices {
    std::vector<Tensor> V;
    Interval E;
};

class IntervalMatrix {
  public:
    explicit IntervalMatrix(Interval bounds) : bounds_(std::move(bounds)) {}

    const Interval &interval() const noexcept { return bounds_; }

    // Matrices of the set ----------------------------------------------------------------------

    /// The entrywise largest absolute value, the lower bound matrix, the upper bound matrix and
    /// the radius matrix.
    Tensor abs() const;
    Tensor infimum() const;
    Tensor supremum() const;
    Tensor rad() const;

    /// The center, bracketed: an interval matrix of the lower and the upper end.
    Interval center() const;

    /// Whether M is in the set, as the oracle decides: true is certain.
    bool contains(const Tensor &M, double tol = 0) const;

    /// The sum with another interval matrix.
    Interval plus(const Interval &B) const;

    /// Whether the entries stacked into a vector are a set of the type (emptySet, point or
    /// origin), as the oracle decides: true is certain.
    bool representsa(const std::string &type, double tol = 0) const;

    /// The corner matrices (2^(m n), at most 2^20, repetitions kept) and a box E that encloses
    /// the rounding errors.
    MatrixVertices vertices() const;

    /// An upper bound of the volume of the box of the entries.
    double volume() const;

    // Squares, powers and products -------------------------------------------------------------

    /// The exact square, the pair of r^2 [A]^2 and I + [A] r + [A]^2 r^2 / 2 (the dependent terms).
    Interval exact_square() const;
    std::pair<Interval, Interval> dependent_terms(double r) const;

    /// The power with k factors, and the product with another interval matrix.
    Interval matrix_power(int64_t exponent) const;
    Interval matmul(const Interval &B) const;

    /// The power of order k + 1 from the initial power P: P * A^k, the factors from the right.
    Interval powers(const Interval &P, int64_t k) const;

    /// The enclosure of exp(A) of a square set: the Taylor series to max_order and a remainder
    /// from a bound of the norm (coarser than CORA's exponentialRemainder).
    Interval expm_independent(int64_t max_order) const;

    // Norms --------------------------------------------------------------------------------------

    /// An upper bound of the infinity norm, and of the norm of type 1, "inf" or "fro".
    double infinity_norm() const;
    double norm(const std::string &type, int64_t steps = 20) const;

  private:
    Interval bounds_;
};

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
