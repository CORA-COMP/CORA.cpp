// mat_zonotope - lean::MatZonotope, a matrix zonotope of exact values plus an error box (CORALean)
//
// Syntax:     lean::MatZonotope A(mZ);   A.matmul(B).plus(C).transpose();   A.gather();
// Values:     the matrices C + sum_i b_i G_i of exact values (a list of generator matrices) and an
//             interval matrix E that the sound float operations add; together they enclose the real
//             result
// Operations: one method per `matZonotope.*` op of the oracle, named as CORA.cpp names CORA's
//             (mtimes is matmul, mpower is matrix_power, intervalMatrix is interval_matrix)
// Crossing:   gather() returns the nominal part as a cora::MatZonotope and the error as an
//             interval, exactly or throws
// See also:   lean/zonotope.h, lean/interval_matrix.h, lean/request.h, mat_zonotope/mat_zonotope.h
//
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/global/oracle/zonotope.h"
#include "cora/matrix_set/mat_zonotope/mat_zonotope.h"

#include <utility>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

class MatZonotope {
  public:
    /// An exact matrix zonotope (empty error box) in the current dtype.
    MatZonotope(const cora::MatZonotope &A);
    MatZonotope(Tensor c, std::vector<Tensor> G, Interval E);

    /// The center (rows, cols), the generators (rows, cols each), the error box (rows, cols).
    const Tensor &center() const noexcept { return c_; }
    const std::vector<Tensor> &generators() const noexcept { return G_; }
    const Interval &error() const noexcept { return E_; }

    /// The center as the oracle returns it, and the box of the set (the error box included).
    Tensor center_value() const;
    Interval interval_matrix() const;

    /// The sum and the product with another matrix zonotope (CORA's plus and mtimes).
    MatZonotope plus(const MatZonotope &B) const;
    MatZonotope matmul(const MatZonotope &B) const;

    /// The transpose, and the power A^k of a square set (the identity for 0).
    MatZonotope transpose() const;
    MatZonotope matrix_power(int64_t exponent) const;

    /// The power P A^k: the initial power P times A, k times from the right.
    MatZonotope powers(const MatZonotope &P, int64_t k) const;

    /// The enclosure of exp(A) of a square set (CORA's expm calls expmInd): the Taylor series to
    /// max_order and a remainder box from a bound of the norm, coarser than CORA's.
    MatZonotope expm(int64_t max_order) const;
    MatZonotope expm_independent(int64_t max_order) const;

    /// An upper bound of the norm "1", "inf" or "fro" of every matrix of the set; `steps` tunes the
    /// square root of the Frobenius norm (CORA has no such argument).
    double norm(const std::string &type, int64_t steps = 20) const;

    /// Whether the stacked set is the point or the origin ("point", "origin"), exactly.
    bool representsa(const std::string &type, double tol = 0) const;

    /// The set of the stacked matrices (column by column) as a zonotope.
    Zonotope zonotope() const;

    /// The nominal part as a cora::MatZonotope and the error as a cora::Interval (double, exact).
    std::pair<cora::MatZonotope, cora::Interval> gather() const;

    Json to_json() const;
    static MatZonotope from_json(const std::string &dtype, const Json &j);

  private:
    Tensor c_;
    std::vector<Tensor> G_;
    Interval E_;
};

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
