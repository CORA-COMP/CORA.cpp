// mat_exp - the exponential of an interval matrix or a matrix zonotope, as CORA's expm family
//
// An interval matrix is an Interval of matrix bounds (..., n, n), a matrix zonotope a Zonotope of
// matrix elements (Zonotope::matrix). Their exponentials, remainders, dependent terms and powers
// are the linear-system code they serve, as CORALean's LinearSys.MatExp: free functions of
// namespace cora::mat_exp.
//
// Syntax:     mat_exp::ExpmResult E = mat_exp::expm(A, max_order);
//             auto [square, H] = mat_exp::dependent_terms(A, r);
// Operations: dependent_terms, exponential_remainder, powers, expm, expm_independent, expm_mixed,
//             expm_independent_mixed, expm_taylor_horner_scaling, expm_different_algorithms,
//             expm_norm_infinity, expm_absolute_bound (one file each in
//             src/cont_dynamics/linear_sys/mat_exp/); of a matrix zonotope: dependent_terms,
//             matrix_power, powers, expm, expm_independent, expm_mixed, expm_independent_mixed,
//             expm_one_param (src/cont_dynamics/linear_sys/mat_exp/zonotope/)
// See also:   cora/cont_set/interval/interval.h, cora/cont_set/zonotope/zonotope.h
//
// CORA.cpp only: the declarations of the matrix exponential functions; each file tags its CORA file
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#pragma once

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/zonotope/zonotope.h"

#include <optional>
#include <utility>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::mat_exp {

// Interval matrices -------------------------------------------------------------------------------

/// The result of expm: the enclosure of the series, the powers of A that were multiplied
/// (powers[k] is of order first_order + k) and the remainder box.
struct ExpmResult {
    Interval enclosure;
    std::vector<Interval> powers;
    int64_t first_order;
    Interval remainder;
};

/// The result of expm_mixed: the sum of the terms to the intermediate order, the rest of the
/// series, the powers of both parts (orders 1.. and intermediate_order + 1..) and the remainder.
struct ExpmMixedResult {
    Interval enclosure, enclosure_rest;
    std::vector<Interval> powers, powers2;
    Interval remainder;
};

/// The dependent terms of the Taylor series of exp(M r): {r^2 [M]^2 (exact), I + [M] r +
/// [M]^2 r^2 / 2 (the loop's H)}, each rebuilt from center and radius. A is square.
std::pair<Interval, Interval> dependent_terms(const Interval &A, double r);

/// The remainder box [-W, W] of the Taylor series to order max_order: W = exp(|M|) - sum
/// |M|^i / i!; Inf where exp(|M|) is not a number. A is square.
Interval exponential_remainder(const Interval &A, int64_t max_order);

/// The powers [A]^i from order initial_order (default 1, A) to max_order, each the previous
/// times A from the right as CORA's powers; powers[k] is of order initial_order + k.
std::vector<Interval> powers(const Interval &A, int64_t max_order);
std::vector<Interval> powers(const Interval &A, int64_t max_order, int64_t initial_order,
                             const Interval &initial_power);

/// The enclosure of exp(M) over the set: the series to order max_order (default 10) with the
/// dependent terms of the second order and the remainder, at the step r = 1.
ExpmResult expm(const Interval &A, int64_t max_order = 10);

/// The same with the step r != 0: the dependent terms of [A] / r at the step r.
ExpmResult expm(const Interval &A, double r, int64_t max_order);

/// The series continued from initial_power = [A]^initial_order: its terms from that order on,
/// with the remainder (the lower ones are the caller's); r is not used, as in CORA.
ExpmResult expm(const Interval &A, double r, int64_t max_order, int64_t initial_order,
                const Interval &initial_power);

/// The enclosure of exp(M) with every power evaluated on its own, without the dependent terms
/// (CORA's expmInd), from order 0 with [A]^0 = I.
ExpmResult expm_independent(const Interval &A, int64_t max_order);

/// The same continued from initial_power = [A]^initial_order, as expm does.
ExpmResult expm_independent(const Interval &A, int64_t max_order, int64_t initial_order,
                            const Interval &initial_power);

/// The series in two parts: the sum to intermediate_order >= 2 with the dependent terms, and the
/// rest of it up to max_order with the remainder.
ExpmMixedResult expm_mixed(const Interval &A, double r, int64_t intermediate_order,
                           int64_t max_order);

/// The series in two parts with independent powers: the sum to intermediate_order >= 1, and the
/// rest of it up to max_order with the remainder (CORA's expmIndMixed).
ExpmMixedResult expm_independent_mixed(const Interval &A, int64_t intermediate_order,
                                       int64_t max_order);

/// The exponential by CORA's other algorithms, for the order max_order: style 0 the Taylor series,
/// 1 Horner's scheme, 2 Horner on [A] / 2^p squared p times. Nothing where the norm is too large.
std::optional<Interval> expm_taylor_horner_scaling(const Interval &A, int64_t max_order,
                                                   int64_t style, int64_t scaling_exponent = 1);

/// The same, and style 4: the enclosure of expm at the step 1.
std::optional<Interval> expm_different_algorithms(const Interval &A, int64_t max_order,
                                                  int64_t style, int64_t scaling_exponent = 1);

/// CORA's closed-form bound of the infinity norm of the exponential's remainder at time t
/// (expmNormInf), and its matrix bound (expmAbsoluteBound) for square A.
Tensor expm_norm_infinity(const Interval &A, double t);
Tensor expm_absolute_bound(const Interval &A, double t);

// Matrix zonotopes -------------------------------------------------------------------------------

/// The result of expm_mixed of a matrix zonotope: the terms to the intermediate order, the rest
/// of the series, the powers of orders 1 to the intermediate order, those of the second part and
/// the remainder box.
struct ZonotopeExpmMixedResult {
    Zonotope enclosure;
    Interval enclosure_rest;
    std::vector<Zonotope> powers;
    std::vector<Interval> powers2;
    Interval remainder;
};

/// The result of expm_one_param: the matrix zonotope of the Taylor terms, the remainder box, the
/// powers (powers[k] of order k + 1, a generator per power of the parameter) and, with an input,
/// the zonotope of the input terms.
struct ExpmOneParamResult {
    Zonotope enclosure;
    Interval remainder;
    std::vector<Zonotope> powers;
    std::optional<Zonotope> constant_input;
};

/// The terms of the Taylor series of exp(M r) that depend on one another, of a square matrix
/// zonotope A: (r^2 M^2, I + M r + M^2 r^2 / 2), each an enclosure over the set.
std::pair<Zonotope, Zonotope> dependent_terms(const Zonotope &A, double r);

/// The power A^k of a square matrix zonotope: the identity for k = 0, A for k = 1, else ((A A) A)
/// ... with k factors, every product Zonotope::matmul.
Zonotope matrix_power(const Zonotope &A, int64_t exponent);

/// The powers A^i for i = initial_order (the first entry) to max_order, each the previous one
/// times A from the right; the first is A (initial_order 1) or initial_power. A is square.
std::vector<Zonotope> powers(const Zonotope &A, int64_t max_order);
std::vector<Zonotope> powers(const Zonotope &A, int64_t max_order, int64_t initial_order,
                             const Zonotope &initial_power);

/// An enclosure of exp(M) over the set: expm_independent (CORA's expm calls expmInd).
Zonotope expm(const Zonotope &A, int64_t max_order = 10);

/// The series to max_order with every power evaluated on its own, plus the remainder box of the
/// interval matrix of the set.
Zonotope expm_independent(const Zonotope &A, int64_t max_order);

/// The series in two parts with independent powers: I + sum A^i / i! to intermediate_order >= 1
/// as a matrix zonotope, the rest up to max_order as interval matrices (CORA's expmIndMixed).
ZonotopeExpmMixedResult expm_independent_mixed(const Zonotope &A, int64_t intermediate_order,
                                               int64_t max_order);

/// The series in two parts: the terms to intermediate_order >= 2 as a matrix zonotope (with the
/// dependent terms of the step r > 0) and the rest up to max_order as interval matrices.
ZonotopeExpmMixedResult expm_mixed(const Zonotope &A, double r, int64_t intermediate_order,
                                   int64_t max_order);

/// The Taylor series of exp(M r) to max_order for the matrices C + b G_1 of a set with one
/// generator (CORALean's K(r), Althoff and Dolan, ACC 2012), with the remainder box beside it.
ExpmOneParamResult expm_one_param(const Zonotope &A, double r, int64_t max_order,
                                  const std::optional<Zonotope> &u_trans = std::nullopt);

} // namespace cora::mat_exp

// ---------------------------------------  END OF CODE  ---------------------------------------- //
