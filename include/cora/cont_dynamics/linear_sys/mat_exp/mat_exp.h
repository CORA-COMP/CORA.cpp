// mat_exp - the matrix exponential of an interval matrix, as CORA's intervalMatrix expm family
//
// An interval matrix is an Interval of matrix bounds (..., n, n). Its exponential, remainders,
// dependent terms and powers are the linear-system code they serve, as CORALean's LinearSys.MatExp,
// not methods of the set: free functions of namespace cora::mat_exp.
//
// Syntax:     mat_exp::ExpmResult E = mat_exp::expm(A, max_order);
//             auto [square, H] = mat_exp::dependent_terms(A, r);
// Operations: dependent_terms, exponential_remainder, powers, expm, expm_independent, expm_mixed,
//             expm_independent_mixed, expm_taylor_horner_scaling, expm_different_algorithms,
//             expm_norm_infinity, expm_absolute_bound (one file each in
//             src/cont_dynamics/linear_sys/mat_exp/)
// See also:   cora/cont_set/interval/interval.h, cora/matrix_set/mat_zonotope/mat_zonotope.h
//
// CORA.cpp only: the declarations of the matrix exponential functions; each file tags its CORA file
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   17-November-2026
// Reviewed:      ---

#pragma once

#include "cora/cont_set/interval/interval.h"

#include <optional>
#include <utility>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::mat_exp {

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

} // namespace cora::mat_exp

// ---------------------------------------  END OF CODE  ---------------------------------------- //
