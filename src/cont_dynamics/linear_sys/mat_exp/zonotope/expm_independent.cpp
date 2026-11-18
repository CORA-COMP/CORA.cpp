// expm_independent - the matrix exponential of a matrix zonotope with independent powers
//
// The sum of A^i / i! for i = 0 to max_order (each power the product of the previous with A) plus
// the remainder box of the interval matrix around A.
//
// Syntax:   Zonotope E = mat_exp::expm_independent(A, max_order);
// Inputs:   max_order - the order of the series, nonnegative;  A is square
// Outputs:  E - an enclosure of exp(M) for every matrix M of A, as a matrix zonotope
// See also: expm, expm_mixed, mat_exp::exponential_remainder
//
// CORA: matrixSet/@matZonotope/expmInd.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/mat_exp/private/priv_factorial.h"
#include "cont_dynamics/linear_sys/mat_exp/private/priv_from_interval_matrix.h"
#include "cora/cont_dynamics/linear_sys/mat_exp/mat_exp.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope mat_exp::expm_independent(const Zonotope &A, int64_t max_order) {
    detail::Entry entry;
    CORA_CHECK(A.center().size(-2) == A.center().size(-1), Error::dimension_mismatch, [&] {
        return "expm_independent: the matrix zonotope must be square, got the shape " +
               shape_text(A.center().shape());
    });
    CORA_CHECK(max_order >= 0, Error::wrong_value, [&] {
        return "expm_independent: the order must be nonnegative, got " + std::to_string(max_order);
    });
    Zonotope sum = matrix_power(A, 0), power = matrix_power(A, 0);
    for (int64_t i = 1; i <= max_order; ++i) {
        power = power * A;
        sum = sum + power * (1.0 / priv_factorial(i));
    }
    const Interval remainder = mat_exp::exponential_remainder(A.interval(), max_order);
    return sum + priv_from_interval_matrix(remainder);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
