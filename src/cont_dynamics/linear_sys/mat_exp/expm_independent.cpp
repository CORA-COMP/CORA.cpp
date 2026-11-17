// expm_independent - the matrix exponential with independent powers, as CORA's expmInd
//
// A is square. The sum starts from order 0 with the identity; every power is the interval matrix
// product. The result also holds the powers and the remainder (see mat_exp::ExpmResult).
//
// Syntax:   mat_exp::ExpmResult R = mat_exp::expm_independent(A, max_order);
//           R = mat_exp::expm_independent(A, max_order, initial_order, initial_power);
// Inputs:   max_order - the order of the series
//           initial_order, initial_power - continue from [A]^initial_order, as given
// Outputs:  R - enclosure: the sum of [A]^i / i! from the initial order to max_order, plus a box
// See also: expm, exponential_remainder
//
// CORA: matrixSet/@intervalMatrix/expmInd.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   17-November-2026
// Reviewed:      ---

#include "cont_dynamics/linear_sys/mat_exp/private/priv_expm_series.h"
#include "cora/cont_dynamics/linear_sys/mat_exp/mat_exp.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

mat_exp::ExpmResult mat_exp::expm_independent(const Interval &A, int64_t max_order) {
    detail::Entry entry;
    if (entry.check()) mat_exp::check_square_order("expm_independent", A, max_order);
    // CORA starts the powers at [A]^1 = A and the sum at the identity [A]^0
    return priv_expm_series(A, priv_identity(A), 1, A, 1, max_order);
}

mat_exp::ExpmResult mat_exp::expm_independent(const Interval &A, int64_t max_order,
                                              int64_t initial_order,
                                              const Interval &initial_power) {
    detail::Entry entry;
    if (entry.check())
        mat_exp::check_continued_input("expm_independent", A, max_order, initial_order,
                                       initial_power);
    return priv_expm_continued(A, max_order, initial_order, initial_power);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
