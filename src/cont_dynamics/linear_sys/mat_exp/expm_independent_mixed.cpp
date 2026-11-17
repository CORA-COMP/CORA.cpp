// expm_independent_mixed - the matrix exponential with independent powers in two parts, as CORA's
//
// A is square. The rest has its remainder box as in expm_independent. The result also holds the
// powers of both parts and the remainder (see mat_exp::ExpmMixedResult).
//
// Syntax:   mat_exp::ExpmMixedResult R = mat_exp::expm_independent_mixed(A, intermediate_order,
// max_order); Inputs:   intermediate_order >= 1, max_order >= 0 - the orders of the two parts
// Outputs:  R - enclosure: the sum of [A]^i / i! from the identity to intermediate_order
//           enclosure_rest - the series continued from [A] [A]^intermediate_order, with a remainder
// See also: expm_independent, expm_mixed, powers
//
// CORA: matrixSet/@intervalMatrix/expmIndMixed.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   17-November-2026
// Reviewed:      ---

#include "cont_dynamics/linear_sys/mat_exp/private/priv_expm_series.h"
#include "cont_dynamics/linear_sys/mat_exp/private/priv_factorial.h"
#include "cora/cont_dynamics/linear_sys/mat_exp/mat_exp.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

mat_exp::ExpmMixedResult
mat_exp::expm_independent_mixed(const Interval &A, int64_t intermediate_order, int64_t max_order) {
    detail::Entry entry;
    CORA_CHECK(A.inf().size(-2) == A.inf().size(-1), Error::dimension_mismatch, [&] {
        return "expm_independent_mixed: the matrix must be square, got the shape " +
               shape_text(A.inf().shape());
    });
    CORA_CHECK(intermediate_order >= 1 && max_order >= 0, Error::wrong_value, [] {
        return std::string("expm_independent_mixed: needs intermediate_order >= 1, max_order >= 0");
    });
    std::vector<Interval> pow = powers(A, intermediate_order);
    Interval first = priv_identity(A);
    for (int64_t i = 1; i <= intermediate_order; ++i)
        first = priv_sum(first, pow[i - 1] * (1.0 / priv_factorial(i)));
    const mat_exp::ExpmResult rest = expm_independent(A, max_order, intermediate_order + 1,
                                                      A.matmul(pow[intermediate_order - 1]));
    return {first, rest.enclosure, std::move(pow), rest.powers, rest.remainder};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
