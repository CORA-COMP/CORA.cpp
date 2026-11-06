// expm_independent_mixed - the matrix exponential with independent powers in two parts, as CORA's
//
// Syntax:   ExpmMixedResult R = A.expm_independent_mixed(intermediate_order, max_order);
// Inputs:   intermediate_order >= 1, max_order >= 0;  A is square
// Outputs:  R - enclosure: the sum of [A]^i / i! from the identity to intermediate_order;
//           enclosure_rest: the series continued from [A] [A]^intermediate_order (orders
//           intermediate_order + 1 to max_order) with the remainder box (as expm_independent);
//           the powers of both parts and the remainder (see ExpmMixedResult)
// See also: expm_independent, expm_mixed, powers
//
// CORA: matrixSet/@intervalMatrix/expmIndMixed.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/matrix_set/interval_matrix/interval_matrix.h"
#include "global/format.h"
#include "matrix_set/interval_matrix/private/priv_expm_series.h"
#include "matrix_set/interval_matrix/private/priv_factorial.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

ExpmMixedResult IntervalMatrix::expm_independent_mixed(int64_t intermediate_order,
                                                       int64_t max_order) const {
    detail::Entry entry;
    CORA_CHECK(rows() == cols(), Error::dimension_mismatch, [&] {
        return "expm_independent_mixed: the matrix must be square, got the shape " +
               shape_text(int_.inf().shape());
    });
    CORA_CHECK(intermediate_order >= 1 && max_order >= 0, Error::wrong_value, [] {
        return std::string("expm_independent_mixed: needs intermediate_order >= 1, max_order >= 0");
    });
    std::vector<IntervalMatrix> pow = powers(intermediate_order);
    IntervalMatrix first = matrix_power(0);
    for (int64_t i = 1; i <= intermediate_order; ++i)
        first = priv_sum(first, pow[i - 1] * (1.0 / priv_factorial(i)));
    const ExpmResult rest =
        expm_independent(max_order, intermediate_order + 1, matmul(pow[intermediate_order - 1]));
    return {first, rest.enclosure, std::move(pow), rest.powers, rest.remainder};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
