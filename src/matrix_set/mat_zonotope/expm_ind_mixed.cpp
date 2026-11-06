// expm_ind_mixed - the matrix exponential of a matrix zonotope, independent powers
//
// The terms to the intermediate order are summed as matrix zonotopes without the dependent terms of
// expm_mixed (exp(A) ~ I + sum A^i / i!); the rest is the interval matrix series of the box around
// the set, continued from the box of A times the power of the intermediate order, with its
// remainder.
//
// Syntax:   auto R = A.expm_ind_mixed(intermediate_order, max_order);
// Inputs:   intermediate_order - where the first part ends, at least 1
//           max_order - the order of the series, nonnegative;  A is square
// Outputs:  R - enclosure: the first part; enclosure_rest: the rest to max_order; powers and
// powers2 as expm_mixed; remainder the box of the rest See also: expm_mixed,
// IntervalMatrix::expm_independent
//
// CORA: matrixSet/@matZonotope/expmIndMixed.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/matrix_set/mat_zonotope/mat_zonotope.h"
#include "global/format.h"
#include "matrix_set/interval_matrix/private/priv_factorial.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

MatZonotope::ExpmMixedResult MatZonotope::expm_ind_mixed(int64_t intermediate_order,
                                                         int64_t max_order) const {
    detail::Entry entry;
    CORA_CHECK(rows() == cols(), Error::dimension_mismatch, [&] {
        return "expm_ind_mixed: the matrix zonotope must be square, got the shape " +
               shape_text(C_.shape());
    });
    CORA_CHECK(intermediate_order >= 1, Error::wrong_value, [&] {
        return "expm_ind_mixed: the intermediate order must be at least 1, got " +
               std::to_string(intermediate_order);
    });
    CORA_CHECK(max_order >= 0, Error::wrong_value, [&] {
        return "expm_ind_mixed: the order must be nonnegative, got " + std::to_string(max_order);
    });
    // the first part: I + A + A^2 / 2! + ... to the intermediate order
    std::vector<MatZonotope> zonotope_powers = powers(intermediate_order);
    MatZonotope first = matrix_power(0);
    for (int64_t order = 1; order <= intermediate_order; ++order)
        first = first + zonotope_powers[std::size_t(order - 1)] * (1.0 / priv_factorial(order));
    // the second part continues the interval matrix series from the next power
    const IntervalMatrix box = interval_matrix();
    const IntervalMatrix start = box.matmul(zonotope_powers.back().interval_matrix());
    const ExpmResult rest = box.expm_independent(max_order, intermediate_order + 1, start);
    return {first, rest.enclosure, std::move(zonotope_powers), rest.powers, rest.remainder};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
