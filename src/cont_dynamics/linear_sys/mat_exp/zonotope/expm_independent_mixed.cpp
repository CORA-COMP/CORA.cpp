// expm_independent_mixed - the matrix exponential of a matrix zonotope, independent powers
//
// The terms to the intermediate order are summed as matrix zonotopes without the dependent terms of
// expm_mixed (exp(A) ~ I + sum A^i / i!); the rest is the interval matrix series of the box around
// the set, continued from the box of A times the power of the intermediate order, with its
// remainder.
//
// Syntax:   auto R = mat_exp::expm_independent_mixed(A, intermediate_order, max_order);
// Inputs:   intermediate_order - where the first part ends, at least 1
//           max_order - the order of the series, nonnegative;  A is square
// Outputs:  R - enclosure: the first part; enclosure_rest: the rest to max_order; powers and
// powers2 as expm_mixed; remainder the box of the rest See also: expm_mixed,
// mat_exp::expm_independent
//
// CORA: matrixSet/@matZonotope/expmIndMixed.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/mat_exp/private/priv_factorial.h"
#include "cora/cont_dynamics/linear_sys/mat_exp/mat_exp.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

mat_exp::ZonotopeExpmMixedResult
mat_exp::expm_independent_mixed(const Zonotope &A, int64_t intermediate_order, int64_t max_order) {
    detail::Entry entry;
    CORA_CHECK(A.center().size(-2) == A.center().size(-1), Error::dimension_mismatch, [&] {
        return "expm_independent_mixed: the matrix zonotope must be square, got the shape " +
               shape_text(A.center().shape());
    });
    CORA_CHECK(intermediate_order >= 1, Error::wrong_value, [&] {
        return "expm_independent_mixed: the intermediate order must be at least 1, got " +
               std::to_string(intermediate_order);
    });
    CORA_CHECK(max_order >= 0, Error::wrong_value, [&] {
        return "expm_independent_mixed: the order must be nonnegative, got " +
               std::to_string(max_order);
    });
    // the first part: I + A + A^2 / 2! + ... to the intermediate order
    std::vector<Zonotope> zonotope_powers = powers(A, intermediate_order);
    Zonotope first = matrix_power(A, 0);
    for (int64_t order = 1; order <= intermediate_order; ++order)
        first = first + zonotope_powers[std::size_t(order - 1)] * (1.0 / priv_factorial(order));
    // the second part continues the interval matrix series from the next power
    const Interval box = A.interval();
    const Interval start = box.matmul(zonotope_powers.back().interval());
    const mat_exp::ExpmResult rest =
        mat_exp::expm_independent(box, max_order, intermediate_order + 1, start);
    return {first, rest.enclosure, std::move(zonotope_powers), rest.powers, rest.remainder};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
