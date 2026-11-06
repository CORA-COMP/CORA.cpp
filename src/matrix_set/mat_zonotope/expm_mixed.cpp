// expm_mixed - the matrix exponential of a matrix zonotope in two parts, as CORA's expmMixed
//
// The step is r; R is a matrix zonotope. The rest is for the interval matrix around A, continued
// from the box of A times the power of the intermediate order. The result also holds the powers
// of both parts and the remainder box.
//
// Syntax:   auto R = A.expm_mixed(r, intermediate_order, max_order);
// Inputs:   r - the step, positive
//           intermediate_order - where the first part ends, at least 2;
//           max_order - the order of the series, nonnegative;  A is square
// Outputs:  R - enclosure: I + M + M^2 / 2 with dependent terms at step r, plus terms to order k
//           enclosure_rest - the series from the next order to max_order with the remainder
// See also: expm, dependent_terms, IntervalMatrix::expm
//
// CORA: matrixSet/@matZonotope/expmMixed.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/matrix_set/mat_zonotope/mat_zonotope.h"
#include "global/format.h"
#include "matrix_set/interval_matrix/private/priv_factorial.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

MatZonotope::ExpmMixedResult MatZonotope::expm_mixed(double r, int64_t intermediate_order,
                                                     int64_t max_order) const {
    detail::Entry entry;
    CORA_CHECK(rows() == cols(), Error::dimension_mismatch, [&] {
        return "expm_mixed: the matrix zonotope must be square, got the shape " +
               shape_text(C_.shape());
    });
    // CORA asks for a nonnegative step and divides by it, which fails for 0
    CORA_CHECK(r > 0, Error::wrong_value,
               [&] { return "expm_mixed: the step must be positive, got " + format_number(r); });
    CORA_CHECK(intermediate_order >= 2, Error::wrong_value, [&] {
        return "expm_mixed: the intermediate order is too low, give at least 2, got " +
               std::to_string(intermediate_order);
    });
    CORA_CHECK(max_order >= 0, Error::wrong_value, [&] {
        return "expm_mixed: the order must be nonnegative, got " + std::to_string(max_order);
    });
    // the first part sums the terms to the intermediate order, without a remainder
    auto [square, H] = ((*this) * (1.0 / r)).dependent_terms(r);
    std::vector<MatZonotope> zonotope_powers{*this, square};
    MatZonotope first = H;
    for (int64_t order = 3; order <= intermediate_order; ++order) {
        zonotope_powers.push_back(zonotope_powers.back() * (*this));
        first = first + zonotope_powers.back() * (1.0 / priv_factorial(order));
    }
    // the second part continues the interval matrix series from the next power
    const IntervalMatrix box = interval_matrix();
    const IntervalMatrix start =
        box.matmul(zonotope_powers[intermediate_order - 1].interval_matrix());
    const ExpmResult rest = box.expm(r, max_order, intermediate_order + 1, start);
    return {first, rest.enclosure, std::move(zonotope_powers), rest.powers, rest.remainder};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
