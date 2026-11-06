// expm_mixed - the matrix exponential in two parts, as CORA's expmMixed
//
// r is the step and k the intermediate order. The rest is continued as expm does. The result also
// holds the powers of both parts and the remainder.
//
// Syntax:   ExpmMixedResult R = A.expm_mixed(r, intermediate_order, max_order);
// Inputs:   r - the step, nonzero
//           intermediate_order - where the first part ends, at least 2;
//           max_order - the order of the series;  A is square
// Outputs:  R - enclosure: I + [A] + [A]^2 / 2 with dependent terms at step r, plus terms to k
//           enclosure_rest - the series from the order k + 1 to max_order with the remainder
// See also: expm, dependent_terms
//
// CORA: matrixSet/@intervalMatrix/expmMixed.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
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

ExpmMixedResult IntervalMatrix::expm_mixed(double r, int64_t intermediate_order,
                                           int64_t max_order) const {
    detail::Entry entry;
    CORA_CHECK(rows() == cols(), Error::dimension_mismatch, [&] {
        return "expm_mixed: the matrix must be square, got the shape " +
               shape_text(int_.inf().shape());
    });
    CORA_CHECK(r != 0, Error::wrong_value,
               [] { return std::string("expm_mixed: the step must not be 0"); });
    CORA_CHECK(intermediate_order >= 2, Error::wrong_value, [] {
        return std::string("expm_mixed: the intermediate order is too low, give at least 2");
    });
    // the first part sums the terms to the intermediate order, without a remainder
    const auto [square, H] = ((*this) * (1.0 / r)).dependent_terms(r);
    std::vector<IntervalMatrix> powers{*this, square};
    IntervalMatrix first = H;
    for (int64_t order = 3; order <= intermediate_order; ++order) {
        powers.push_back(powers.back().matmul(*this));
        first = priv_sum(first, powers.back() * (1.0 / priv_factorial(order)));
    }
    // the second continues from the next power, as the five-argument expm
    const IntervalMatrix start = matmul(powers[intermediate_order - 1]);
    const ExpmResult rest = expm(r, max_order, intermediate_order + 1, start);
    return {first, rest.enclosure, std::move(powers), rest.powers, rest.remainder};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
