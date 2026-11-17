// expm_mixed - the matrix exponential in two parts, as CORA's expmMixed
//
// r is the step and k the intermediate order. The rest is continued as expm does. The result also
// holds the powers of both parts and the remainder.
//
// Syntax:   mat_exp::ExpmMixedResult R = mat_exp::expm_mixed(A, r, intermediate_order, max_order);
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
// Last update:   17-November-2026
// Reviewed:      ---

#include "cont_dynamics/linear_sys/mat_exp/private/priv_expm_series.h"
#include "cont_dynamics/linear_sys/mat_exp/private/priv_factorial.h"
#include "cora/cont_dynamics/linear_sys/mat_exp/mat_exp.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

mat_exp::ExpmMixedResult mat_exp::expm_mixed(const Interval &A, double r,
                                             int64_t intermediate_order, int64_t max_order) {
    detail::Entry entry;
    CORA_CHECK(A.inf().size(-2) == A.inf().size(-1), Error::dimension_mismatch, [&] {
        return "expm_mixed: the matrix must be square, got the shape " +
               shape_text(A.inf().shape());
    });
    CORA_CHECK(r != 0, Error::wrong_value,
               [] { return std::string("expm_mixed: the step must not be 0"); });
    CORA_CHECK(intermediate_order >= 2, Error::wrong_value, [] {
        return std::string("expm_mixed: the intermediate order is too low, give at least 2");
    });
    // the first part sums the terms to the intermediate order, without a remainder
    const auto [square, H] = dependent_terms(A * (1.0 / r), r);
    std::vector<Interval> powers{A, square};
    Interval first = H;
    for (int64_t order = 3; order <= intermediate_order; ++order) {
        powers.push_back(powers.back().matmul(A));
        first = priv_sum(first, powers.back() * (1.0 / priv_factorial(order)));
    }
    // the second continues from the next power, as the five-argument expm
    const Interval start = A.matmul(powers[intermediate_order - 1]);
    const mat_exp::ExpmResult rest = expm(A, r, max_order, intermediate_order + 1, start);
    return {first, rest.enclosure, std::move(powers), rest.powers, rest.remainder};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
