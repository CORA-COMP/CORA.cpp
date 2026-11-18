// expm_mixed - the matrix exponential of a matrix zonotope in two parts, as CORA's expmMixed
//
// The step is r; R is a matrix zonotope. The rest is for the interval matrix around A, continued
// from the box of A times the power of the intermediate order. The result also holds the powers
// of both parts and the remainder box.
//
// Syntax:   auto R = mat_exp::expm_mixed(A, r, intermediate_order, max_order);
// Inputs:   r - the step, positive
//           intermediate_order - where the first part ends, at least 2;
//           max_order - the order of the series, nonnegative;  A is square
// Outputs:  R - enclosure: I + M + M^2 / 2 with dependent terms at step r, plus terms to order k
//           enclosure_rest - the series from the next order to max_order with the remainder
// See also: expm, dependent_terms, mat_exp::expm
//
// CORA: matrixSet/@matZonotope/expmMixed.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/mat_exp/private/priv_factorial.h"
#include "cora/cont_dynamics/linear_sys/mat_exp/mat_exp.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

mat_exp::ZonotopeExpmMixedResult
mat_exp::expm_mixed(const Zonotope &A, double r, int64_t intermediate_order, int64_t max_order) {
    detail::Entry entry;
    CORA_CHECK(A.center().size(-2) == A.center().size(-1), Error::dimension_mismatch, [&] {
        return "expm_mixed: the matrix zonotope must be square, got the shape " +
               shape_text(A.center().shape());
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
    auto [square, H] = dependent_terms(A * (1.0 / r), r);
    std::vector<Zonotope> zonotope_powers{A, square};
    Zonotope first = H;
    for (int64_t order = 3; order <= intermediate_order; ++order) {
        zonotope_powers.push_back(zonotope_powers.back() * A);
        first = first + zonotope_powers.back() * (1.0 / priv_factorial(order));
    }
    // the second part continues the interval matrix series from the next power
    const Interval box = A.interval();
    const Interval start = box.matmul(zonotope_powers[intermediate_order - 1].interval());
    const mat_exp::ExpmResult rest =
        mat_exp::expm(box, r, max_order, intermediate_order + 1, start);
    return {first, rest.enclosure, std::move(zonotope_powers), rest.powers, rest.remainder};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
