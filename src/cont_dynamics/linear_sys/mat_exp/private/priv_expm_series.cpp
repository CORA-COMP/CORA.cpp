// priv_expm_series - the Taylor series of the interval matrix exponential, from a given start
//
// See also: priv_expm_series.h
//
// CORA.cpp only: a helper of the interval matrix exponential.
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   17-November-2026
// Reviewed:      ---

#include "cont_dynamics/linear_sys/mat_exp/private/priv_expm_series.h"
#include "cont_dynamics/linear_sys/mat_exp/private/priv_factorial.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

mat_exp::ExpmResult priv_expm_series(const Interval &A, const Interval &start, int64_t first_order,
                                     const Interval &first_power, int64_t loop_from,
                                     int64_t max_order) {
    std::vector<Interval> powers = mat_exp::powers(A, max_order, first_order, first_power);
    Interval sum = start;
    for (int64_t order = loop_from; order <= max_order; ++order)
        sum = priv_sum(sum, powers[order - first_order] * (1.0 / priv_factorial(order)));
    const Interval remainder = mat_exp::exponential_remainder(A, max_order);
    return {priv_sum(sum, remainder), std::move(powers), first_order, remainder};
}

mat_exp::ExpmResult priv_expm_continued(const Interval &A, int64_t max_order, int64_t initial_order,
                                        const Interval &initial_power) {
    const Interval start = initial_power * (1.0 / priv_factorial(initial_order));
    return priv_expm_series(A, start, initial_order, initial_power, initial_order + 1, max_order);
}

Interval priv_power(const Interval &A, int64_t exponent) {
    // the power A (A (... A)) of the interval matrix products, a 1 x 1 matrix as well
    Interval P = A;
    for (int64_t i = 1; i < exponent; ++i) P = A.matmul(P);
    return P;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
