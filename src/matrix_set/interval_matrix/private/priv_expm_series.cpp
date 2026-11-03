// priv_expm_series - the Taylor series of the interval matrix exponential, from a given start
//
// See also: priv_expm_series.h

#include "matrix_set/interval_matrix/private/priv_expm_series.h"
#include "matrix_set/interval_matrix/private/priv_factorial.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

ExpmResult priv_expm_series(const IntervalMatrix &A, const IntervalMatrix &start,
                            int64_t first_order, const IntervalMatrix &first_power,
                            int64_t loop_from, int64_t max_order) {
    std::vector<IntervalMatrix> powers = A.powers(max_order, first_order, first_power);
    IntervalMatrix sum = start;
    for (int64_t order = loop_from; order <= max_order; ++order)
        sum = priv_sum(sum, powers[order - first_order] * (1.0 / priv_factorial(order)));
    const IntervalMatrix remainder = A.exponential_remainder(max_order);
    return {priv_sum(sum, remainder), std::move(powers), first_order, remainder};
}

ExpmResult priv_expm_continued(const IntervalMatrix &A, int64_t max_order, int64_t initial_order,
                               const IntervalMatrix &initial_power) {
    const IntervalMatrix start = initial_power * (1.0 / priv_factorial(initial_order));
    return priv_expm_series(A, start, initial_order, initial_power, initial_order + 1, max_order);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
