// prod - the product of the rows or the columns of an interval matrix, as CORA's interval.prod
//
// Syntax:   Interval R = I.prod();   Interval R = I.prod(dim);
// Inputs:   dim - 1: multiply the rows (a row of column products), 2: the columns (a column of
//           row products); default: 1, or 2 for a single row
// Outputs:  R - the product of the entries of each column (row) with the interval product, a
//           NaN corner (0 * Inf) taken as 0. The factors are folded from the last to the first,
//           as CORALean's prod, so the roundings agree with it
// See also: times, sum
//
// CORA: contSet/@interval/prod.m

#include "cont_set/interval/private/priv_times.h"
#include "cora/cont_set/interval/interval.h"

#include <tuple>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::prod(std::optional<int> dim) const {
    detail::Entry entry;
    const int along = dim.value_or(inf_.size(-2) == 1 ? 2 : 1);
    CORA_CHECK(along == 1 || along == 2, Error::wrong_value,
               [] { return std::string("prod: the second argument must be either 1 or 2"); });
    // columns are rows of the transpose
    const Tensor a = along == 1 ? inf_ : inf_.matrix_transpose();
    const Tensor b = along == 1 ? sup_ : sup_.matrix_transpose();
    Tensor lo = a.slice(-2, 0, 1).ones_like(), hi = lo;
    for (int64_t i = a.size(-2) - 1; i >= 0; --i)
        std::tie(lo, hi) = priv_times(a.slice(-2, i, i + 1), b.slice(-2, i, i + 1), lo, hi);
    return along == 1 ? Interval(lo, hi) : Interval(lo.matrix_transpose(), hi.matrix_transpose());
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
