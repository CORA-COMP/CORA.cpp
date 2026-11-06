// unbounded - the box of all vectors, as CORA's interval.Inf
//
// Syntax:   Interval I = Interval::unbounded(n);
// Inputs:   n - dimension, n >= 0
// Outputs:  I - (-Inf, Inf)^n, as columns (n, 1)
// See also: Interval::cart_prod
//
// CORA: contSet/@interval/Inf.m
// CORA authors: Mark Wetzlinger
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

CORA_FROM_NUMBERS Interval Interval::unbounded(int64_t n) {
    detail::Entry entry;
    CORA_CHECK(n >= 0, Error::wrong_value, [&] {
        return "unbounded: the dimension must be nonnegative, got " + std::to_string(n);
    });
    const double kInf = std::numeric_limits<double>::infinity();
    return {Tensor::from_data(std::vector<double>(n, -kInf), {n, 1}),
            Tensor::from_data(std::vector<double>(n, kInf), {n, 1})};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
