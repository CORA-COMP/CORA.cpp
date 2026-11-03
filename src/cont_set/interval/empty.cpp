// empty - the empty interval, as CORA's interval.empty
//
// Syntax:   Interval I = Interval::empty(n);
// Inputs:   n - dimension, n >= 0
// Outputs:  I - bounds of the shape (n, 0), no columns; is_empty() is true
// See also: unbounded, Interval::is_empty
//
// CORA: contSet/@interval/empty.m

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

CORA_FROM_NUMBERS Interval Interval::empty(int64_t n) {
    detail::Entry entry;
    CORA_CHECK(n >= 0, Error::wrong_value, [&] {
        return "empty: the dimension must be nonnegative, got " + std::to_string(n);
    });
    return {Tensor::zeros({n, 0}), Tensor::zeros({n, 0})};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
