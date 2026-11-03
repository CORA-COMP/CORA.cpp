// origin - the box {0}, as CORA's interval.origin
//
// Syntax:   Interval I = Interval::origin(n);
// Inputs:   n - dimension, n >= 1
// Outputs:  I - [0, 0]^n as columns (n, 1)
// See also: unbounded, empty
//
// CORA: contSet/@interval/origin.m

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

CORA_FROM_NUMBERS Interval Interval::origin(int64_t n) {
    detail::Entry entry;
    CORA_CHECK(n >= 1, Error::wrong_value, [&] {
        return "origin: the dimension must be positive, got " + std::to_string(n);
    });
    const Tensor zero = Tensor::zeros({n, 1});
    return {zero, zero};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
