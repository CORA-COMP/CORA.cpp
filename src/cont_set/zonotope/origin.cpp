// origin - the zonotope {0}, as CORA's zonotope.origin
//
// Syntax:   Zonotope Z = Zonotope::origin(n);
// Inputs:   n - dimension, n >= 1
// Outputs:  Z - the zonotope with center 0 (n, 1) and no generators
// See also: Interval::origin
//
// CORA: contSet/@zonotope/origin.m

#include "cora/cont_set/zonotope/zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

CORA_FROM_NUMBERS Zonotope Zonotope::origin(int64_t n) {
    detail::Entry entry;
    CORA_CHECK(n >= 1, Error::wrong_value,
               [&] { return "origin: the dimension must be positive, got " + std::to_string(n); });
    return {Tensor::zeros({n, 1}), Tensor::zeros({n, 0})};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
