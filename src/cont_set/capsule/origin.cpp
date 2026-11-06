// origin - the capsule that is the origin, as CORA's capsule.origin
//
// Syntax:   Capsule C = Capsule::origin(n);
// Inputs:   n - dimension, an integer >= 1
// Outputs:  C - center 0, generator 0 and radius 0 in R^n
// See also: empty
//
// CORA: contSet/@capsule/origin.m

#include "cora/cont_set/capsule/capsule.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

CORA_FROM_NUMBERS Capsule Capsule::origin(int64_t n) {
    detail::Entry entry;
    CORA_CHECK(n >= 1, Error::wrong_value, [&] {
        return "origin: the dimension must be a positive integer, got " + std::to_string(n);
    });
    return {Tensor::zeros({n, 1}), Tensor::zeros({n, 1}), 0.0};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
