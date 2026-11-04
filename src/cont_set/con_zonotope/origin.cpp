// origin - the constrained zonotope {0}, as CORA's conZonotope.origin
//
// Syntax:   ConZonotope cz = ConZonotope::origin(n);
// Inputs:   n - dimension, n >= 1
// Outputs:  cz - the center 0 (n, 1), no generators and no constraints
// See also: empty, Zonotope::origin
//
// CORA: contSet/@conZonotope/origin.m

#include "cora/cont_set/con_zonotope/con_zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

CORA_FROM_NUMBERS ConZonotope ConZonotope::origin(int64_t n) {
    detail::Entry entry;
    CORA_CHECK(n >= 1, Error::wrong_value,
               [&] { return "origin: the dimension must be positive, got " + std::to_string(n); });
    return {Tensor::zeros({n, 1}), Tensor::zeros({n, 0})};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
