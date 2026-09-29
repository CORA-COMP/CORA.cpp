// plus - the Minkowski sum, as CORA's zonotope.plus
//
// Syntax:   Zres = Z.plus(Z2);
// Inputs:   Z2 - zonotope of the same dimension
// Outputs:  Zres - Z + Z2: center c + c2, the generators of both
// See also: mtimes, linComb

#include "contSet/zonotope/zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::ct {


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::plus(const Zonotope &Z2) const {
    return {c + Z2.c, Tensor::catLast({G, Z2.G})};
}

} // namespace cora::ct

// ---------------------------------------  END OF CODE  ---------------------------------------- //
