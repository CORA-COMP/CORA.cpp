// linComb - a zonotope enclosing the segments between two zonotopes, as CORA's linComb
//
// Syntax:   Zres = Z.linComb(Z2);
// Inputs:   Z2 - zonotope Z2 = M*Z + t: it shares its generator factors with Z (m generators)
// Outputs:  Zres - encloses conv(Z, Z2) with 2m + 1 generators:
//           center (c+c2)/2; generators (G+G2)/2, (G-G2)/2 and (c-c2)/2
// See also: mtimes, plus

#include "contSet/zonotope/zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::ct {


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::linComb(const Zonotope &Z2) const {
    const Tensor mean = (G + Z2.G) * 0.5;  // the shared part of matching generators
    const Tensor spread = (G - Z2.G) * 0.5;  // how far apart they are
    const Tensor shift = (c - Z2.c) * 0.5;  // the distance of the centers
    return {(c + Z2.c) * 0.5, Tensor::catLast({mean, spread, shift})};
}

} // namespace cora::ct

// ---------------------------------------  END OF CODE  ---------------------------------------- //
