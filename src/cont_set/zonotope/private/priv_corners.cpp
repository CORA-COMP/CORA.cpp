// priv_corners - the corners of a zonotope
//
// Syntax:   Tensor corners = priv_corners(Z);
// Inputs:   Z - zonotope with m generators
// Outputs:  corners - c + G s for the 2^m sign vectors s in {-1, 1}^m, as columns (..., n, 2^m);
//           they contain the vertices, so a convex set holds Z exactly if it holds them
// See also: Capsule::contains, ConZonotope::contains
//
// CORA.cpp only: containment tests need only the corners, which MATLAB takes from vertices

#include "cont_set/zonotope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor priv_corners(const Zonotope &Z) {
    const int64_t m = Z.generators().shape().back(), count = int64_t{1} << m;
    std::vector<double> signs(std::size_t(m * count));
    for (int64_t j = 0; j < m; ++j)
        for (int64_t k = 0; k < count; ++k) signs[j * count + k] = ((k >> j) & 1) ? 1.0 : -1.0;
    return Z.center() + Z.generators().matmul(Tensor::like(Z.center(), signs, {m, count}));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
