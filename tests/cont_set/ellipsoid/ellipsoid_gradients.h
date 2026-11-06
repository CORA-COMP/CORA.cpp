// ellipsoid_gradients - what the torch tests of the ellipsoid share: inputs and outputs of a check
//
// Syntax:   gradcheck("interval", [](const Inputs &x) { return parts(ellipsoid_of(x).interval());
// },
//                     random_ellipsoid_factors(3));
// Sets:     two inputs (L, q) per set, with Q = L L' + I/2 symmetric by construction; parts are the
//           outputs of a result
// See also: cont_set/zonotope/zonotope_gradients.h
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cont_set/zonotope/zonotope_gradients.h"
#include "cora/cont_set/ellipsoid/ellipsoid.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace test {

/// The shape matrix and center of an ellipsoid, as the outputs of a function.
inline Inputs parts(const cora::Ellipsoid &E) { return {E.Q(), E.q()}; }

/// The ellipsoid of the factor L (n, n) and the center q (n, 1) from `first` on.
inline cora::Ellipsoid ellipsoid_of(const Inputs &x, std::size_t first = 0) {
    const cora::Tensor &L = x[first];
    const cora::Tensor LL = L.matmul(L.matrix_transpose());
    const cora::Tensor Q = (LL + LL.matrix_transpose()) * 0.5 + L.eye_like(L.shape()[0]) * 0.5;
    return {Q, x[first + 1]};
}

/// L (n, n) and q (n, 1) away from zero.
inline std::vector<torch::Tensor> random_ellipsoid_factors(int64_t n) {
    const std::vector<torch::Tensor> zonotope = random_zonotope(n, n);
    return {zonotope[1], zonotope[0]};
}

} // namespace test

// ---------------------------------------  END OF CODE  ---------------------------------------- //
