// radius - a bound on the distance of the zonotope from its center
//
// r is the smaller of the sum of the lengths of the generators and the 2-norm of the radius of
// the box around Z.
//
// Syntax:   r = Z.radius();
// Inputs:   -
// Outputs:  r - (..., 1, 1): every point lies within r of the center
// See also: generator_length, interval, norm
//
// CORA: contSet/@zonotope/radius.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Zonotope::radius() const {
    detail::Entry entry;
    const Tensor lengths = G_.vector_norm(Tensor::Norm::two, -1).sum(-1, true).expand_dims(-1);

    const Tensor box = interval().rad().vector_norm(Tensor::Norm::two, -2, true);
    return lengths.minimum(box);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
