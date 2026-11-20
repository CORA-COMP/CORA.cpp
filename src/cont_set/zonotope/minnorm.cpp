// minnorm - the distance from the center of a zonotope to its boundary, and a point there
//
// val is the least offset of a facet from the center, 0 if the generators do not span the space;
// x is the center then. Not a batch.
//
// Syntax:   [val, x] = Z.minnorm();
// Inputs:   -
// Outputs:  val - the radius () of the largest ball about the center in the zonotope
//           x - the point (n) of the nearest facet closest to the center
// See also: norm, contains
//
// CORA: contSet/@zonotope/minnorm.m
// CORA authors: Victor Gassmann
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   20-November-2026
// Reviewed:      ---

#include "cont_set/zonotope/private/priv.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/tensor/tolerance.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::pair<Tensor, Tensor> Zonotope::minnorm() const {
    detail::Entry entry;
    CORA_CHECK(c_.ndim() == 1, Error::not_supported, [] {
        return std::string("minnorm: facets of one zonotope at a time; it supports a single "
                           "zonotope, not a batch");
    });
    // CORA's polytope(Z - c), on CORA's columns: no zero generators; a flat one has a facet at c
    const Zonotope compacted = compact();
    const Tensor G = compacted.generators().matrix_transpose(), c = c_.expand_dims(-1);
    if (G.shape().back() == 0 || G.matrix_rank(resolve_tol({}, G.dtype())) < dim())
        return {c.zeros_like({}), c_};
    const Tensor C = priv_zonotope_facet_normals(compacted);
    // the offset of each facet of Z - c: sum |C G|, and the nearest one (the first on a tie)
    const Tensor offsets = C.matmul(G).abs_sum();
    const Tensor nearest = offsets.argmin(0);
    const Tensor b = offsets.take(nearest, 0);
    const Tensor value = b.square().sqrt().squeeze(-1).squeeze(-1);
    // the nearest point of the boundary, as a vector (n)
    return {value, (C.take(nearest, 0).matrix_transpose().multiply(b) + c).squeeze(-1)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
