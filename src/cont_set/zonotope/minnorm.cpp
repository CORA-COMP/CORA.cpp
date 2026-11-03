// minnorm - the distance from the center of a zonotope to its boundary, and a point there
//
// Syntax:   [val, x] = Z.minnorm();
// Inputs:   -
// Outputs:  val - the least offset of a facet from the center, i.e. the radius of the largest
//           ball about the center in the zonotope, (1, 1); 0 if the generators do not span the
//           space;  x - the point of that facet closest to the center, (n, 1) (the center if val
//           is 0). Not a batch
// See also: norm, contains
//
// CORA: contSet/@zonotope/minnorm.m

#include "cont_set/zonotope/private/priv.h"
#include "cora/cont_set/zonotope/zonotope.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The tolerance of CORA's polytope(Z) for a generator matrix to count as rank deficient.
constexpr double kFlatTol = 1e-12;

} // namespace


// ===========================================  MAIN  =========================================== //

std::pair<Tensor, Tensor> Zonotope::minnorm() const {
    detail::Entry entry;
    CORA_CHECK(c_.ndim() == 2, Error::not_supported, [] {
        return std::string("minnorm: facets of one zonotope at a time; it supports a single "
                           "zonotope, not a batch");
    });
    // CORA's polytope(Z - c): without zero generators; a flat zonotope has a facet through c
    const Zonotope compacted = compact();
    const Tensor &G = compacted.generators();
    if (G.shape().back() == 0 || G.matrix_rank(kFlatTol) < dim())
        return {c_.zeros_like({1, 1}), c_};
    const Tensor C = priv_zonotope_facet_normals(compacted);
    // the offset of each facet of Z - c: sum |C G|, and the nearest one (the first on a tie)
    const Tensor offsets = C.matmul(G).abs().sum(-1, true);
    const Tensor nearest = offsets.argmin(0);
    const Tensor b = offsets.take(nearest, 0);
    const Tensor value = b.square().sqrt();
    return {value, C.take(nearest, 0).matrix_transpose().multiply(b) + c_};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
