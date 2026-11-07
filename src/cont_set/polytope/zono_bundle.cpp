// zono_bundle - the zonotope bundle of a bounded polytope, as CORA's polytope.zonoBundle
//
// For every row a of [A; Ae; -Ae] an orthonormal basis B with first vector a / |a| (Gram-Schmidt)
// gives the box of the vertices in the coordinates B; B times that box is a zonotope with the strip
// along a, and the intersection of all of them is the polytope. R^n and unbounded sets are
// Error::special_error. Not a batch.
//
// Syntax:   ZonoBundle zB = P.zono_bundle();
// Inputs:   -
// Outputs:  zB - the bundle with one zonotope per row
// See also: Polytope::zonotope, ZonoBundle::ZonoBundle
//
// CORA: contSet/@polytope/zonoBundle.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The zonotope B box(B' V) for the vertices V (n, p) and the basis B (n, n).
Zonotope aux_strip(const Tensor &V, const Tensor &B) {
    const Tensor coordinates = B.matrix_transpose().matmul(V);
    const Interval box(coordinates.min(-1, true), coordinates.max(-1, true));
    return box.zonotope().linear_map(B);
}

} // namespace


// ===========================================  MAIN  =========================================== //

ZonoBundle Polytope::zono_bundle() const {
    detail::Entry entry;
    const Tensor &like = is_h_rep() ? A_ : V_;
    CORA_CHECK(like.ndim() == 2, Error::not_supported, [] {
        return std::string("zono_bundle: it supports a single polytope, not a batch");
    });
    CORA_CHECK(!priv_is_fullspace(*this) && priv_polytope_is_bounded(*this), Error::special_error,
               [] {
                   return std::string("Polytope is unbounded and can therefore not be converted "
                                      "into a zonotope bundle.");
               });
    const Rows vertices = priv_polytope_vertices(*this);
    CORA_CHECK(!vertices.empty(), Error::special_error,
               [] { return std::string("The empty set is no zonotope bundle."); });
    const Tensor V = priv_points_tensor(like, dim(), vertices);
    // one strip per row of [A; Ae; -Ae]
    const HostH H = priv_host_h(constraints());
    std::vector<Zonotope> strips;
    for (const Row &a : priv_equality_to_inequality(H).first)
        strips.push_back(aux_strip(V, priv_points_tensor(like, dim(), priv_gram_schmidt(a))));
    return ZonoBundle(strips);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
