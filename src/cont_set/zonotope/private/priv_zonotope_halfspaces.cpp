// priv_zonotope_halfspaces - the halfspaces C x <= d whose intersection is the zonotope
//
// The exact form of CORA's polytope(Z): the facet normals of the cross products of the generators
// with both signs and the support values as offsets; a zonotope without generators is its box; a
// degenerate one (singular values up to twice numerics::kDegenerateZonotope) is the facets of its
// projection onto the span of the generators, in the rotated frame of the singular vectors, plus
// two halfspaces per flat direction.
//
// Syntax:   auto [C, d] = priv_zonotope_halfspaces(Z);
// Inputs:   Z - a single zonotope
// Outputs:  C - the normals as rows (k, n)
//           d - the offsets (k, 1)
// See also: priv_zonotope_facet_normals, Zonotope::unite
//
// CORA: contSet/@zonotope/polytope.m
// CORA authors: Niklas Kochdumper, Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/zonotope/private/priv.h"
#include "cora/global/tensor/tolerance.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The halfspaces +-C0 x <= +-C0 c + sum|C0 G| of the facet normals C0 of a full-dimensional
/// zonotope (CORA's aux_polytope_fullDim), the normals in the order of CORALean's hPolytopeFacets:
/// the reverse of CORA's.
std::pair<Tensor, Tensor> aux_full_dim(const Zonotope &Z) {
    const Tensor &c = Z.center(), &G = Z.generators();
    const Tensor facets = priv_zonotope_facet_normals(Z);
    std::vector<int64_t> reversed(std::size_t(facets.shape()[0]));
    for (std::size_t i = 0; i < reversed.size(); ++i)
        reversed[i] = int64_t(reversed.size() - 1 - i);
    const Tensor C0 = facets.take(reversed, 0);
    const Tensor shift = C0.matmul(c), spread = C0.matmul(G).abs().sum(-1, true);
    return {Tensor::concat({C0, C0.negative()}, 0),
            Tensor::concat({shift + spread, shift.negative() + spread}, 0)};
}

/// The box of a point: +-I x <= +-c.
std::pair<Tensor, Tensor> aux_point(const Tensor &c) {
    const Tensor I = c.eye_like(c.shape()[0]);
    return {Tensor::concat({I, I.negative()}, 0), Tensor::concat({c, c.negative()}, 0)};
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::pair<Tensor, Tensor> priv_zonotope_halfspaces(const Zonotope &Z_in) {
    const Zonotope Z = Z_in.compact("zeros", machine_eps(Z_in.center().dtype()));
    const Tensor &c = Z.center(), &G = Z.generators();
    const int64_t n = Z.dim(), m = G.shape().back();
    if (m == 0) return aux_point(c);
    if (Z.is_full_dim(numerics::kDegenerateZonotope)) return aux_full_dim(Z);

    // degenerate: the rotated frame U' x, whose last coordinates are flat
    const Tensor padded = m < n ? Tensor::concat({G, G.zeros_like({n, n - m})}, -1) : G;
    const auto svd = padded.svd();
    const Tensor &U = svd[0];
    const std::vector<double> s = svd[1].data();
    int64_t span = 0; // the singular values come in descending order
    while (span < n && s[span] > 2 * numerics::kDegenerateZonotope) ++span;
    if (span == 0) return aux_point(c);
    const Tensor Ut = U.matrix_transpose();
    const Tensor rows = Ut.slice(0, 0, span), flat = U.slice(1, span, n).matrix_transpose();
    const auto [Cs, ds] = aux_full_dim({rows.matmul(c), rows.matmul(G)});
    const Tensor shift = flat.matmul(c);
    return {Tensor::concat({Cs.matmul(rows), flat, flat.negative()}, 0),
            Tensor::concat({ds, shift, shift.negative()}, 0)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
