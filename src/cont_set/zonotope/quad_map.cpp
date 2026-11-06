// quad_map - an enclosure of a quadratic map of zonotopes, as CORA's zonotope.quadMap
//
// Output i is x' Q[i] x (or x1' Q[i] x2 for the zonotopes Z and Z2). For N generators the single
// map has the center x0' Q x0 plus the squares, the N squares halved and the N(N+1)/2 mixed terms
// as generators; the mixed map has the (N+1)(N2+1) - 1 products. With at most one nonzero Q the
// generators are one box generator, else those that are not zero.
//
// Syntax:   Zres = Z.quad_map(Q);   Zres = Z.quad_map(Z2, Q);
// Inputs:   Q - the matrices of the outputs, Q[i] (n, n); a CORA matZonotope entry is not supported
// Outputs:  Zres - a zonotope with one dimension per matrix that encloses the image
// See also: cubic_map, linear_map
//
// CORA: contSet/@zonotope/quadMap.m
// CORA authors: Matthias Althoff, Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   04-November-2026
// Reviewed:      ---

#include "cont_set/zonotope/private/priv.h"
#include "cora/cont_set/zonotope/zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Shapes and checks -------------------------------------------------------------------------------

/// [c G]: the center as the first column.
Tensor aux_full(const Zonotope &Z) { return Tensor::concat({Z.center(), Z.generators()}, -1); }

/// Whether any entry of the matrix is nonzero (CORA's any(Q{i}(:))).
bool aux_nonempty(const Tensor &Q) { return Q.abs().sum().item() != 0; }

// The rows of the outputs -------------------------------------------------------------------------

/// The mixed map from M = [c1 G1]' Q [c2 G2]: the entries in MATLAB's column-major order, the
/// first one is the center.
QuadRow aux_mixed_row(const Tensor &M) {
    const int64_t width = M.shape()[M.ndim() - 2] * M.shape().back();
    const Tensor all = M.matrix_transpose().reshape(priv_batch_shape(M, 1, width));
    return {all.slice(-1, 0, 1), all.slice(-1, 1, width)};
}

/// The zonotope of the outputs: row_of(Q[i]) for a nonzero Q[i], else a zero center and row.
template <class F>
Zonotope aux_outputs(const Tensor &like, int64_t width, const std::vector<Tensor> &Q, F row_of) {
    std::vector<Tensor> centers, rows;
    int nonempty = 0;
    for (const Tensor &M : Q) {
        const bool used = aux_nonempty(M);
        nonempty += used ? 1 : 0;
        QuadRow row = used ? row_of(M)
                           : QuadRow{like.zeros_like(priv_batch_shape(like, 1, 1)),
                                     like.zeros_like(priv_batch_shape(like, 1, width))};
        centers.push_back(row.first);
        rows.push_back(row.second);
    }
    const Tensor c = Tensor::concat(centers, -2), G = Tensor::concat(rows, -2);
    if (nonempty <= 1) return {c, G.abs_sum()};
    return Zonotope(c, G).compact("zeros", 0.0);
}

} // namespace


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::quad_map(const std::vector<Tensor> &Q) const {
    detail::Entry entry;
    const std::string problem = priv_quad_map_problem(Q, dim());
    CORA_CHECK(problem.empty(), Error::dimension_mismatch, [&] { return "quad_map: " + problem; });
    // M = [c G]' Q [c G] for every output; the squares and mixed terms are its entries
    const Tensor full = aux_full(*this), full_t = full.matrix_transpose();
    const int64_t N = G_.shape().back();
    return aux_outputs(full, N + N * (N + 1) / 2, Q, [&](const Tensor &Qi) {
        return priv_quad_map_single_row(full_t.matmul(Qi).matmul(full), N);
    });
}

Zonotope Zonotope::quad_map(const Zonotope &Z2, const std::vector<Tensor> &Q) const {
    detail::Entry entry;
    const std::string problem = priv_quad_map_problem(Q, dim());
    CORA_CHECK(problem.empty(), Error::dimension_mismatch, [&] { return "quad_map: " + problem; });
    CORA_CHECK(Z2.dim() == dim(), Error::dimension_mismatch, [&] {
        return "quad_map: the zonotopes differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(Z2.dim());
    });
    // M = [c G]' Q [c2 G2] for every output; its entries in column-major order are the products
    const Tensor full = aux_full(*this), full2 = aux_full(Z2), full_t = full.matrix_transpose();
    const int64_t width = (G_.shape().back() + 1) * (Z2.G_.shape().back() + 1) - 1;
    return aux_outputs(full, width, Q, [&](const Tensor &Qi) {
        return aux_mixed_row(full_t.matmul(Qi).matmul(full2));
    });
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
