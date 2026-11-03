// quad_map - an enclosure of a quadratic map of zonotopes, as CORA's zonotope.quadMap
//
// Syntax:   Zres = Z.quad_map(Q);   Zres = Z.quad_map(Z2, Q);
// Inputs:   Q - the matrices of the outputs, Q[i] (n, n): output i is x' Q[i] x (or x1' Q[i] x2 for
//           the zonotopes Z and Z2); a CORA matZonotope entry is not supported
// Outputs:  Zres - a zonotope with one dimension per matrix that encloses the image. For N
//           generators the single map has the center x0' Q x0 plus the squares, the N squares
//           halved and the N(N+1)/2 mixed terms as generators; the mixed map has the
//           (N+1)(N2+1) - 1 products. With at most one nonzero Q the generators are one box
//           generator, else those that are not zero
// See also: cubic_map, linear_map
//
// CORA: contSet/@zonotope/quadMap.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Shapes and checks -------------------------------------------------------------------------------

/// [c G]: the center as the first column.
Tensor aux_full(const Zonotope &Z) { return Tensor::concat({Z.center(), Z.generators()}, -1); }

/// The shape of the batch of `like` followed by (rows, cols).
Shape aux_shape(const Tensor &like, int64_t rows, int64_t cols) {
    Shape shape;
    for (int64_t i = 0; i + 2 < like.ndim(); ++i) shape.push_back(like.shape()[i]);
    shape.push_back(rows);
    shape.push_back(cols);
    return shape;
}

/// Whether any entry of the matrix is nonzero (CORA's any(Q{i}(:))).
bool aux_nonempty(const Tensor &Q) { return Q.abs().sum().item() != 0; }

/// What is wrong with Q for sets of dimension n, or an empty text if nothing.
std::string aux_problem(const std::vector<Tensor> &Q, int64_t n) {
    if (Q.empty()) return "Q needs at least one matrix";
    for (const Tensor &M : Q)
        if (M.shape() != Shape{n, n})
            return "every matrix of Q must be (" + std::to_string(n) + ", " + std::to_string(n) +
                   "), got the shape " + shape_text(M.shape());
    return "";
}

// The rows of the outputs -------------------------------------------------------------------------

/// One output: its center (..., 1, 1) and a row of generators (..., 1, width).
using Row = std::pair<Tensor, Tensor>;

/// The single map from M = [c G]' Q [c G] (..., N+1, N+1): the center is M(0,0) plus the halved
/// squares, which are the first N generators, then the strictly lower triangle of M + M' in the
/// order of MATLAB's column-major vector.
Row aux_single_row(const Tensor &M, int64_t N) {
    const Tensor squares = M.slice(-2, 1, N + 1).slice(-1, 1, N + 1).diagonal() * 0.5;
    const Tensor center = M.slice(-2, 0, 1).slice(-1, 0, 1) + squares.sum(-2, true);
    std::vector<int64_t> lower;
    for (int64_t j = 0; j <= N; ++j)
        for (int64_t r = j + 1; r <= N; ++r) lower.push_back(r * (N + 1) + j);
    const Tensor mixed = lower.empty() ? M.zeros_like(aux_shape(M, 1, 0))
                                       : (M + M.matrix_transpose())
                                             .reshape(aux_shape(M, 1, (N + 1) * (N + 1)))
                                             .take(lower, -1);
    return {center, Tensor::concat({squares.matrix_transpose(), mixed}, -1)};
}

/// The mixed map from M = [c1 G1]' Q [c2 G2]: the entries in MATLAB's column-major order, the
/// first one is the center.
Row aux_mixed_row(const Tensor &M) {
    const int64_t width = M.shape()[M.ndim() - 2] * M.shape().back();
    const Tensor all = M.matrix_transpose().reshape(aux_shape(M, 1, width));
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
        Row row = used ? row_of(M)
                       : Row{like.zeros_like(aux_shape(like, 1, 1)),
                             like.zeros_like(aux_shape(like, 1, width))};
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
    const std::string problem = aux_problem(Q, dim());
    CORA_CHECK(problem.empty(), Error::dimension_mismatch, [&] { return "quad_map: " + problem; });
    // M = [c G]' Q [c G] for every output; the squares and mixed terms are its entries
    const Tensor full = aux_full(*this), full_t = full.matrix_transpose();
    const int64_t N = G_.shape().back();
    return aux_outputs(full, N + N * (N + 1) / 2, Q, [&](const Tensor &Qi) {
        return aux_single_row(full_t.matmul(Qi).matmul(full), N);
    });
}

Zonotope Zonotope::quad_map(const Zonotope &Z2, const std::vector<Tensor> &Q) const {
    detail::Entry entry;
    const std::string problem = aux_problem(Q, dim());
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
