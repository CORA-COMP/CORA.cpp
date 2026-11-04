// quad_map - an enclosure of a quadratic map of constrained zonotopes, as CORA's quadMap
//
// Syntax:   cZres = cz.quad_map(Q);   cZres = cz.quad_map(cz2, Q);
// Inputs:   Q - the matrices of the outputs, Q[i] (n, n): output i is x' Q[i] x (or x1' Q[i] x2
//           for the sets cz and cz2); a set with constraints is rescaled first ("iter")
// Outputs:  cZres - a constrained zonotope with one dimension per matrix that encloses the image.
//           For N factors the single map has the N squares, the N linear terms and the N(N-1)/2
//           mixed terms as generators; the constraints stay on the linear block and the other
//           factors are free. The mixed map has the factors of cz2, of cz and their (N+1)(N2+1) -
//           N2 - N - 1 products, the constraints of both on their own factors
// See also: cubic_map, rescale, Zonotope::quad_map
//
// CORA: contSet/@conZonotope/quadMap.m

#include "cont_set/con_zonotope/private/priv.h"
#include "cont_set/zonotope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The set rescaled to its factor ranges if it has constraints (CORA's rescale(cZ, 'iter')).
ConZonotope aux_rescaled(const ConZonotope &cz) {
    return cz.A().numel() == 0 ? cz : cz.rescale("iter");
}

/// [c G]: the center as the first column.
Tensor aux_full(const ConZonotope &cz) { return Tensor::concat({cz.c(), cz.generators()}, -1); }

/// The output of the mixed map from M = [c1 G1]' Q [c2 G2]: the center M(0, 0), then the first
/// row, the first column and the rest in MATLAB's column-major order.
QuadRow aux_mixed_row(const Tensor &M) {
    const int64_t rows = M.shape()[M.ndim() - 2], cols = M.shape().back();
    const Tensor products = M.slice(-2, 1, rows).slice(-1, 1, cols);
    const Tensor first_column = M.slice(-1, 0, 1).slice(-2, 1, rows).matrix_transpose();
    return {M.slice(-2, 0, 1).slice(-1, 0, 1),
            Tensor::concat({M.slice(-2, 0, 1).slice(-1, 1, cols), first_column,
                            products.matrix_transpose().reshape(
                                priv_batch_shape(M, 1, (rows - 1) * (cols - 1)))},
                           -1)};
}

/// The set of the outputs: the centers and rows of generators stacked, the constraints A (p, W).
ConZonotope aux_outputs(const std::vector<QuadRow> &rows, const Tensor &A, const Tensor &b) {
    std::vector<Tensor> centers, generators;
    for (const QuadRow &row : rows) {
        centers.push_back(row.first);
        generators.push_back(row.second);
    }
    return {Tensor::concat(centers, -2), Tensor::concat(generators, -2), A, b};
}

} // namespace


// ===========================================  MAIN  =========================================== //

ConZonotope ConZonotope::quad_map(const std::vector<Tensor> &Q) const {
    detail::Entry entry;
    const std::string problem = priv_quad_map_problem(Q, dim());
    CORA_CHECK(problem.empty(), Error::dimension_mismatch, [&] { return "quad_map: " + problem; });
    const ConZonotope cz = aux_rescaled(*this);
    // M = [c G]' Q [c G] for every output; the squares and mixed terms are its entries
    const Tensor full = aux_full(cz), full_t = full.matrix_transpose();
    const int64_t N = cz.generators().shape().back(), p = cz.A().shape()[cz.A().ndim() - 2];
    std::vector<QuadRow> rows;
    rows.reserve(Q.size());
    for (const Tensor &Qi : Q)
        rows.push_back(priv_quad_map_single_row(full_t.matmul(Qi).matmul(full), N));
    // the constraints sit on the linear block, between the N squares and the N(N-1)/2 products
    const Tensor A = Tensor::concat(
        {priv_zeros(cz.A(), p, N), cz.A(), priv_zeros(cz.A(), p, N * (N - 1) / 2)}, -1);
    return aux_outputs(rows, A, cz.b());
}

ConZonotope ConZonotope::quad_map(const ConZonotope &cz2, const std::vector<Tensor> &Q) const {
    detail::Entry entry;
    const std::string problem = priv_quad_map_problem(Q, dim());
    CORA_CHECK(problem.empty(), Error::dimension_mismatch, [&] { return "quad_map: " + problem; });
    const detail::Refusal refusal = priv_pair_refusal("quad_map", *this, cz2, true);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    const ConZonotope first = aux_rescaled(*this), second = aux_rescaled(cz2);
    // M = [c1 G1]' Q [c2 G2] for every output
    const Tensor full = aux_full(first), full2 = aux_full(second), full_t = full.matrix_transpose();
    const int64_t N1 = first.generators().shape().back(), N2 = second.generators().shape().back();
    const int64_t p1 = first.A().shape()[first.A().ndim() - 2];
    const int64_t p2 = second.A().shape()[second.A().ndim() - 2];
    std::vector<QuadRow> rows;
    rows.reserve(Q.size());
    for (const Tensor &Qi : Q) rows.push_back(aux_mixed_row(full_t.matmul(Qi).matmul(full2)));
    // the constraints of cz2 sit on the first N2 factors and those of cz on the next N1
    const int64_t width = (N1 + 1) * (N2 + 1) - 1;
    const Tensor A =
        Tensor::concat({Tensor::concat({second.A(), priv_zeros(second.A(), p2, width - N2)}, -1),
                        Tensor::concat({priv_zeros(first.A(), p1, N2), first.A(),
                                        priv_zeros(first.A(), p1, width - N2 - N1)},
                                       -1)},
                       -2);
    return aux_outputs(rows, A, Tensor::concat({second.b(), first.b()}, -2));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
