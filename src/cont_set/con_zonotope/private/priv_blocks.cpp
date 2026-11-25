// priv_blocks - the exact set maps that the operations of the class share
//
// Syntax:   ConZonotope c = priv_compact(cz);   ConZonotope s = priv_minkowski_sum(cz1, cz2);
// Inputs:   czs - constrained zonotopes, or their tensors, of one batch shape
// Outputs:  R - the compacted set, the sum or the intersection
// See also: con_zonotope.h, global/blocks.h
//
// CORA.cpp only: MATLAB writes blkdiag and horizontal concatenation inline
// Authors:       Claude Sonnet 5.5, Tobias Ladner, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   25-November-2026
// Reviewed:      ---

#include "cont_set/con_zonotope/private/priv.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "global/blocks.h"
#include "global/format.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The positions in [0, count) where `keep` (a tensor with `count` entries) is nonzero.
std::vector<int64_t> aux_positions(const Tensor &keep, int64_t count) {
    Tensor flat = keep.detach();
    while (flat.ndim() > 1) flat = flat.any(0); // of a batch, what some member keeps
    return flat.reshape({count}).nonzero();
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Pairs and corners -------------------------------------------------------------------------------

detail::Refusal priv_pair_refusal(const char *operation, const ConZonotope &a, const ConZonotope &b,
                                  bool same_dim) {
    const std::string name = std::string(operation) + ": ";
    // the leading axes are the batch
    const Shape shape_a = a.c().shape(), shape_b = b.c().shape();
    if (!std::equal(shape_a.begin(), shape_a.end() - 1, shape_b.begin(), shape_b.end() - 1))
        return std::pair{Error::dimension_mismatch,
                         name + "the sets must have the same batch shape, got " +
                             shape_text(shape_a) + " and " + shape_text(shape_b)};
    if (same_dim && a.dim() != b.dim())
        return std::pair{Error::dimension_mismatch, name + "the sets differ in dimension, " +
                                                        std::to_string(a.dim()) + " and " +
                                                        std::to_string(b.dim())};
    return std::nullopt;
}

// Compacting --------------------------------------------------------------------------------------

std::vector<int64_t> priv_nonzero_rows(const Tensor &A, const Tensor &b) {
    // a row is kept if [A b] has a nonzero entry
    const Tensor rows = Tensor::concat({A, b}, -1);
    return aux_positions(rows.not_equal(rows.zeros_like()).any(-1), A.shape()[A.ndim() - 2]);
}

std::pair<std::vector<int64_t>, std::vector<int64_t>> priv_nonzero_parts(const ConZonotope &cz) {
    // a column is kept if [G; A] has a nonzero entry
    const Tensor cols = Tensor::concat({cz.generators().matrix_transpose(), cz.A()}, -2);
    const Tensor col_used = cols.not_equal(cols.zeros_like()).any(-2);
    return {priv_nonzero_rows(cz.A(), priv_b_column(cz)),
            aux_positions(col_used, cz.A().shape().back())};
}

ConZonotope priv_compact(const ConZonotope &cz) {
    const auto [rows, cols] = priv_nonzero_parts(cz);
    // the constraint rows 0 = 0 go, an infeasible zero row 0 = b stays
    return {cz.c(), (select_columns(cz.generators().matrix_transpose(), cols)).matrix_transpose(),
            cz.factor_polytope().select_cols(cols).select_rows({}, rows)};
}

// Set maps ----------------------------------------------------------------------------------------

ConZonotope priv_affine_image(const ConZonotope &cz, const Tensor &M,
                              const std::optional<Tensor> &t) {
    auto [c, G] = Tensor::zonotope_linear_map(M, cz.c(), cz.generators());
    if (t) c = c + *t; // the offset is a point (..., n)
    return {c, G, cz.factor_polytope()};
}

Polytope priv_factor_product(const ConZonotope &cz1, const ConZonotope &cz2) {
    return cz1.factor_polytope()
        .pad_left(cz2.nr_generators())
        .stack_rows(cz2.factor_polytope().pad_right(cz1.nr_generators()));
}

ConZonotope priv_minkowski_sum(const ConZonotope &cz1, const ConZonotope &cz2) {
    return {cz1.c() + cz2.c(), Tensor::concat({cz1.generators(), cz2.generators()}, -2),
            priv_factor_product(cz1, cz2)};
}

ConZonotope priv_agreement(const ConZonotope &cz1, const ConZonotope &cz2, const Tensor &Mc,
                           const Tensor &MG) {
    const int64_t n = cz1.dim(), m2 = cz2.nr_generators();
    // the factors of both sets, the first set's generators carry the result
    const Tensor G = Tensor::concat({cz1.generators(), zeros_block(cz1.generators(), m2, n)}, -2);
    // a row per dimension makes the images agree; 0 - G keeps zeros +0, as the oracle reads them
    const Tensor negated = cz2.generators().zeros_like() - cz2.generators();
    const Tensor agree = Tensor::concat({MG, negated}, -2).matrix_transpose();
    return {cz1.c(), G,
            priv_factor_product(cz1, cz2).stack_rows(Polytope::equalities(agree, cz2.c() - Mc))};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
