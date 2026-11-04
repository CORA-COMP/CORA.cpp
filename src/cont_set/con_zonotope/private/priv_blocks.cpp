// priv_blocks - the block matrices and the exact set maps that the operations of the class share
//
// Syntax:   Tensor Z = priv_zeros(like, rows, cols);   Tensor B = priv_block_diag(A, B);
//           ConZonotope c = priv_compact(cz);   ConZonotope s = priv_minkowski_sum(cz1, cz2);
// Inputs:   tensors of one batch shape, constrained zonotopes
// Outputs:  the block matrix, the compacted set, the sum, the intersection with agreement rows
// See also: con_zonotope.h
//
// CORA.cpp only: MATLAB writes blkdiag and horizontal concatenation inline

#include "cont_set/con_zonotope/private/priv.h"

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

Tensor priv_zeros(const Tensor &like, int64_t rows, int64_t cols) {
    Shape shape = like.shape();
    shape[shape.size() - 2] = rows;
    shape.back() = cols;
    return like.zeros_like(shape);
}

Tensor priv_full(const Tensor &like, int64_t rows, int64_t cols, double value) {
    Shape shape = like.shape();
    shape[shape.size() - 2] = rows;
    shape.back() = cols;
    return like.full_like(shape, value);
}

Tensor priv_identity(const Tensor &like, int64_t k) {
    Shape shape = like.shape();
    shape[shape.size() - 2] = k;
    shape.back() = k;
    return like.eye_like(k).broadcast_to(shape);
}

Tensor priv_block_diag(const Tensor &A, const Tensor &B) {
    const int64_t rows_a = A.shape()[A.ndim() - 2], rows_b = B.shape()[B.ndim() - 2];
    const int64_t cols_a = A.shape().back(), cols_b = B.shape().back();
    // A with zeros to its right, zeros to the left of B
    const Tensor top = Tensor::concat({A, priv_zeros(A, rows_a, cols_b)}, -1);
    const Tensor bottom = Tensor::concat({priv_zeros(B, rows_b, cols_a), B}, -1);
    return Tensor::concat({top, bottom}, -2);
}

Tensor priv_select_columns(const Tensor &A, const std::vector<int64_t> &columns) {
    if (columns.empty()) return priv_zeros(A, A.shape()[A.ndim() - 2], 0);
    return A.take(columns, -1);
}

std::vector<int64_t> priv_nonzero_rows(const Tensor &A, const Tensor &b) {
    // a row is kept if [A b] has a nonzero entry
    const Tensor rows = Tensor::concat({A, b}, -1);
    return aux_positions(rows.not_equal(rows.zeros_like()).any(-1), A.shape()[A.ndim() - 2]);
}

std::pair<std::vector<int64_t>, std::vector<int64_t>> priv_nonzero_parts(const ConZonotope &cz) {
    // a column is kept if [G; A] has a nonzero entry
    const Tensor cols = Tensor::concat({cz.generators(), cz.A()}, -2);
    const Tensor col_used = cols.not_equal(cols.zeros_like()).any(-2);
    return {priv_nonzero_rows(cz.A(), cz.b()), aux_positions(col_used, cz.A().shape().back())};
}

ConZonotope priv_compact(const ConZonotope &cz) {
    const auto [rows, cols] = priv_nonzero_parts(cz);
    const Tensor A = priv_select_columns(cz.A(), cols);
    // the constraint rows 0 = 0 go, an infeasible zero row 0 = b stays
    const Tensor kept_A = rows.empty() ? priv_zeros(A, 0, A.shape().back()) : A.take(rows, -2);
    const Tensor kept_b = rows.empty() ? priv_zeros(cz.b(), 0, 1) : cz.b().take(rows, -2);
    return {cz.c(), priv_select_columns(cz.generators(), cols), kept_A, kept_b};
}

ConZonotope priv_affine_image(const ConZonotope &cz, const Tensor &M,
                              const std::optional<Tensor> &t) {
    auto [c, G] = Tensor::zonotope_linear_map(M, cz.c(), cz.generators());
    if (t) c = c + *t;
    return {c, G, cz.A(), cz.b()};
}

ConZonotope priv_minkowski_sum(const ConZonotope &cz1, const ConZonotope &cz2) {
    return {cz1.c() + cz2.c(), Tensor::concat({cz1.generators(), cz2.generators()}, -1),
            priv_block_diag(cz1.A(), cz2.A()), Tensor::concat({cz1.b(), cz2.b()}, -2)};
}

ConZonotope priv_agreement(const ConZonotope &cz1, const ConZonotope &cz2, const Tensor &Mc,
                           const Tensor &MG) {
    const int64_t n = cz1.dim(), m2 = cz2.generators().shape().back();
    // the factors of both sets, the first set's generators carry the result
    const Tensor G = Tensor::concat({cz1.generators(), priv_zeros(cz1.generators(), n, m2)}, -1);
    // each set keeps its constraints; the images of the generators must agree
    const Tensor rows = Tensor::concat(
        {priv_block_diag(cz1.A(), cz2.A()), Tensor::concat({MG, -cz2.generators()}, -1)}, -2);
    const Tensor b = Tensor::concat({cz1.b(), cz2.b(), cz2.c() - Mc}, -2);
    return {cz1.c(), G, rows, b};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
