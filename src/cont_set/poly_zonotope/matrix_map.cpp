// matrix_map - the product of two matrix-valued polynomial zonotopes, as CORA's matMap
//
// Both sets hold a matrix in their coordinates, column by column (n x k and k x m). With
// [c, G] the center and dependent generators, the product has one generator per pair of them (the
// first set's the slower index), the matrix product of the two, with the sum of the exponents:
// exact over the shared factors. The center is 0 (the pair of the centers is a constant
// generator). The independent generators of the result are the pairs GI1 x [c2, G2], GI1 x GI2 and
// [c1, G1] x GI2, in that order, as CORALean's matMap; products of two independent factors are
// new independent factors, so the result encloses. CORA orders the blocks differently and moves
// the constant to the center (docs/cora-differences.md).
//
// Syntax:   S = pZ1.matrix_map(pZ2, n, k, m);
// Inputs:   pZ2 - polynomial zonotope of the dimension k * m
//           n, k, m - the sizes of the matrices: pZ1 has the dimension n * k
// Outputs:  S - polynomial zonotope of the dimension n * m
// See also: linear_map, quad_map
//
// CORA: contSet/@polyZonotope/matMap.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The products of every row of A (a, n k) with every row of B (b, k m), as the matrices column by
/// column: one row (n m) per pair, the row of A the slower index.
Tensor aux_products(const Tensor &A, const Tensor &B, int64_t n, int64_t k, int64_t m) {
    const int64_t a = A.shape()[0], b = B.shape()[0];
    // a matrix column by column is its transpose row by row: C' = B' A'
    const Tensor At = A.reshape({a, 1, k, n}), Bt = B.reshape({1, b, m, k});
    return Bt.matmul(At).reshape({a * b, n * m});
}

} // namespace


// ===========================================  MAIN  =========================================== //

PolyZonotope PolyZonotope::matrix_map(const PolyZonotope &S, int64_t n, int64_t k,
                                      int64_t m) const {
    detail::Entry entry;
    CORA_CHECK(n >= 0 && k >= 0 && m >= 0 && dim() == n * k, Error::wrong_value, [&] {
        return "matrix_map: the dimension of the first set must be n * k = " +
               std::to_string(n * k) + ", it is " + std::to_string(dim());
    });
    CORA_CHECK(S.dim() == k * m, Error::wrong_value, [&] {
        return "matrix_map: the dimension of the second set must be k * m = " +
               std::to_string(k * m) + ", it is " + std::to_string(S.dim());
    });
    // both exponent matrices over the union of the identifiers
    const MergedIds ids = priv_merge_ids(id_, S.id_);
    const Exponents E1 = priv_extended_exponents(
        priv_extend_exponents(E_, ids.position1, ids.id.size()), ids.id.size());
    const Exponents E2 = priv_extended_exponents(
        priv_extend_exponents(S.E_, ids.position2, ids.id.size()), ids.id.size());
    Exponents E;
    E.reserve(E1.size() * E2.size());
    for (const std::vector<int> &x : E1)
        for (const std::vector<int> &y : E2) {
            std::vector<int> e = x;
            for (std::size_t r = 0; r < e.size(); ++r) e[r] += y[r];
            E.push_back(std::move(e));
        }
    // the pairs of generators, then the three blocks of independent pairs
    const Tensor ext1 = Tensor::concat({c_.expand_dims(0), G_}, -2);
    const Tensor ext2 = Tensor::concat({S.c_.expand_dims(0), S.G_}, -2);
    const Tensor GI =
        Tensor::concat({aux_products(GI_, ext2, n, k, m), aux_products(GI_, S.GI_, n, k, m),
                        aux_products(ext1, S.GI_, n, k, m)},
                       -2);
    return {c_.zeros_like({n * m}), aux_products(ext1, ext2, n, k, m), GI, std::move(E), ids.id,
            detail::Verbatim{}};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
