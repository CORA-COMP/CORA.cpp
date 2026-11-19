// matmul - the product of a matrix zonotope with a matrix zonotope or a zonotope, as CORA's mtimes
//
// The inner dimensions agree and the batches are equal. P has the center C_A C_B and the
// generators G_A(i) C_B, C_A G_B(j) and G_A(i) G_B(j) (j the fastest index), in CORALean's order;
// CORA lists them in the order Z_A(i) Z_B(j), i fastest, for Z_A = [C_A, G_A], Z_B = [C_B, G_B]:
// the same set. Z2 has the generators [C_A G, G_i c, G_i g_j (i fastest)]. A 1 x 1 matrix
// zonotope as a scalar factor is refused (Error::not_supported). Matrices and numbers are
// linear_map, matmul_right and the scalar product.
//
// Syntax:   Zonotope P = A.matmul(B);   Zonotope Z2 = A.matmul(Z);   P = A * B;
// Inputs:   B - matrix zonotope (Zonotope::matrix)
//           Z - vector zonotope of as many dimensions as A has columns
// Outputs:  P - the enclosure of {M_A M_B}
//           Z2 - the zonotope with the center C_A c, an enclosure of A Z
// See also: linear_map, matmul_right, mat_exp::matrix_power
//
// CORA: matrixSet/@matZonotope/mtimes.m
// CORA authors: Matthias Althoff, Tobias Ladner
// Authors:       Claude Sonnet 5.5, Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cont_set/zonotope/private/priv_stacking.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

bool aux_scalar(const Zonotope &A) { return A.elem_shape() == Shape{1, 1}; }

/// A B of two matrix zonotopes.
Zonotope aux_of_matrices(const Tensor &C, const Tensor &G, const Tensor &BC, const Tensor &BG) {
    const int64_t both = G.size(-3) * BG.size(-3), rows = C.size(-2), cols = BC.size(-1);
    // G_A(i) C_B, then C_A G_B(j), then every G_A(i) G_B(j) with j fastest
    const Tensor on_center = G.matmul(BC.expand_dims(-3));
    const Tensor of_center = C.expand_dims(-3).matmul(BG);
    const Tensor crossed = G.expand_dims(-3).matmul(BG.expand_dims(-4));
    const Tensor cross = crossed.reshape(priv_replace_tail(crossed, 4, {both, rows, cols}));
    return Zonotope::matrix(C.matmul(BC), Tensor::concat({on_center, of_center, cross}, -3));
}

/// A z of a matrix zonotope and a vector zonotope (c, g).
Zonotope aux_of_vectors(const Tensor &C, const Tensor &G, const Tensor &c, const Tensor &g) {
    const int64_t h = G.size(-3), k = g.size(-2), rows = C.size(-2);
    const Tensor column = c.expand_dims(-1); // (..., n, 1)
    // G_i c as the rows (..., h, r)
    const Tensor on_center = G.matmul(column.expand_dims(-3)).squeeze(-1);
    // G_i g_j as the rows (..., k h, r), the generators of A varying fastest
    const Tensor each_g = g.expand_dims(-1).expand_dims(-3);
    const Tensor on_generators = G.expand_dims(-4).matmul(each_g).squeeze(-1);
    const Tensor mixed = on_generators.reshape(priv_replace_tail(on_generators, 3, {k * h, rows}));
    // C g as the rows g C' (the same products); an empty part is left out (not every backend)
    std::vector<Tensor> rows_of_g{g.matmul(C.matrix_transpose())};
    for (const Tensor &part : {on_center, mixed})
        if (part.size(-2) > 0) rows_of_g.push_back(part);
    return {C.matmul(column).squeeze(-1), Tensor::concat(rows_of_g, -2)};
}

} // namespace


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::matmul(const Zonotope &B) const {
    detail::Entry entry;
    CORA_CHECK(elem_rank_ == 2 && B.elem_rank_ <= 2, Error::not_supported, [] {
        return std::string("matmul: a matrix zonotope times a matrix or a vector zonotope");
    });
    // the inner dimensions, CORA's scalar expansion of a 1 x 1 factor, and the batches
    const int64_t inner = B.elem_rank_ == 2 ? B.c_.size(-2) : B.c_.size(-1);
    CORA_CHECK(c_.size(-1) == inner || aux_scalar(*this) || (B.elem_rank_ == 2 && aux_scalar(B)),
               Error::dimension_mismatch, [&] {
                   return "matmul: the inner dimensions differ, " + shape_text(c_.shape()) +
                          " times " + shape_text(B.c_.shape());
               });
    CORA_CHECK(c_.size(-1) == inner, Error::not_supported, [] {
        return std::string("matmul: a 1 x 1 matrix zonotope as a scalar factor is not supported");
    });
    CORA_CHECK(batch_shape() == B.batch_shape(), Error::dimension_mismatch, [&] {
        return "matmul: the batches differ, " + shape_text(c_.shape()) + " and " +
               shape_text(B.c_.shape());
    });
    return B.elem_rank_ == 2 ? aux_of_matrices(c_, G_, B.c_, B.G_)
                             : aux_of_vectors(c_, G_, B.c_, B.G_);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
