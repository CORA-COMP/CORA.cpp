// matmul - the product of a matrix zonotope with matrix zonotopes, zonotopes, matrices or numbers
//
// Syntax:   MatZonotope P = A.matmul(B);   P = A.matmul(M);   P = MatZonotope::matmul(M, A);
//           P = A.matmul(s);   Zonotope Z2 = A.matmul(Z);   P = A * B;
// Inputs:   B - matrix zonotope
//           M - matrix of numbers
//           s - number
//           Z - zonotope (columns, as
//           many dimensions as A has columns); the inner dimensions agree and the batches are equal
// Outputs:  P - for B the enclosure of {M_A M_B}: the center C_A C_B and the generators G_A(i) C_B,
//           C_A G_B(j) and G_A(i) G_B(j) (j the fastest index), in CORALean's order; CORA lists
//           them in the order Z_A(i) Z_B(j), i fastest, for Z_A = [C_A, G_A], Z_B = [C_B, G_B]:
//           the same set. For M and s the exact product; Z2 - the zonotope with the center
//           C_A c and the generators [C_A G, G_i c, G_i g_j (i fastest)], an enclosure of A Z.
//           A 1 x 1 matrix zonotope as a scalar factor is refused (Error::not_supported)
// See also: plus, matrix_power
//
// CORA: matrixSet/@matZonotope/mtimes.m
// CORA authors: Matthias Althoff, Tobias Ladner
// Authors:       Claude Sonnet 5.5, Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   04-November-2026
// Reviewed:      ---

#include "cora/matrix_set/mat_zonotope/mat_zonotope.h"
#include "global/format.h"
#include "matrix_set/mat_zonotope/private/priv_stacking.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

bool aux_scalar(const MatZonotope &A) { return A.rows() == 1 && A.cols() == 1; }

} // namespace


// ===========================================  MAIN  =========================================== //

// Two matrix zonotopes ----------------------------------------------------------------------------

MatZonotope MatZonotope::matmul(const MatZonotope &B) const {
    detail::Entry entry;
    // the inner dimensions, CORA's scalar expansion of a 1 x 1 factor, and the batches
    CORA_CHECK(cols() == B.rows() || aux_scalar(*this) || aux_scalar(B), Error::dimension_mismatch,
               [&] {
                   return "matmul: the inner dimensions differ, " + shape_text(C_.shape()) +
                          " times " + shape_text(B.C_.shape());
               });
    CORA_CHECK(cols() == B.rows(), Error::not_supported, [] {
        return std::string("matmul: a 1 x 1 matrix zonotope as a scalar factor is not supported");
    });
    CORA_CHECK(priv_replace_tail(C_, 2, {}) == priv_replace_tail(B.C_, 2, {}),
               Error::dimension_mismatch, [&] {
                   return "matmul: the batches differ, " + shape_text(C_.shape()) + " and " +
                          shape_text(B.C_.shape());
               });
    const int64_t both = num_generators() * B.num_generators();
    // G_A(i) C_B, then C_A G_B(j), then every G_A(i) G_B(j) with j fastest
    const Tensor on_center = G_.matmul(B.C_.expand_dims(-3));
    const Tensor of_center = C_.expand_dims(-3).matmul(B.G_);
    const Tensor crossed = G_.expand_dims(-3).matmul(B.G_.expand_dims(-4));
    const Tensor cross = crossed.reshape(priv_replace_tail(crossed, 4, {both, rows(), B.cols()}));
    return MatZonotope(C_.matmul(B.C_), Tensor::concat({on_center, of_center, cross}, -3));
}


// Matrices and numbers ----------------------------------------------------------------------------

MatZonotope MatZonotope::matmul(const Tensor &M) const {
    detail::Entry entry;
    // a matrix on the right multiplies the center and every generator
    CORA_CHECK(M.ndim() >= 2 && M.size(-2) == cols(), Error::dimension_mismatch, [&] {
        return "matmul: the matrix needs " + std::to_string(cols()) + " rows, got the shape " +
               shape_text(M.shape());
    });
    return MatZonotope(C_.matmul(M), G_.matmul(M.expand_dims(-3)));
}

MatZonotope MatZonotope::matmul(const Tensor &M, const MatZonotope &A) {
    detail::Entry entry;
    CORA_CHECK(M.ndim() >= 2 && M.size(-1) == A.rows(), Error::dimension_mismatch, [&] {
        return "matmul: the matrix needs " + std::to_string(A.rows()) + " columns, got the shape " +
               shape_text(M.shape());
    });
    return MatZonotope(M.matmul(A.C_), M.expand_dims(-3).matmul(A.G_));
}

MatZonotope MatZonotope::matmul(double s) const {
    detail::Entry entry;
    return MatZonotope(C_ * s, G_ * s);
}


// Zonotopes ---------------------------------------------------------------------------------------

// the zonotope that encloses {M z}: the generators of the product are the three kinds of CORA
Zonotope MatZonotope::matmul(const Zonotope &Z) const {
    detail::Entry entry;
    const Tensor &c = Z.center(), &g = Z.generators();
    // the dimension of the zonotope, the scalar expansion of a 1 x 1 set, and the batches
    CORA_CHECK(cols() == c.size(-2) || aux_scalar(*this), Error::dimension_mismatch, [&] {
        return "matmul: the zonotope needs " + std::to_string(cols()) + " dimensions, got " +
               std::to_string(c.size(-2));
    });
    CORA_CHECK(cols() == c.size(-2), Error::not_supported, [] {
        return std::string("matmul: a 1 x 1 matrix zonotope as a scalar factor is not supported");
    });
    CORA_CHECK(priv_replace_tail(C_, 2, {}) == priv_replace_tail(c, 2, {}),
               Error::dimension_mismatch, [&] {
                   return "matmul: the batches differ, " + shape_text(C_.shape()) + " and " +
                          shape_text(c.shape());
               });
    const int64_t h = num_generators(), k = g.size(-1);
    // G_i c as the columns (..., n, h)
    const Tensor on_center = G_.matmul(c.expand_dims(-3)).squeeze(-1).matrix_transpose();
    // G_i g_j as the columns (..., n, k h), the generators of A varying fastest
    const Tensor each_g = g.matrix_transpose().expand_dims(-1).expand_dims(-3);
    const Tensor on_generators = G_.expand_dims(-4).matmul(each_g).squeeze(-1);
    const Tensor mixed = on_generators.reshape(priv_replace_tail(on_generators, 3, {k * h, rows()}))
                             .matrix_transpose();
    // a part without columns is left out: not every backend joins empty tensors
    std::vector<Tensor> columns_of_g{C_.matmul(g)};
    for (const Tensor &part : {on_center, mixed})
        if (part.size(-1) > 0) columns_of_g.push_back(part);
    return Zonotope(C_.matmul(c), Tensor::concat(columns_of_g, -1));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
