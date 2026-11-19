// cart_prod - the Cartesian product of a zonotope with a zonotope, a box or a point
//
// The centers stack and the generators sit on the block diagonal; a point adds zero rows. As the
// first operand the point p is (n1, 1).
//
// Syntax:   Zres = Z.cart_prod(Z2);   Zres = Z.cart_prod(I);   Zres = Zonotope::cart_prod(p, Z);
// Inputs:   Z2, I, p - a zonotope, a box or a point (n2, 1), of any dimension
// Outputs:  Zres - exact, of dimension n + n2 in the order of the operands
// See also: plus, ContSet cart_prod
//
// CORA: contSet/@zonotope/cartProd_.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/zonotope/private/priv.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Zeros of the batch shape of `like` with `rows` rows and `cols` columns.
Tensor aux_zeros(const Tensor &like, int64_t rows, int64_t cols) {
    Shape shape = like.shape();
    shape[shape.size() - 2] = rows;
    shape.back() = cols;
    return like.zeros_like(shape);
}

/// The leading (batch) axes of a tensor whose last `elem` axes are not.
std::vector<int64_t> aux_batch(const Tensor &t, int elem) {
    const std::vector<int64_t> shape = t.shape();
    return {shape.begin(), shape.end() - elem};
}

/// The product of two zonotopes: centers join, generators on the block diagonal.
Zonotope aux_blocks(const Zonotope &Z1, const Zonotope &Z2) {
    const Tensor &G1 = Z1.generators(), &G2 = Z2.generators();
    const int64_t n1 = Z1.dim(), n2 = Z2.dim(), m1 = G1.size(-2), m2 = G2.size(-2);
    // G1 with zeros to its right, zeros with G2 to their right, one above the other
    const Tensor top = Tensor::concat({G1, aux_zeros(G1, m1, n2)}, -1);
    const Tensor bottom = Tensor::concat({aux_zeros(G2, m2, n1), G2}, -1);
    return {Tensor::concat({Z1.center(), Z2.center()}, -1), Tensor::concat({top, bottom}, -2)};
}

} // namespace


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::cart_prod(const Zonotope &Z2) const {
    detail::Entry entry;
    CORA_CHECK(elem_rank_ == 1 && Z2.elem_rank() == 1 &&
                   aux_batch(c_, 1) == aux_batch(Z2.center(), 1),
               Error::dimension_mismatch, [&] {
                   return "cart_prod: the zonotopes must have the same batch shape, got " +
                          shape_text(c_.shape()) + " and " + shape_text(Z2.center().shape());
               });
    return aux_blocks(*this, Z2);
}

Zonotope Zonotope::cart_prod(const Interval &I) const {
    detail::Entry entry;
    // a box of column bounds, as its zonotope
    CORA_CHECK(elem_rank_ == 1 && I.elem_rank() == 1 && aux_batch(c_, 1) == aux_batch(I.inf(), 1),
               Error::dimension_mismatch, [&] {
                   return "cart_prod: the interval must be a box of the batch shape of the "
                          "zonotope, got the shape " +
                          shape_text(I.inf().shape());
               });
    return aux_blocks(*this, I.zonotope());
}

Zonotope Zonotope::cart_prod(const Tensor &p) const {
    // the product with a point
    detail::Entry entry;
    // a point is a column; the product adds its rows
    CORA_CHECK(elem_rank_ == 1 && p.ndim() >= 2 && p.shape().back() == 1, Error::dimension_mismatch,
               [&] {
                   return "cart_prod: the point must be a column (n, 1), got the shape " +
                          shape_text(p.shape());
               });
    const int64_t n2 = p.shape()[p.ndim() - 2];
    // the point has no generators: its entries of every generator are zero
    return {Tensor::concat({c_, p.squeeze(-1)}, -1),
            Tensor::concat({G_, aux_zeros(G_, G_.size(-2), n2)}, -1)};
}

Zonotope Zonotope::cart_prod(const Tensor &p, const Zonotope &Z) {
    detail::Entry entry;
    CORA_CHECK(Z.elem_rank() == 1 && p.ndim() >= 2 && p.shape().back() == 1,
               Error::dimension_mismatch, [&] {
                   return "cart_prod: the point must be a column (n, 1), got the shape " +
                          shape_text(p.shape());
               });
    const int64_t n1 = p.shape()[p.ndim() - 2];
    // the point has no generators: its entries of every generator are zero
    return {Tensor::concat({p.squeeze(-1), Z.center()}, -1),
            Tensor::concat({aux_zeros(Z.generators(), Z.nr_generators(), n1), Z.generators()}, -1)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
