// cart_prod - the Cartesian product of a zonotope with a zonotope, a box or a point
//
// Syntax:   Zres = Z.cart_prod(Z2);   Zres = Z.cart_prod(I);   Zres = Zonotope::cart_prod(p, Z);
// Inputs:   Z2, I, p - a zonotope, a box or a point (n2, 1), of any dimension; as the first
//           operand, the point p (n1, 1)
// Outputs:  Zres - exact: the centers stack, the generators sit on the block diagonal (a point
//           adds zero rows); of dimension n + n2, in the order of the operands
// See also: plus, ContSet cart_prod
//
// CORA: contSet/@zonotope/cartProd_.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "global/format.h"
#include "cont_set/zonotope/private/priv.h"

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

/// The leading (batch) axes of a tensor.
std::vector<int64_t> aux_batch(const Tensor &t) {
    const std::vector<int64_t> shape = t.shape();
    return {shape.begin(), shape.end() - 2};
}

/// The product of two zonotopes: centers stack, generators on the block diagonal.
Zonotope aux_blocks(const Zonotope &Z1, const Zonotope &Z2) {
    const Tensor &G1 = Z1.generators(), &G2 = Z2.generators();
    const int64_t n1 = Z1.dim(), n2 = Z2.dim(), m1 = G1.shape().back(), m2 = G2.shape().back();
    // G1 with zeros to its right, zeros with G2 below
    const Tensor top = Tensor::concat({G1, aux_zeros(G1, n1, m2)}, -1);
    const Tensor bottom = Tensor::concat({aux_zeros(G2, n2, m1), G2}, -1);
    return {Tensor::concat({Z1.center(), Z2.center()}, -2), Tensor::concat({top, bottom}, -2)};
}

} // namespace


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::cart_prod(const Zonotope &Z2) const {
    detail::Entry entry;
    CORA_CHECK(aux_batch(c_) == aux_batch(Z2.center()), Error::dimension_mismatch, [&] {
        return "cart_prod: the zonotopes must have the same batch shape, got " +
               shape_text(c_.shape()) + " and " + shape_text(Z2.center().shape());
    });
    return aux_blocks(*this, Z2);
}

Zonotope Zonotope::cart_prod(const Interval &I) const {
    detail::Entry entry;
    CORA_CHECK(I.inf().shape().back() == 1 && aux_batch(c_) == aux_batch(I.inf()),
               Error::dimension_mismatch, [&] {
        return "cart_prod: the interval must be a box with column bounds and the batch shape of "
               "the zonotope, got the shape " + shape_text(I.inf().shape());
    });
    return aux_blocks(*this, priv_zonotope_of_interval(I));
}

Zonotope Zonotope::cart_prod(const Tensor &p) const {
    detail::Entry entry;
    // a point is a column; the product adds its rows
    CORA_CHECK(p.ndim() >= 2 && p.shape().back() == 1, Error::dimension_mismatch, [&] {
        return "cart_prod: the point must be a column (n, 1), got the shape " +
               shape_text(p.shape());
    });
    const int64_t n2 = p.shape()[p.ndim() - 2];
    // the point has no generators: its rows are zero
    return {Tensor::concat({c_, p}, -2),
            Tensor::concat({G_, aux_zeros(G_, n2, G_.shape().back())}, -2)};
}

Zonotope Zonotope::cart_prod(const Tensor &p, const Zonotope &Z) {
    detail::Entry entry;
    CORA_CHECK(p.ndim() >= 2 && p.shape().back() == 1, Error::dimension_mismatch, [&] {
        return "cart_prod: the point must be a column (n, 1), got the shape " +
               shape_text(p.shape());
    });
    const int64_t n1 = p.shape()[p.ndim() - 2];
    // the point has no generators: its rows are zero
    return {Tensor::concat({p, Z.center()}, -2),
            Tensor::concat({aux_zeros(Z.generators(), n1, Z.generators().shape().back()),
                            Z.generators()}, -2)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
