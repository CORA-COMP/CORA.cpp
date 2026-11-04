// cart_prod - the Cartesian product of a constrained zonotope with a set or a point
//
// Syntax:   cZres = cz.cart_prod(S);   cZres = ConZonotope::cart_prod(p, cz);
// Inputs:   S - a constrained zonotope, a zonotope, a box or a point (n2, 1) of any dimension;
//           as the first operand, the point p (n1, 1); one batch shape
// Outputs:  cZres - exact: the centers stack, the generators and the constraints sit on the
//           block diagonal; of dimension n + n2, in the order of the operands
// See also: intersect, Zonotope::cart_prod
//
// CORA: contSet/@conZonotope/cartProd_.m

#include "cont_set/con_zonotope/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The leading (batch) axes of a tensor.
std::vector<int64_t> aux_batch(const Tensor &t) {
    const std::vector<int64_t> shape = t.shape();
    return {shape.begin(), shape.end() - 2};
}

/// Why the product is refused, if it is: the sets must have one batch shape.
detail::Refusal aux_refusal(const ConZonotope &cz1, const ConZonotope &cz2) {
    if (aux_batch(cz1.c()) == aux_batch(cz2.c())) return std::nullopt;
    return std::pair{Error::dimension_mismatch,
                     "cart_prod: the sets must have the same batch shape, got " +
                         shape_text(cz1.c().shape()) + " and " + shape_text(cz2.c().shape())};
}

/// The point p as a set without generators; it must be a column.
ConZonotope aux_point(const Tensor &p) { return {p, priv_zeros(p, p.shape()[p.ndim() - 2], 0)}; }

/// Why the point is refused, if it is.
detail::Refusal aux_point_refusal(const Tensor &p) {
    if (p.ndim() >= 2 && p.shape().back() == 1) return std::nullopt;
    return std::pair{Error::dimension_mismatch,
                     "cart_prod: the point must be a column (n, 1), got the shape " +
                         shape_text(p.shape())};
}

/// The product of two sets: centers stack, generators and constraints on the block diagonal.
ConZonotope aux_product(const ConZonotope &cz1, const ConZonotope &cz2) {
    return {Tensor::concat({cz1.c(), cz2.c()}, -2),
            priv_block_diag(cz1.generators(), cz2.generators()), priv_block_diag(cz1.A(), cz2.A()),
            Tensor::concat({cz1.b(), cz2.b()}, -2)};
}

} // namespace


// ===========================================  MAIN  =========================================== //

ConZonotope ConZonotope::cart_prod(const ConZonotope &S) const {
    detail::Entry entry;
    const detail::Refusal refusal = aux_refusal(*this, S);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return aux_product(*this, S);
}

ConZonotope ConZonotope::cart_prod(const Zonotope &Z) const {
    detail::Entry entry;
    const ConZonotope S(Z);
    const detail::Refusal refusal = aux_refusal(*this, S);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return aux_product(*this, S);
}

ConZonotope ConZonotope::cart_prod(const Interval &I) const {
    detail::Entry entry;
    const ConZonotope S(I);
    const detail::Refusal refusal = aux_refusal(*this, S);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return aux_product(*this, S);
}

ConZonotope ConZonotope::cart_prod(const Tensor &p) const {
    // the refusal of a point and of the dimensions comes before the product
    detail::Entry entry;
    detail::Refusal refusal = aux_point_refusal(p);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    // a point has no generators
    const ConZonotope S = aux_point(p);
    refusal = aux_refusal(*this, S);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return aux_product(*this, S);
}

ConZonotope ConZonotope::cart_prod(const Tensor &p, const ConZonotope &cz) {
    detail::Entry entry;
    detail::Refusal refusal = aux_point_refusal(p);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    const ConZonotope S = aux_point(p);
    refusal = aux_refusal(S, cz);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return aux_product(S, cz);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
