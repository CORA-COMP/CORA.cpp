// cart_prod - the Cartesian product of a constrained zonotope with a set or a point
//
// The centers stack, the generators and the constraints sit on the block diagonal. As the first
// operand the point p is (n1); one batch shape.
//
// Syntax:   cZres = cz.cart_prod(S);   cZres = ConZonotope::cart_prod(p, cz);
// Inputs:   S - a constrained zonotope, a zonotope, a box or a point (n2), of any dimension
// Outputs:  cZres - exact, of dimension n + n2, in the order of the operands
// See also: intersect, Zonotope::cart_prod
//
// CORA: contSet/@conZonotope/cartProd_.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Tobias Ladner, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/con_zonotope/private/priv.h"
#include "global/blocks.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Refusals ----------------------------------------------------------------------------------------

/// The leading (batch) axes of a tensor.
std::vector<int64_t> aux_batch(const Tensor &t) {
    const std::vector<int64_t> shape = t.shape();
    return {shape.begin(), shape.end() - 2};
}

/// Why the product is refused, if it is: the sets must have one batch shape.
detail::Refusal aux_refusal(const ConZonotope &cz1, const ConZonotope &cz2) {
    if (aux_batch(cz1.c().expand_dims(-1)) == aux_batch(cz2.c().expand_dims(-1)))
        return std::nullopt;
    return std::pair{Error::dimension_mismatch,
                     "cart_prod: the sets must have the same batch shape, got " +
                         shape_text(cz1.c().shape()) + " and " + shape_text(cz2.c().shape())};
}

/// The point p (n) as a set without generators.
ConZonotope aux_point(const Tensor &p) {
    return {p, zeros_block(p.expand_dims(-1), 0, p.size(-1))};
}

/// Why the point is refused, if it is.
detail::Refusal aux_point_refusal(const Tensor &p) {
    if (p.ndim() >= 1) return std::nullopt;
    return std::pair{Error::dimension_mismatch,
                     "cart_prod: the point must be a vector (n), got the shape " +
                         shape_text(p.shape())};
}

// The product -------------------------------------------------------------------------------------

/// The product of two sets: centers stack, generators and constraints on the block diagonal.
ConZonotope aux_product(const ConZonotope &cz1, const ConZonotope &cz2) {
    return {Tensor::concat({cz1.c(), cz2.c()}, -1),
            (Tensor::block_diag(
                 {cz1.generators().matrix_transpose(), cz2.generators().matrix_transpose()}))
                .matrix_transpose(),
            Tensor::block_diag({cz1.A(), cz2.A()}), Tensor::concat({cz1.b(), cz2.b()}, -1)};
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

// the point is refused before the product is formed
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
