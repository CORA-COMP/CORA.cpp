// cart_prod - the Cartesian product of a polytope with a polytope, a box or a point
//
// Syntax:   Pres = P.cart_prod(P2);   Pres = P.cart_prod(I);   Pres = P.cart_prod(p);
//           Pres = Polytope::cart_prod(p, P);
// Inputs:   P2, I - a polytope or a box;  p - a point (n2, 1) of any dimension, as the second
//           operand or, with the static form, the first; one batch shape
// Outputs:  Pres - exact, of dimension n + n2, in the order of the operands: the halfspaces on
//           the block diagonal (a vertex polytope is converted by constraints first), or all
//           pairs of vertices if neither operand has halfspaces. A point is the equalities
//           I x = p. A batch needs the halfspaces (or the vertices of both) of one shape
// See also: intersect, Zonotope::cart_prod
//
// CORA: contSet/@polytope/cartProd_.m
// CORA: contSet/@polytope/private/priv_cartProd.m

#include "cont_set/polytope/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Shapes and refusals -----------------------------------------------------------------------------

/// The tensor that holds the data of the polytope: its halfspaces' matrix or its vertices.
const Tensor &aux_data(const Polytope &P) { return P.is_h_rep() ? P.A() : P.V(); }

/// The leading (batch) axes of the data.
std::vector<int64_t> aux_batch(const Polytope &P) {
    const std::vector<int64_t> shape = aux_data(P).shape();
    return {shape.begin(), shape.end() - 2};
}

/// Why the product is refused, if it is: the polytopes must have one batch shape.
detail::Refusal aux_refusal(const Polytope &P1, const Polytope &P2) {
    if (aux_batch(P1) == aux_batch(P2)) return std::nullopt;
    return std::pair{Error::dimension_mismatch,
                     "cart_prod: the sets must have the same batch shape, got " +
                         shape_text(aux_data(P1).shape()) + " and " +
                         shape_text(aux_data(P2).shape())};
}

/// Why the point is refused, if it is: it must be one column.
detail::Refusal aux_point_refusal(const Tensor &p) {
    if (!p.defined() || p.ndim() < 2)
        return std::pair{Error::dimension_mismatch, "cart_prod: the point must be a column (n, 1)"};
    if (p.shape().back() != 1)
        return std::pair{
            Error::not_supported,
            "cart_prod: the Cartesian product is only supported for single points, got "
            "the shape " +
                shape_text(p.shape())};
    return std::nullopt;
}

// The product -------------------------------------------------------------------------------------

/// The point as a polytope: the equalities I x = p.
Polytope aux_point(const Tensor &p) {
    const int64_t n = p.shape()[p.ndim() - 2];
    return {priv_zeros(p, 0, n), priv_zeros(p, 0, 1), priv_identity(p, n), p};
}

/// The product: the pairs of vertices if no operand has halfspaces, else the block diagonal.
Polytope aux_product(const Polytope &P1, const Polytope &P2) {
    if (!P1.is_h_rep() && !P2.is_h_rep()) {
        const int64_t n1 = P1.dim(), n2 = P2.dim(), m1 = P1.V().shape().back(),
                      m2 = P2.V().shape().back();
        if (m1 * m2 == 0) return Polytope(priv_zeros(P1.V(), n1 + n2, 0));
        // [repelem(V1, 1, m2); repmat(V2, 1, m1)]
        std::vector<int64_t> first(std::size_t(m1 * m2)), second(std::size_t(m1 * m2));
        for (int64_t k = 0; k < m1 * m2; ++k) {
            first[std::size_t(k)] = k / m2;
            second[std::size_t(k)] = k % m2;
        }
        return Polytope(Tensor::concat({P1.V().take(first, -1), P2.V().take(second, -1)}, -2));
    }
    const Polytope::Constraints C1 = P1.constraints(), C2 = P2.constraints();
    return {priv_block_diag(C1.A, C2.A), Tensor::concat({C1.b, C2.b}, -2),
            priv_block_diag(C1.Ae, C2.Ae), Tensor::concat({C1.be, C2.be}, -2)};
}

} // namespace


// ===========================================  MAIN  =========================================== //

// With a polytope ---------------------------------------------------------------------------------

Polytope Polytope::cart_prod(const Polytope &P2) const {
    detail::Entry entry;
    const detail::Refusal refusal = aux_refusal(*this, P2);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return aux_product(*this, P2);
}

Polytope Polytope::cart_prod(const Interval &I) const {
    detail::Entry entry;
    // a box is the polytope of its bounds
    const Polytope P2 = priv_polytope_of_interval(I);
    const detail::Refusal refusal = aux_refusal(*this, P2);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return aux_product(*this, P2);
}

// With a point ------------------------------------------------------------------------------------

Polytope Polytope::cart_prod(const Tensor &p) const {
    detail::Entry entry;
    detail::Refusal refusal = aux_point_refusal(p);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    const Polytope P2 = aux_point(p);
    refusal = aux_refusal(*this, P2);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return aux_product(*this, P2);
}

Polytope Polytope::cart_prod(const Tensor &p, const Polytope &P) {
    detail::Entry entry;
    detail::Refusal refusal = aux_point_refusal(p);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    const Polytope P1 = aux_point(p);
    refusal = aux_refusal(P1, P);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return aux_product(P1, P);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
